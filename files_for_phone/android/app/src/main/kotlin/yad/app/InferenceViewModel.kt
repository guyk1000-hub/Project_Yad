package yad.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.yield
import yad.app.armband.ArmbandReceiver
import yad.app.hand.Haifa3dBleHand
import yad.app.hand.SimulatedHand
import yad.app.ml.Engine
import yad.app.ml.TfliteFeatureExtractor
import yad.core.ChannelLevels
import yad.core.CommandGate
import yad.core.DecisionPolicy
import yad.core.EmgWindow
import yad.core.Calibration
import yad.core.FilterChain
import yad.core.MlpTrainer
import yad.core.TrainingSet
import yad.core.GuidedCheck
import yad.core.Gesture
import yad.core.GroupAverageDecisionPolicy
import yad.core.GroupScorer
import yad.core.HandCommand
import yad.core.MlpHead
import yad.core.ReplayRecording
import yad.core.ScalarFeatures
import yad.core.Windower
import java.nio.MappedByteBuffer
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executors

enum class Source { REPLAY, ARMBAND }

data class RecordingInfo(val fileName: String, val seconds: Int)

data class LogEntry(val time: String, val source: String, val command: HandCommand, val outcome: String)

data class UiState(
    val models: List<String> = emptyList(),
    val model: String? = null,
    val status: String = "Loading…",
    val source: Source = Source.REPLAY,
    val running: Boolean = false,
    val realTime: Boolean = true,
    val stopped: Boolean = false,
    val threshold: Double = PI_THRESHOLD,
    val groupSize: Int = PI_GROUP_SIZE,
    val sendOnlyOnChange: Boolean = false,
    val decided: Gesture? = null,
    val confidence: Double = 0.0,
    val windowGesture: Gesture? = null,
    val trueGesture: Gesture? = null,
    val windows: Int = 0,
    val windowCorrect: Int = 0,
    val scoring: GroupScorer.Totals = GroupScorer.Totals(),
    val steadyAccuracy: Double = 0.0,
    val cnnMs: Double = 0.0,
    val headMs: Double = 0.0,
    // armband
    val armband: ArmbandReceiver.Status = ArmbandReceiver.Status(),
    val acceptBroadcast: Boolean = false,
    val lagMs: Int = 0,
    val liveLevels: DoubleArray? = null,
    val restReference: DoubleArray? = null,
    // hand
    val realHand: Boolean = false,
    val handState: String = "simulated",
    val simulatedPosition: HandCommand? = null,
    val log: List<LogEntry> = emptyList(),
    // checks
    val benchmark: List<String> = emptyList(),
    val benchmarkRunning: Boolean = false,
    val selfTest: String? = null,
    val selfTestPassed: Boolean? = null,
    val captureMessage: String? = null,
    // guided check
    val checkPrompt: String? = null,
    val checkStep: String? = null,
    val checkResults: List<String> = emptyList(),
    val checkStatus: String? = null,
    val trainingStatus: String? = null,
    val recordings: List<RecordingInfo> = emptyList(),
    val newModelLabel: String = "",
) {
    val decisionEveryMs: Int get() = groupSize * WINDOW_MS

    /** Live signal level vs the user's recorded rest level: the units check for hardware session A. */
    val levelVerdict: String?
        get() {
            val live = liveLevels ?: return null
            val ref = restReference ?: return null
            val ratios = live.indices.map { live[it] / ref[it] }.sorted()
            val median = (ratios[3] + ratios[4]) / 2
            return when {
                median in 0.33..3.0 -> "✓ same scale as the recording (×%.2f)".format(median)
                median in 0.1..10.0 -> "⚠ ×%.2f the recorded rest level: check contact/placement, and rest the arm".format(median)
                else -> "✗ ×%.2g the recorded level: probably a units problem — tell me".format(median)
            }
        }
}

const val WINDOW_MS = 200            // 100 samples at 500 Hz
const val SAMPLE_RATE = 500
const val PI_THRESHOLD = 0.7

fun displayName(name: String): String = yad.core.ModelNames.display(name)

const val PI_GROUP_SIZE = 6
private const val LOG_SIZE = 40
private const val TIMING_WINDOW = 50

/**
 * Runs the pipeline from either a bundled recording (REPLAY) or the live armband (ARMBAND) into
 * the simulated or real hand. All model work runs on one background thread; hand commands go
 * through their own queue so a slow BLE write never stalls the signal path.
 */
class InferenceViewModel(app: Application) : AndroidViewModel(app) {
    private val catalog = ModelCatalog(app)
    private val worker = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val simulatedHand = SimulatedHand()
    private val realHand = Haifa3dBleHand(app)
    private val receiver = ArmbandReceiver(app)
    private val gate = CommandGate()

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    /** Classifier commands, newest kept if the hand falls behind. */
    private val handQueue = Channel<HandCommand>(capacity = 4, onBufferOverflow = BufferOverflow.DROP_OLDEST)

    // Owned by the worker thread.
    private var modelBuffer: MappedByteBuffer? = null
    private var extractor: TfliteFeatureExtractor? = null
    private var golden: GoldenCheck? = null
    private var head: MlpHead? = null
    private var replay: ReplayRecording? = null
    private var policy: DecisionPolicy = GroupAverageDecisionPolicy(PI_GROUP_SIZE, PI_THRESHOLD)
    private var policyKey = PI_GROUP_SIZE to PI_THRESHOLD
    private var scorer = GroupScorer(PI_GROUP_SIZE)
    private val filters = FilterChain.pi()
    private val windower = Windower()
    private val levels = ChannelLevels()
    private var cursor = 0
    private val cnnTimes = ArrayDeque<Double>()
    private val headTimes = ArrayDeque<Double>()
    private var loop: Job? = null

    init {
        viewModelScope.launch(worker) {
            try {
                modelBuffer = TfliteFeatureExtractor.mapAsset(app)
                extractor = TfliteFeatureExtractor(modelBuffer!!)
                val names = catalog.names
                _state.update { it.copy(models = names) }
                if ("tomer" in names) {
                    golden = GoldenCheck(app)
                    val check = golden!!.run(extractor!!, catalog.head("tomer"))
                    _state.update { it.copy(selfTest = "On-device check ${if (check.passed) "PASSED" else "FAILED"}: ${check.summary}", selfTestPassed = check.passed) }
                }
                val first = catalog.latestFor("tomer") ?: if ("tomer" in names) "tomer" else names.firstOrNull()
                if (first == null) _state.update { it.copy(status = "No models bundled") } else loadModel(first)
            } catch (e: Exception) {
                _state.update { it.copy(status = "Failed to load: ${e.message}") }
            }
        }
        viewModelScope.launch { simulatedHand.position.collect { p -> _state.update { it.copy(simulatedPosition = p) } } }
        viewModelScope.launch { receiver.status.collect { st -> _state.update { it.copy(armband = st) } } }
        viewModelScope.launch {
            realHand.state.collect { st ->
                val text = when (st) {
                    Haifa3dBleHand.State.Disconnected -> "disconnected"
                    is Haifa3dBleHand.State.Connecting -> st.step + "…"
                    Haifa3dBleHand.State.Ready -> "connected"
                    is Haifa3dBleHand.State.Failed -> "failed: ${st.reason}"
                }
                _state.update { if (it.realHand) it.copy(handState = text) else it }
            }
        }
        viewModelScope.launch(Dispatchers.Default) {
            for (cmd in handQueue) {
                // Re-check at send time: STOP may have been pressed after the decision was queued.
                if (gate.stopped) log("classifier", cmd, "blocked by STOP") else deliver(cmd, "classifier")
            }
        }
    }

    // ---- model & source ------------------------------------------------------------------

    fun selectModel(name: String) {
        stopRun()
        viewModelScope.launch(worker) { loadModel(name) }
    }

    private fun loadModel(name: String) {
        val h = catalog.head(name)
        val r = catalog.replay(name)
        head = h
        replay = r
        resetSignalPath()
        refreshRecordings(catalog.userOf(name))
        _state.update {
            it.copy(
                model = name,
                restReference = ChannelLevels.restReference(r),
                status = "$name: features RMS over ${h.rmsOver.jsonName}; replay ${r.size} windows (${r.size * WINDOW_MS / 1000} s)",
            )
        }
    }

    fun selectSource(source: Source) {
        stopRun()
        _state.update { it.copy(source = source) }
        viewModelScope.launch(worker) { resetSignalPath() }
    }

    fun requestEegMode() = receiver.requestEegMode()

    fun diagnosticsText(): String = receiver.diagnosticsText()

    /** Saves the next 5 s of raw armband packets to Downloads, for offline analysis of the packet format. */
    fun captureRawPackets() {
        receiver.captureRaw(5) { bytes, count ->
            val msg = saveToDownloads("yad_raw_packets_${timestamp()}.bin", bytes)
            _state.update { it.copy(captureMessage = "$count packets: $msg") }
        }
        _state.update { it.copy(captureMessage = "recording 5 s of raw packets…") }
    }

    private fun timestamp(): String =
        java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))

    /** Writes a file to the phone's Downloads folder (no storage permission needed on Android 10+). */
    private fun saveToDownloads(name: String, bytes: ByteArray): String = try {
        val app = getApplication<Application>()
        val values = android.content.ContentValues().apply {
            put(android.provider.MediaStore.Downloads.DISPLAY_NAME, name)
            put(android.provider.MediaStore.Downloads.MIME_TYPE, "application/octet-stream")
            put(android.provider.MediaStore.Downloads.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS)
        }
        val uri = app.contentResolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: error("couldn't create the file")
        app.contentResolver.openOutputStream(uri)!!.use { it.write(bytes) }
        "saved to Downloads/$name (${bytes.size / 1024} KB)"
    } catch (e: Exception) {
        "couldn't save $name: ${e.message}"
    }

    fun setAcceptBroadcast(on: Boolean) {
        val wasRunning = _state.value.running && _state.value.source == Source.ARMBAND
        _state.update { it.copy(acceptBroadcast = on) }
        if (wasRunning) { stopRun(); start() }
    }

    /** Worker thread only. */
    private fun resetSignalPath() {
        cursor = 0
        filters.reset()
        windower.reset()
        levels.clear()
        policy = GroupAverageDecisionPolicy(_state.value.groupSize, _state.value.threshold)
        policyKey = _state.value.groupSize to _state.value.threshold
        scorer = GroupScorer(policyKey.first)
        cnnTimes.clear(); headTimes.clear()
        _state.update {
            it.copy(
                decided = null, confidence = 0.0, windowGesture = null, trueGesture = null,
                windows = 0, windowCorrect = 0, scoring = GroupScorer.Totals(), steadyAccuracy = 0.0,
                liveLevels = null, lagMs = 0,
            )
        }
    }

    // ---- running -------------------------------------------------------------------------

    fun start() {
        if (loop?.isActive == true) return
        _state.update { it.copy(running = true) }
        loop = when (_state.value.source) {
            Source.REPLAY -> viewModelScope.launch(worker) { replayLoop() }
            Source.ARMBAND -> {
                receiver.start(viewModelScope, _state.value.acceptBroadcast)
                viewModelScope.launch(worker) { liveLoop() }
            }
        }
    }

    fun stopRun() {
        loop?.cancel()
        receiver.stop()
        _state.update { it.copy(running = false) }
    }

    private suspend fun replayLoop() {
        while (kotlinx.coroutines.currentCoroutineContext().isActive) {
            val r = replay ?: return
            val begin = System.nanoTime()
            val i = cursor
            cursor = (cursor + 1) % r.size
            handleWindow(r.windows[i], Gesture.fromId(r.labels[i]))
            if (_state.value.realTime) delay((WINDOW_MS - (System.nanoTime() - begin) / 1_000_000).coerceAtLeast(0)) else yield()
        }
    }

    private suspend fun liveLoop() {
        resetSignalPath()
        // Drain anything left from an earlier run.
        while (receiver.samples.tryReceive().isSuccess) receiver.backlog.decrementAndGet()
        receiver.backlog.set(0)
        while (kotlinx.coroutines.currentCoroutineContext().isActive) {
            val sample = withTimeoutOrNull(500) { receiver.samples.receive() } ?: continue
            val backlog = receiver.backlog.decrementAndGet()
            val c = check
            val phase = c?.let { phaseAt(it, it.sampleCount) }
            if (c != null && phase != null) c.addSample(sample.copyOf(), phase)   // raw, before filtering
            val window = windower.push(filters.process(sample)) ?: continue
            levels.add(window)
            val probs = handleWindow(window, null)
            if (c != null && phase != null && probs != null) {
                val h = head!!
                c.addWindow(phase, h.classes[probs.indices.maxBy { probs[it] }], probs, window.length)
            }
            _state.update { it.copy(lagMs = backlog * 1000 / SAMPLE_RATE, liveLevels = levels.mean()) }
            if (c != null) updateCheck(c)
        }
    }

    /** Worker thread: one filtered window through CNN, head, policy; then score and maybe command the hand. */
    private fun handleWindow(window: EmgWindow, label: Gesture?): DoubleArray? {
        val h = head ?: return null
        val fx = extractor ?: return null
        val s = _state.value
        if ((s.groupSize to s.threshold) != policyKey) {
            policy = GroupAverageDecisionPolicy(s.groupSize, s.threshold)
            policyKey = s.groupSize to s.threshold
            scorer = GroupScorer(s.groupSize)
        }
        gate.sendOnlyOnChange = s.sendOnlyOnChange

        val t0 = System.nanoTime()
        val cnn = fx.extract(window)
        val t1 = System.nanoTime()
        val probs = h.predictProba(ScalarFeatures.headInput(cnn, window, h.rmsOver))
        val t2 = System.nanoTime()
        cnnTimes.addTiming((t1 - t0) / 1e6)
        headTimes.addTiming((t2 - t1) / 1e6)

        val windowGesture = Gesture.fromId(h.classes[probs.indices.maxBy { probs[it] }])
        val decision = policy.offer(probs)
        val decidedGesture = decision?.let { Gesture.fromId(h.classes[it.classIndex]) }
        if (label != null) scorer.add(label.id, decidedGesture?.id)
        if (decidedGesture != null) {
            val cmd = gate.fromClassifier(decidedGesture)
            when {
                cmd != null -> handQueue.trySend(cmd)
                gate.stopped && decidedGesture.command != null -> log("classifier", decidedGesture.command!!, "blocked by STOP")
            }
        }
        _state.update {
            it.copy(
                decided = decidedGesture ?: it.decided,
                confidence = decision?.confidence ?: it.confidence,
                windowGesture = windowGesture, trueGesture = label,
                windows = it.windows + 1,
                windowCorrect = it.windowCorrect + if (label != null && windowGesture == label) 1 else 0,
                scoring = scorer.totals, steadyAccuracy = scorer.steadyAccuracy,
                cnnMs = cnnTimes.average(), headMs = headTimes.average(),
            )
        }
        return probs
    }

    // ---- guided check --------------------------------------------------------------------

    /** Worker thread only. */
    private var check: GuidedCheck? = null

    /** Phase index for the n-th sample of the check, or null once the check is over. Timed by samples, not the clock. */
    private fun phaseAt(c: GuidedCheck, n: Int): Int? {
        var start = 0
        for ((i, p) in c.phases.withIndex()) {
            val end = start + p.seconds * SAMPLE_RATE
            if (n < end) return i
            start = end
        }
        return null
    }

    /** Worker thread only: number of takes while the running GuidedCheck is a recording to train on, else 0. */
    private var recordingTakes = 0

    fun startCheck() = startGuided(takes = 0)

    /**
     * Records [takes] × 10 s of each gesture into the user's phone recordings, then trains the
     * user's model from zero on all of them (Calibration.RECORDING_*). 1 take = a quick calibration.
     */
    fun startRecording(takes: Int) = startGuided(takes)

    private fun startGuided(takes: Int) {
        val s = _state.value
        if (s.source != Source.ARMBAND || !s.running) {
            _state.update { it.copy(checkStatus = "Start the armband first (Signal: Armband, then Start armband).") }
            return
        }
        viewModelScope.launch(worker) {
            val h = head ?: return@launch
            recordingTakes = takes
            check = GuidedCheck(s.model ?: "?", h.rmsOver.jsonName, h.classes,
                phaseSeconds = if (takes > 0) Calibration.PHASE_SECONDS else GuidedCheck.PHASE_SECONDS,
                takes = maxOf(takes, 1))
            _state.update { it.copy(checkResults = emptyList(), checkStatus = null, trainingStatus = null) }
        }
    }

    /** Worker thread: show the current phase; when the last phase ends, save the file and show results. */
    private fun updateCheck(c: GuidedCheck) {
        val n = c.sampleCount
        val phase = phaseAt(c, n)
        if (phase != null) {
            var start = 0
            for (i in 0 until phase) start += c.phases[i].seconds * SAMPLE_RATE
            val left = (start + c.phases[phase].seconds * SAMPLE_RATE - n + SAMPLE_RATE - 1) / SAMPLE_RATE
            val p = c.phases[phase]
            val title = if (p.label < 0) p.title else "Hold ${p.title}"
            _state.update { it.copy(checkPrompt = "$title   ($left s)", checkStep = "${phase + 1}/${c.phases.size}") }
            return
        }
        check = null
        if (recordingTakes > 0) {
            val user = catalog.userOf(c.model)
            val bytes = c.toBytes()
            val fileName = "rec_${user}_${timestamp()}.bin"
            catalog.saveRecording(user, fileName, bytes)
            val saved = saveToDownloads(fileName, bytes)
            _state.update { it.copy(checkPrompt = null, checkStep = null, checkStatus = saved, trainingStatus = "Recording done. Preparing…") }
            refreshRecordings(user)
            stopRun()
            viewModelScope.launch(worker) { trainFromRecordings(user, afterNewRecording = true) }
            return
        }
        val results = c.results()
        val saved = saveToDownloads("yad_check_${c.model}_${timestamp()}.bin", c.toBytes())
        _state.update { it.copy(checkPrompt = null, checkStep = null, checkResults = results, checkStatus = saved) }
    }

    fun setNewModelLabel(label: String) = _state.update { it.copy(newModelLabel = label) }

    /** Retrains the current user's model on their phone recordings (no armband needed). */
    fun retrain() {
        val user = catalog.userOf(_state.value.model ?: return)
        stopRun()
        viewModelScope.launch(worker) { trainFromRecordings(user, afterNewRecording = false) }
    }

    private fun trainingStatus(msg: String) = _state.update { it.copy(trainingStatus = msg) }

    /**
     * Worker thread: trains [user]'s model from zero on all their phone recordings (not the PC
     * recording), saves it as a new "<user>-<label>@<time>" version (label from the name field) and selects it.
     */
    private fun trainFromRecordings(user: String, afterNewRecording: Boolean) {
        val fx = extractor ?: return
        try {
            val base = catalog.head(user)      // same features and classes as the user's bundled model
            val files = catalog.recordings(user)
            if (files.isEmpty()) { trainingStatus("No recordings for $user yet: tap Record."); return }
            val sets = files.mapIndexed { k, f ->
                recordingFeatures(f, base.rmsOver, fx) { pct -> trainingStatus("Computing features, recording ${k + 1}/${files.size}: $pct%") }
            }
            // An honest number, before retraining: the previous phone model on data it has never seen.
            val previous = catalog.latestPhoneTrained(user)
            val before = if (afterNewRecording && previous != null && files.size > 1)
                "${displayName(previous)} scored ${accuracyPercent(catalog.head(previous), sets.last())}% on this new recording. " else ""
            val data = sets.reduce { a, b -> a + b }
            val t0 = System.nanoTime()
            val result = MlpTrainer(maxIterations = Calibration.ITERATIONS).train(data) { iter, _ ->
                if (iter % 5 == 0) trainingStatus("Training on ${data.size} windows from ${files.size} recordings: ${iter * 100 / Calibration.ITERATIONS}%")
                true
            }
            val secs = (System.nanoTime() - t0) / 1_000_000_000
            val name = yad.core.ModelNames.phoneTrained(user, _state.value.newModelLabel, timestamp())
            val note = "trained on the phone from zero: ${files.size} recordings (${files.joinToString { it.name }}), " +
                "${data.size} windows, ${result.iterations} iterations"
            val json = result.toHeadJson(name, base.rmsOver, base.classNames, note)
            catalog.saveCalibrated(name, json)
            // A copy for the PC: tools/analyze_check.py needs it to rerun checks made with this version.
            saveToDownloads("yad_model_$name.json", json.toByteArray(Charsets.UTF_8))
            _state.update { it.copy(models = catalog.names) }
            loadModel(name)
            trainingStatus("✓ Saved ${displayName(name)}: ${files.size} recordings, ${data.size} windows, trained in $secs s. " +
                before + "Start the armband and run the guided check.")
        } catch (e: Exception) {
            trainingStatus("Training failed: ${e.message}")
        }
    }

    /** Worker thread: a recording's training windows as head inputs, cached in a .feat file next to it. */
    private fun recordingFeatures(f: java.io.File, rmsOver: yad.core.RmsOver, fx: TfliteFeatureExtractor, progress: (Int) -> Unit): TrainingSet {
        val cache = java.io.File(f.path.removeSuffix(".bin") + ".feat")
        if (cache.exists()) {
            val (set, rms) = TrainingSet.parse(cache.readBytes())
            if (rms == rmsOver) return set
        }
        val (raw, labels) = GuidedCheck.readSamples(f.readBytes())
        val windows = Calibration.windows(raw, labels, hop = Calibration.RECORDING_HOP)
        require(windows.isNotEmpty()) { "${f.name} has no gesture windows" }
        val features = windows.mapIndexed { i, (w, _) ->
            if (i % 20 == 0) progress(i * 100 / windows.size)
            ScalarFeatures.headInput(fx.extract(w), w, rmsOver)
        }
        val set = TrainingSet(features, windows.map { it.second }.toIntArray())
        cache.writeBytes(set.toBytes(rmsOver))
        return set
    }

    private fun accuracyPercent(h: MlpHead, set: TrainingSet): Int {
        val correct = set.features.indices.count { i ->
            val p = h.predictProba(set.features[i])
            h.classes[p.indices.maxBy { p[it] }] == set.labels[i]
        }
        return correct * 100 / set.size
    }

    /** Copies recordings the user picked (e.g. yad_calib_* / rec_* from Downloads) into the current user's recordings, then retrains. */
    fun importRecordings(uris: List<android.net.Uri>) {
        if (uris.isEmpty()) return
        val user = catalog.userOf(_state.value.model ?: return)
        stopRun()
        viewModelScope.launch(worker) {
            val resolver = getApplication<Application>().contentResolver
            val notes = mutableListOf<String>()
            var added = 0
            for (uri in uris) {
                val name = resolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                    ?.use { if (it.moveToFirst()) it.getString(0) else null } ?: "rec_${user}_${timestamp()}.bin"
                try {
                    val bytes = resolver.openInputStream(uri)!!.use { it.readBytes() }
                    GuidedCheck.readSamples(bytes)     // a Yad recording at 500 samples/s, or this throws
                    if (catalog.recordings(user).any { it.name == name }) notes += "$name: already imported"
                    else { catalog.saveRecording(user, name, bytes); added++ }
                } catch (e: Exception) {
                    notes += "$name: ${e.message}"
                }
            }
            refreshRecordings(user)
            trainingStatus("Imported $added for $user. " + notes.joinToString("; "))
            if (added > 0) trainFromRecordings(user, afterNewRecording = false)
        }
    }

    fun deleteRecording(fileName: String) {
        val user = catalog.userOf(_state.value.model ?: return)
        viewModelScope.launch(worker) {
            catalog.recordings(user).firstOrNull { it.name == fileName }?.let { catalog.deleteRecording(it) }
            refreshRecordings(user)
            trainingStatus("Deleted $fileName. Tap Retrain to update the model without it.")
        }
    }

    private fun refreshRecordings(user: String) {
        val list = catalog.recordings(user).map { f -> RecordingInfo(f.name, (f.length() / 65 / SAMPLE_RATE).toInt()) }
        _state.update { it.copy(recordings = list) }
    }

    /** Deletes a phone-made version and goes back to the newest remaining one for that user. */
    fun deleteModel(name: String) {
        if (!catalog.isPhoneMade(name)) return
        stopRun()
        viewModelScope.launch(worker) {
            val user = catalog.userOf(name)
            catalog.delete(name)
            _state.update { it.copy(models = catalog.names) }
            loadModel(catalog.latestFor(user) ?: user)
        }
    }

    // ---- hand ----------------------------------------------------------------------------

    /** STOP: classifier commands are blocked immediately; manual buttons keep working. */
    fun toggleStop() {
        if (gate.stopped) gate.resume() else gate.stop()
        _state.update { it.copy(stopped = gate.stopped) }
    }

    fun manual(command: HandCommand) {
        viewModelScope.launch(Dispatchers.Default) { deliver(gate.manual(command), "manual") }
    }

    /** Call only after BLUETOOTH_CONNECT / BLUETOOTH_SCAN are granted. */
    fun setRealHand(on: Boolean) {
        _state.update { it.copy(realHand = on, handState = if (on) "connecting…" else "simulated") }
        viewModelScope.launch(Dispatchers.Default) { if (on) realHand.connect() else realHand.disconnect() }
    }

    private suspend fun deliver(cmd: HandCommand, source: String) {
        if (!_state.value.realHand) {
            simulatedHand.send(cmd)
            log(source, cmd, "simulated")
            return
        }
        try {
            realHand.send(cmd)
            log(source, cmd, "sent to hand")
        } catch (e: Exception) {
            log(source, cmd, "NOT sent: ${e.message}")
        }
    }

    private fun log(source: String, cmd: HandCommand, outcome: String) =
        _state.update { it.copy(log = (listOf(LogEntry(now(), source, cmd, outcome)) + it.log).take(LOG_SIZE)) }

    // ---- settings ------------------------------------------------------------------------

    fun setThreshold(v: Double) = _state.update { it.copy(threshold = v) }
    fun setGroupSize(n: Int) = _state.update { it.copy(groupSize = n) }
    fun setSendOnlyOnChange(on: Boolean) = _state.update { it.copy(sendOnlyOnChange = on) }
    fun setRealTime(on: Boolean) = _state.update { it.copy(realTime = on) }
    fun piPreset() = _state.update { it.copy(threshold = PI_THRESHOLD, groupSize = PI_GROUP_SIZE, sendOnlyOnChange = false) }

    // ---- benchmark -----------------------------------------------------------------------

    /** Times the CNN on this phone for each engine, and checks each engine's accuracy on the golden windows. */
    fun benchmark() {
        stopRun()
        _state.update { it.copy(benchmark = emptyList(), benchmarkRunning = true) }
        viewModelScope.launch(worker) {
            val buf = modelBuffer
            val g = golden
            if (buf == null || g == null) {
                _state.update { it.copy(benchmark = listOf("needs the tomer model bundled"), benchmarkRunning = false) }
                return@launch
            }
            val tomer = catalog.head("tomer")
            for (engine in Engine.BENCHMARK) {
                val line = try {
                    TfliteFeatureExtractor(buf, engine).use { fx ->
                        repeat(5) { fx.extract(g.windows[it % g.windows.size]) }   // warm-up
                        val n = 40
                        val t0 = System.nanoTime()
                        repeat(n) { fx.extract(g.windows[it % g.windows.size]) }
                        val ms = (System.nanoTime() - t0) / 1e6 / n
                        val check = g.run(fx, tomer)
                        "%-30s %6.1f ms  %s err %.0e".format(engine.label, ms, if (check.passed) "✓" else "✗", check.featureError)
                    }
                } catch (t: Throwable) {
                    "%-30s not available (%s)".format(engine.label, t.message?.take(60) ?: t.javaClass.simpleName)
                }
                _state.update { it.copy(benchmark = it.benchmark + line) }
            }
            _state.update { it.copy(benchmark = it.benchmark + "budget: $WINDOW_MS ms per window", benchmarkRunning = false) }
        }
    }

    override fun onCleared() {
        loop?.cancel()
        receiver.stop()
        realHand.disconnect()
        extractor?.close()
        worker.close()
    }

    private fun ArrayDeque<Double>.addTiming(ms: Double) {
        addLast(ms)
        if (size > TIMING_WINDOW) removeFirst()
    }

    private companion object {
        val FMT: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS")
        fun now(): String = LocalTime.now().format(FMT)
    }
}
