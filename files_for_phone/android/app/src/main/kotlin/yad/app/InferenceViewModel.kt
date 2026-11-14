package yad.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import yad.app.hand.SimulatedHand
import yad.app.ml.TfliteFeatureExtractor
import yad.core.CommandGate
import yad.core.DecisionPolicy
import yad.core.Gesture
import yad.core.GroupAverageDecisionPolicy
import yad.core.HandCommand
import yad.core.MlpHead
import yad.core.ReplayRecording
import yad.core.ScalarFeatures
import java.nio.MappedByteBuffer
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executors

data class LogEntry(val time: String, val source: String, val command: HandCommand, val blocked: Boolean)

data class UiState(
    val models: List<String> = emptyList(),
    val model: String? = null,
    val status: String = "Loading…",
    val playing: Boolean = false,
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
    val decisions: Int = 0,
    val decisionCorrect: Int = 0,
    val cnnMs: Double = 0.0,
    val headMs: Double = 0.0,
    val handPosition: HandCommand? = null,
    val log: List<LogEntry> = emptyList(),
    val benchmark: String? = null,
    val selfTest: String? = null,
    val selfTestPassed: Boolean? = null,
) {
    /** Time from the start of a group until its decision, in Pi-compatible windows of 200 ms. */
    val decisionEveryMs: Int get() = groupSize * WINDOW_MS
}

const val WINDOW_MS = 200            // 100 samples at 500 Hz
const val PI_THRESHOLD = 0.7
const val PI_GROUP_SIZE = 6
private const val THREADS = 4
private const val LOG_SIZE = 30
private const val TIMING_WINDOW = 50

/**
 * M3: plays a bundled recording through the real pipeline (TFLite CNN + exported head + the
 * Pi's decision policy) into a simulated hand. All model work runs on one background thread.
 */
class InferenceViewModel(app: Application) : AndroidViewModel(app) {
    private val catalog = ModelCatalog(app)
    private val worker = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val hand = SimulatedHand()
    private val gate = CommandGate()

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    // Owned by the worker thread.
    private var modelBuffer: MappedByteBuffer? = null
    private var extractor: TfliteFeatureExtractor? = null
    private var head: MlpHead? = null
    private var replay: ReplayRecording? = null
    private var policy: DecisionPolicy = GroupAverageDecisionPolicy(PI_GROUP_SIZE, PI_THRESHOLD)
    private var policyKey = PI_GROUP_SIZE to PI_THRESHOLD
    private var cursor = 0
    private val cnnTimes = ArrayDeque<Double>()
    private val headTimes = ArrayDeque<Double>()
    private var loop: Job? = null

    init {
        viewModelScope.launch(worker) {
            try {
                modelBuffer = TfliteFeatureExtractor.mapAsset(app)
                extractor = TfliteFeatureExtractor(modelBuffer!!, THREADS)
                val names = catalog.names
                _state.update { it.copy(models = names) }
                if ("tomer" in names) {
                    val check = SelfTest.run(app, extractor!!, catalog.head("tomer"))
                    _state.update { it.copy(selfTest = check.summary, selfTestPassed = check.passed) }
                }
                val first = if ("tomer" in names) "tomer" else names.firstOrNull()
                if (first == null) _state.update { it.copy(status = "No models bundled") } else loadModel(first)
            } catch (e: Exception) {
                _state.update { it.copy(status = "Failed to load: ${e.message}") }
            }
        }
        viewModelScope.launch { hand.position.collect { p -> _state.update { it.copy(handPosition = p) } } }
    }

    fun selectModel(name: String) {
        pause()
        viewModelScope.launch(worker) { loadModel(name) }
    }

    private fun loadModel(name: String) {
        val h = catalog.head(name)
        val r = catalog.replay(name)
        head = h
        replay = r
        cursor = 0
        policy.reset()
        cnnTimes.clear(); headTimes.clear()
        _state.update {
            it.copy(
                model = name,
                status = "$name: ${r.size} recorded windows (${r.size * WINDOW_MS / 1000} s), features: RMS over ${h.rmsOver.jsonName}",
                decided = null, confidence = 0.0, windowGesture = null, trueGesture = null,
                windows = 0, windowCorrect = 0, decisions = 0, decisionCorrect = 0, benchmark = null,
            )
        }
    }

    fun play() {
        if (loop?.isActive == true) return
        _state.update { it.copy(playing = true) }
        loop = viewModelScope.launch(worker) {
            while (isActive) {
                val start = System.nanoTime()
                step()
                if (_state.value.realTime) {
                    val elapsedMs = (System.nanoTime() - start) / 1_000_000
                    delay((WINDOW_MS - elapsedMs).coerceAtLeast(0))
                } else {
                    yield()
                }
            }
        }
    }

    fun pause() {
        loop?.cancel()
        _state.update { it.copy(playing = false) }
    }

    private suspend fun step() {
        val r = replay ?: return
        val h = head ?: return
        val fx = extractor ?: return
        val s = _state.value
        if ((s.groupSize to s.threshold) != policyKey) {
            policy = GroupAverageDecisionPolicy(s.groupSize, s.threshold)
            policyKey = s.groupSize to s.threshold
        }
        gate.sendOnlyOnChange = s.sendOnlyOnChange

        val window = r.windows[cursor]
        val label = Gesture.fromId(r.labels[cursor])
        cursor = (cursor + 1) % r.size

        val t0 = System.nanoTime()
        val cnn = fx.extract(window)
        val t1 = System.nanoTime()
        val probs = h.predictProba(ScalarFeatures.headInput(cnn, window, h.rmsOver))
        val t2 = System.nanoTime()
        cnnTimes.addTiming((t1 - t0) / 1e6)
        headTimes.addTiming((t2 - t1) / 1e6)

        val windowGesture = Gesture.fromId(h.classes[probs.indices.maxBy { probs[it] }])
        val decision = policy.offer(probs)
        var entry: LogEntry? = null
        var decided = s.decided
        var confidence = s.confidence
        var decisions = s.decisions
        var decisionCorrect = s.decisionCorrect
        if (decision != null) {
            val g = Gesture.fromId(h.classes[decision.classIndex])
            decided = g
            confidence = decision.confidence
            decisions++
            if (g == label) decisionCorrect++
            if (g != null) {
                val cmd = gate.fromClassifier(g)
                if (cmd != null) {
                    hand.send(cmd)
                    entry = LogEntry(now(), "classifier", cmd, blocked = false)
                } else if (gate.stopped && g.command != null) {
                    entry = LogEntry(now(), "classifier", g.command!!, blocked = true)
                }
            }
        }
        _state.update {
            it.copy(
                decided = decided, confidence = confidence,
                windowGesture = windowGesture, trueGesture = label,
                windows = it.windows + 1,
                windowCorrect = it.windowCorrect + if (windowGesture == label) 1 else 0,
                decisions = decisions, decisionCorrect = decisionCorrect,
                cnnMs = cnnTimes.average(), headMs = headTimes.average(),
                log = if (entry != null) (listOf(entry) + it.log).take(LOG_SIZE) else it.log,
            )
        }
    }

    /** Stop: classifier commands are blocked immediately; manual buttons keep working. */
    fun toggleStop() {
        if (gate.stopped) gate.resume() else gate.stop()
        _state.update { it.copy(stopped = gate.stopped) }
    }

    fun manual(command: HandCommand) {
        viewModelScope.launch {
            hand.send(gate.manual(command))
            _state.update { it.copy(log = (listOf(LogEntry(now(), "manual", command, false)) + it.log).take(LOG_SIZE)) }
        }
    }

    fun setThreshold(v: Double) = _state.update { it.copy(threshold = v) }
    fun setGroupSize(n: Int) = _state.update { it.copy(groupSize = n) }
    fun setSendOnlyOnChange(on: Boolean) = _state.update { it.copy(sendOnlyOnChange = on) }
    fun setRealTime(on: Boolean) = _state.update { it.copy(realTime = on) }
    fun piPreset() = _state.update { it.copy(threshold = PI_THRESHOLD, groupSize = PI_GROUP_SIZE, sendOnlyOnChange = false) }

    /** Times the CNN alone on this phone with 1, 2 and 4 threads. */
    fun benchmark() {
        pause()
        _state.update { it.copy(benchmark = "Running…") }
        viewModelScope.launch(worker) {
            val buf = modelBuffer ?: return@launch
            val windows = replay?.windows ?: return@launch
            val lines = listOf(1, 2, 4).map { threads ->
                TfliteFeatureExtractor(buf, threads).use { fx ->
                    repeat(5) { fx.extract(windows[it % windows.size]) }        // warm-up
                    val n = 40
                    val t0 = System.nanoTime()
                    repeat(n) { fx.extract(windows[it % windows.size]) }
                    "%d thread%s: %.1f ms/window".format(threads, if (threads > 1) "s" else "", (System.nanoTime() - t0) / 1e6 / n)
                }
            }
            _state.update { it.copy(benchmark = lines.joinToString("\n") + "\n(budget: $WINDOW_MS ms per window)") }
        }
    }

    override fun onCleared() {
        loop?.cancel()
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
