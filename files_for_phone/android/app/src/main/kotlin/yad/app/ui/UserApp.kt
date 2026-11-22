@file:OptIn(ExperimentalLayoutApi::class)

package yad.app.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import yad.app.DecisionMode
import yad.app.InferenceViewModel
import yad.app.Source
import yad.app.UiState
import yad.core.Calibration
import yad.core.Gesture
import yad.core.GestureSet
import yad.core.HandModel
import kotlin.math.roundToInt

/**
 * The user view: pick a person, then four tabs. Home walks through the three things needed
 * (armband, a model, the hand) and then shows what the hand does; Train records, calibrates and
 * checks; Hand has the selected hand's controls; Settings keeps the few choices a user needs.
 */
@Composable
fun UserApp(s: UiState, vm: InferenceViewModel) {
    var picking by rememberSaveable { mutableStateOf(true) }
    var tab by rememberSaveable { mutableStateOf(0) }

    if (picking || s.person == null) {
        WelcomeScreen(s, vm, onPicked = { picking = false; tab = 0 })
    } else {
        Scaffold(
            bottomBar = {
                NavigationBar {
                    listOf("🏠" to "Home", "🎯" to "Train", "🤖" to "Hand", "⚙" to "Settings").forEachIndexed { i, (icon, label) ->
                        NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = { Text(icon, fontSize = 20.sp) }, label = { Text(label) })
                    }
                }
            },
        ) { padding ->
            Column(
                Modifier.padding(padding).verticalScroll(rememberScrollState()).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                when (tab) {
                    0 -> HomeTab(s, vm, onChangePerson = { picking = true }, onTrain = { tab = 1 }, onHand = { tab = 2 })
                    1 -> TrainTab(s, vm, onHome = { tab = 0 })
                    2 -> HandTab(s, vm)
                    else -> SettingsTab(s, vm)
                }
            }
        }
    }
    if (s.checkPrompt != null) GuidedOverlay(s, vm)
}

// ---- welcome & people ------------------------------------------------------------------------

@Composable
private fun WelcomeScreen(s: UiState, vm: InferenceViewModel, onPicked: () -> Unit) {
    var adding by remember { mutableStateOf(false) }
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Spacer(Modifier.height(24.dp))
        Text("✋", fontSize = 56.sp)
        Text("Project Yad", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
        Text("Move a prosthetic hand with your muscles. The armband reads them, this phone learns your gestures, and the hand follows.",
            style = MaterialTheme.typography.bodyLarge)
        Spacer(Modifier.height(8.dp))
        Text("Who is using the hand?", style = MaterialTheme.typography.titleLarge)
        s.people.forEach { p ->
            val model = s.peopleModels[p]
            Card(
                Modifier.fillMaxWidth().clickable { vm.selectPerson(p); onPicked() },
                colors = CardDefaults.cardColors(containerColor = if (p == s.person) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant),
            ) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    Avatar(p)
                    Column(Modifier.weight(1f)) {
                        Text(p.replaceFirstChar { it.uppercase() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                        Text(if (model != null) "Ready · model $model" else "New · needs a 3-minute recording", style = MaterialTheme.typography.bodySmall)
                    }
                    Text("›", fontSize = 28.sp)
                }
            }
        }
        OutlinedButton(onClick = { adding = true }, Modifier.fillMaxWidth().height(56.dp)) { Text("+  Add a new person", fontSize = 16.sp) }
        Spacer(Modifier.height(16.dp))
        TextButton(onClick = { vm.setEngineering(true) }) { Text("Engineering view") }
    }
    if (adding) AddPersonDialog(vm, onDone = { adding = false; onPicked() }, onCancel = { adding = false })
}

@Composable
private fun Avatar(name: String) {
    Box(Modifier.size(44.dp).background(MaterialTheme.colorScheme.primary, CircleShape), contentAlignment = Alignment.Center) {
        Text(name.take(1).uppercase(), color = MaterialTheme.colorScheme.onPrimary, fontWeight = FontWeight.Bold, fontSize = 20.sp)
    }
}

@Composable
private fun AddPersonDialog(vm: InferenceViewModel, onDone: () -> Unit, onCancel: () -> Unit) {
    var name by remember { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text("Add a new person") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it; error = null }, label = { Text("Name") }, singleLine = true)
                Text("Next you'll record their gestures once (about 3 minutes); the phone then learns them.", style = MaterialTheme.typography.bodySmall)
                error?.let { Text(it, color = Danger) }
            }
        },
        confirmButton = { TextButton(onClick = { error = vm.addPerson(name); if (error == null) onDone() }) { Text("Add") } },
        dismissButton = { TextButton(onClick = onCancel) { Text("Cancel") } },
    )
}

// ---- home --------------------------------------------------------------------------------------

private enum class StepState { TODO, BUSY, DONE }

@Composable
private fun HomeTab(s: UiState, vm: InferenceViewModel, onChangePerson: () -> Unit, onTrain: () -> Unit, onHand: () -> Unit) {
    val name = s.person!!.replaceFirstChar { it.uppercase() }
    var choosingModel by remember { mutableStateOf(false) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("Hi, $name", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Text("Three steps, then you're ready.", style = MaterialTheme.typography.bodyMedium)
        }
        TextButton(onClick = onChangePerson) { Text("Change person") }
    }

    val armbandLive = s.source == Source.ARMBAND && s.running
    val receiving = armbandLive && s.armband.samplesPerSecond > 400
    Step(
        1, "Armband",
        when { receiving -> StepState.DONE; armbandLive -> StepState.BUSY; else -> StepState.TODO },
        when {
            receiving -> "Connected · ${s.armband.samplesPerSecond.roundToInt()} readings a second"
            armbandLive -> "Waiting for the armband… Is it switched on, and is this phone on its WiFi network? If it was on before, switch it off and on."
            else -> "Put the armband on your forearm and switch it on. Connect this phone to the armband's WiFi, then tap Connect."
        },
    ) {
        if (armbandLive) OutlinedButton(onClick = vm::stopRun) { Text("Disconnect") }
        else Button(onClick = { connectArmband(s, vm) }) { Text("Connect") }
    }

    Step(
        2, "Your gestures",
        if (s.model != null) StepState.DONE else StepState.TODO,
        if (s.model != null) "Model ready (${yad.app.displayName(s.model)}). Just put the armband on? A 1-minute calibration makes it more accurate."
        else "The phone doesn't know your gestures yet. Record them once: about 3 minutes.",
    ) {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (s.model != null) OutlinedButton(onClick = onTrain) { Text("Calibrate (1 min)") }
            else Button(onClick = onTrain) { Text("Record my gestures") }
            if (personModels(s).isNotEmpty()) {
                OutlinedButton(onClick = { choosingModel = true }) { Text("Models (${personModels(s).size})") }
            }
        }
    }
    if (choosingModel) ModelChooser(s, vm, onDone = { choosingModel = false })

    val handReady = s.realHand && s.handState == "connected"
    Step(
        3, "Hand",
        when { handReady -> StepState.DONE; s.realHand -> StepState.BUSY; else -> StepState.TODO },
        when {
            handReady -> "${s.handModel.title} connected. It moves when you make a gesture."
            s.realHand -> "Connecting to the ${s.handModel.title.lowercase()}: ${s.handState}" +
                if (s.handModel == HandModel.CONTROLLER_V2) "\nTurn the hand on now (or restart it): it finds the phone by itself." else ""
            else -> "Choose your hand and connect it. Without a hand, the app just shows what it would do."
        },
    ) {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            HandModel.entries.forEach { m -> FilterChip(selected = s.handModel == m, onClick = { vm.selectHandModel(m) }, label = { Text(m.title) }) }
        }
        HandConnectButton(s, vm)
    }

    if (armbandLive && s.model != null) LiveCard(s, vm, onHand)
}

private fun connectArmband(s: UiState, vm: InferenceViewModel) {
    if (s.source != Source.ARMBAND) vm.selectSource(Source.ARMBAND)
    vm.start()
}

@Composable
private fun Step(n: Int, title: String, state: StepState, text: String, actions: @Composable () -> Unit) {
    val color = when (state) { StepState.DONE -> Connected; StepState.BUSY -> Waiting; StepState.TODO -> MaterialTheme.colorScheme.primary }
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            Box(Modifier.size(36.dp).background(color, CircleShape), contentAlignment = Alignment.Center) {
                Text(if (state == StepState.DONE) "✓" else "$n", color = Color.White, fontWeight = FontWeight.Bold)
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(text, style = MaterialTheme.typography.bodyMedium)
                actions()
            }
        }
    }
}

/** Asks for Bluetooth permission when needed, then connects or disconnects the real hand. */
@Composable
private fun HandConnectButton(s: UiState, vm: InferenceViewModel) {
    val context = LocalContext.current
    val perms = arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_ADVERTISE)
    var denied by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { r ->
        denied = !r.values.all { it }
        if (!denied) vm.setRealHand(true)
    }
    if (s.realHand) OutlinedButton(onClick = { vm.setRealHand(false) }) { Text("Disconnect hand") }
    else Button(onClick = {
        if (perms.all { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }) vm.setRealHand(true) else launcher.launch(perms)
    }) { Text("Connect hand") }
    if (denied) Text("Bluetooth permission is needed to connect the hand.", color = Danger)
}

internal fun gestureIcon(g: Gesture?): String = when (g) {
    Gesture.REST -> "😌"; Gesture.CLOSE -> "✊"; Gesture.OPEN -> "🖐"; Gesture.RIGHT -> "↪"; Gesture.LEFT -> "↩"
    Gesture.POINT -> "☝"; Gesture.PEACE -> "✌"; null -> "…"
}

@Composable
private fun LiveCard(s: UiState, vm: InferenceViewModel, onHand: () -> Unit) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
        Column(Modifier.padding(20.dp).fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Now", style = MaterialTheme.typography.labelLarge)
            Text(gestureIcon(s.decided), fontSize = 72.sp)
            Text(s.decided?.name ?: "Make a gesture", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            LinearProgressIndicator(progress = { s.confidence.toFloat() }, modifier = Modifier.fillMaxWidth().height(8.dp))
            Text("sure: ${(s.confidence * 100).roundToInt()}%", style = MaterialTheme.typography.bodySmall)
            Button(
                onClick = vm::toggleStop, modifier = Modifier.fillMaxWidth().height(72.dp),
                colors = ButtonDefaults.buttonColors(containerColor = if (s.stopped) Connected else Danger),
            ) { Text(if (s.stopped) "RESUME" else "STOP", fontSize = 26.sp, fontWeight = FontWeight.Bold) }
            if (!s.realHand) SimulatedHandView(s, vm)
            Text(
                when {
                    s.stopped -> "Stopped: gestures don't move the hand. Buttons in the Hand tab still work."
                    s.realHand -> "Your gestures move the hand."
                    else -> "No hand connected: this shows what the hand would do."
                },
                style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center,
            )
            TextButton(onClick = onHand) { Text("Hand controls ›") }
        }
    }
}

// ---- train -------------------------------------------------------------------------------------

@Composable
private fun TrainTab(s: UiState, vm: InferenceViewModel, onHome: () -> Unit) {
    val live = s.source == Source.ARMBAND && s.running
    Text("Train", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
    if (!live) {
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
            Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Connect the armband first.", Modifier.weight(1f))
                Button(onClick = { connectArmband(s, vm); onHome() }) { Text("Connect") }
            }
        }
    }

    TrainingProgress(s)

    OutlinedTextField(
        value = s.newModelLabel, onValueChange = vm::setNewModelLabel, singleLine = true, modifier = Modifier.fillMaxWidth(),
        label = { Text("Name for the new model (optional)") },
        supportingText = { Text("e.g. \"morning\" or \"left arm\". Recording or calibrating saves a new model with this name; your other models stay.") },
    )

    if (s.model != null) ActionCard(
        "Quick calibration", "1 minute",
        "Hold each gesture for 10 seconds. Do this each time you put the armband on: it keeps the phone in tune with you.",
        "Start calibration", live,
    ) { vm.startRecording(1) }

    var fingers by rememberSaveable { mutableStateOf(s.recordGestures == GestureSet.FINGERS) }
    ActionCard(
        if (s.model == null) "Record your gestures" else "Full recording", "about 3 minutes",
        "Hold each gesture for 10 seconds, three rounds, with short breaks. The phone then learns them (a few minutes).",
        "Start recording", live,
        extra = {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Switch(checked = fingers, onCheckedChange = { fingers = it; vm.setRecordGestures(if (it) GestureSet.FINGERS else GestureSet.BASIC) })
                Text(if (fingers) "With finger gestures (point, peace sign)" else "Hand gestures only (fist, open, wrist)", style = MaterialTheme.typography.bodyMedium)
            }
        },
    ) { vm.startRecording(Calibration.RECORDING_TAKES) }

    ActionCard(
        "Check how well it works", "35 seconds",
        "Hold each gesture for 6 seconds. You'll see how often it was recognised and how fast.",
        "Start check", live && s.model != null,
    ) { vm.startCheck() }

    s.checkScore?.let { score -> CheckResult(score, s.checkReactionMs) }
}

@Composable
private fun ActionCard(title: String, time: String, text: String, button: String, enabled: Boolean, extra: @Composable () -> Unit = {}, onClick: () -> Unit) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                Text(time, style = MaterialTheme.typography.labelMedium)
            }
            Text(text, style = MaterialTheme.typography.bodyMedium)
            extra()
            Button(onClick = onClick, enabled = enabled, modifier = Modifier.fillMaxWidth()) { Text(button) }
        }
    }
}

@Composable
private fun TrainingProgress(s: UiState) {
    val status = s.trainingStatus ?: return
    val pct = Regex("(\\d+)%").find(status)?.groupValues?.get(1)?.toIntOrNull()
    val done = status.startsWith("✓")
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = if (done) Connected.copy(alpha = 0.2f) else MaterialTheme.colorScheme.secondaryContainer)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (done) "Learned your gestures" else "Learning your gestures…", fontWeight = FontWeight.Bold)
            if (!done && pct != null) LinearProgressIndicator(progress = { pct / 100f }, modifier = Modifier.fillMaxWidth())
            Text(if (done) "Saved as a new version. Run a check to see how well it works." else status, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun CheckResult(score: Int, reactionMs: Int?) {
    val (title, advice, color) = when {
        score >= 90 -> Triple("Great: $score% right", "You're ready to go.", Connected)
        score >= 75 -> Triple("Good: $score% right", "A quick calibration can make it even better.", Waiting)
        else -> Triple("$score% right", "Try a quick calibration, or record again with the armband in the same spot.", Danger)
    }
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.18f))) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, color = color)
            reactionMs?.let { Text("Reacts in about ${"%.1f".format(it / 1000.0)} s after you start a gesture.") }
            Text(advice)
        }
    }
}

/** Full screen while recording or checking: what to do now, big and readable from a distance. */
@Composable
private fun GuidedOverlay(s: UiState, vm: InferenceViewModel) {
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(
            Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(if (s.checkIsRecording) "Recording" else "Check", style = MaterialTheme.typography.titleMedium)
            LinearProgressIndicator(progress = { s.checkProgress }, modifier = Modifier.fillMaxWidth().height(8.dp))
            Spacer(Modifier.weight(1f))
            val g = s.checkGesture
            Text(if (g == null) "🧘" else gestureIcon(g), fontSize = 120.sp)
            Text(g?.prompt ?: s.checkTitle, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
            if (g != null && " (take" in s.checkTitle) Text(s.checkTitle.substringAfter(g.name).trim(' ', '(', ')'), style = MaterialTheme.typography.bodyLarge)
            Text("${s.checkSecondsLeft}", fontSize = 64.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
            Text(if (g == null) "Get ready…" else "Hold it until the count reaches 0", style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.weight(1f))
            OutlinedButton(onClick = vm::cancelGuided, Modifier.fillMaxWidth()) { Text("Cancel") }
        }
    }
}

// ---- hand & settings ---------------------------------------------------------------------------

@Composable
private fun HandTab(s: UiState, vm: InferenceViewModel) {
    Text("Hand", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        HandModel.entries.forEach { m -> FilterChip(selected = s.handModel == m, onClick = { vm.selectHandModel(m) }, label = { Text(m.title) }) }
    }
    HandConnectButton(s, vm)
    if (s.handModel == HandModel.CONTROLLER_V2) NewHandPanel(s, vm) else OldHandPanel(s, vm)
}

@Composable
private fun SettingsTab(s: UiState, vm: InferenceViewModel) {
    Text("Settings", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Response", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = s.decisionMode == DecisionMode.GROUPS, label = { Text("Steady") }, onClick = {
                    vm.setGroupSize(4); vm.setDecisionMode(DecisionMode.GROUPS)
                })
                FilterChip(selected = s.decisionMode == DecisionMode.FAST, label = { Text("Fast") }, onClick = {
                    vm.setFastWindows(3); vm.setFastHop(200); vm.setDecisionMode(DecisionMode.FAST)
                })
            }
            Text(if (s.decisionMode == DecisionMode.FAST) "Reacts as soon as it's sure: quicker." else "Waits a little longer before moving: calmer.",
                style = MaterialTheme.typography.bodySmall)

            Text("How sure before moving: ${(s.threshold * 100).roundToInt()}%", fontWeight = FontWeight.Bold)
            Slider(value = s.threshold.toFloat(), onValueChange = { vm.setThreshold((it * 100).roundToInt() / 100.0) }, valueRange = 0.5f..0.95f, steps = 8)
            Text("Higher = fewer mistakes, but it may miss weak gestures.", style = MaterialTheme.typography.bodySmall)

            if (s.handModel == HandModel.HAIFA3D_V1) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Switch(checked = s.proportionalWrist, onCheckedChange = vm::setProportionalWrist)
                    Text("Wrist follows effort (harder = faster turning)")
                }
            }
        }
    }
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("For developers", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text("All recordings, models, armband details and measurements.", style = MaterialTheme.typography.bodySmall)
            OutlinedButton(onClick = { vm.setEngineering(true) }) { Text("Open the engineering view") }
        }
    }
    Text("Project Yad · ${s.person?.replaceFirstChar { it.uppercase() }} · ${s.model?.let { yad.app.displayName(it) } ?: "no model yet"}",
        style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(PaddingValues(top = 8.dp)))
}

/** The user view, or the engineering screens when switched on in Settings. */
@Composable
fun AppRoot(vm: InferenceViewModel) {
    val s by vm.state.collectAsStateWithLifecycle()
    if (s.engineering) InferenceScreen(vm) else UserApp(s, vm)
}

/** [s.person]'s models, newest first (phone-made ones carry their date). */
private fun personModels(s: UiState): List<String> =
    s.models.filter { yad.core.ModelNames.userOf(it) == s.person }.sortedByDescending { it.substringAfter('@', "") }

/** Pick which of the person's models to use; phone-made ones can be deleted. */
@Composable
private fun ModelChooser(s: UiState, vm: InferenceViewModel, onDone: () -> Unit) {
    var deleting by remember { mutableStateOf<String?>(null) }
    AlertDialog(
        onDismissRequest = onDone,
        title = { Text("${s.person?.replaceFirstChar { it.uppercase() }}'s models") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                personModels(s).forEach { m ->
                    Card(
                        Modifier.fillMaxWidth().clickable { vm.selectModel(m); onDone() },
                        colors = CardDefaults.cardColors(containerColor = if (m == s.model) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant),
                    ) {
                        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(yad.app.displayName(m), fontWeight = FontWeight.Bold)
                                Text(
                                    when {
                                        m == s.model -> "in use"
                                        '@' !in m -> "original model (from the PC recording)"
                                        else -> "made on this phone"
                                    }, style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            if ('@' in m) TextButton(onClick = { deleting = m }) { Text("Delete", color = Danger) }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDone) { Text("Close") } },
    )
    deleting?.let { m ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete ${yad.app.displayName(m)}?") },
            text = { Text("The recordings stay, so you can train again any time.") },
            confirmButton = { TextButton(onClick = { vm.deleteModel(m); deleting = null }) { Text("Delete", color = Danger) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
}

/** What the hand would be doing, for trying things out without a hand. */
@Composable
private fun SimulatedHandView(s: UiState, vm: InferenceViewModel) {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Simulated hand", style = MaterialTheme.typography.labelLarge)
            val pose = when (s.simulatedPosition) {
                yad.core.HandCommand.OPEN -> "🖐 open"
                yad.core.HandCommand.CLOSE -> "✊ closed"
                yad.core.HandCommand.POINT -> "☝ pointing"
                yad.core.HandCommand.PEACE -> "✌ peace sign"
                else -> "not moved yet"
            }
            Text("Fingers: $pose", style = MaterialTheme.typography.titleMedium)
            if (s.handModel == HandModel.HAIFA3D_V1) {
                val share = s.simWristMs.toFloat() / yad.app.SIM_WRIST_MS          // -1 (left) .. 1 (right)
                Text("Wrist: " + when {
                    s.simWristMs < 0 -> "turned left (${"%.1f".format(-s.simWristMs / 1000.0)} s of turning)"
                    s.simWristMs > 0 -> "turned right (${"%.1f".format(s.simWristMs / 1000.0)} s of turning)"
                    else -> "straight"
                })
                Row(Modifier.fillMaxWidth().height(14.dp)) {
                    // A bar from the middle: left half fills for left, right half for right.
                    Box(Modifier.weight(1f).fillMaxSize().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(topStart = 7.dp, bottomStart = 7.dp)),
                        contentAlignment = Alignment.CenterEnd) {
                        if (share < 0) Box(Modifier.fillMaxWidth(-share).fillMaxSize().background(MaterialTheme.colorScheme.primary))
                    }
                    Box(Modifier.weight(1f).fillMaxSize().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(topEnd = 7.dp, bottomEnd = 7.dp)),
                        contentAlignment = Alignment.CenterStart) {
                        if (share > 0) Box(Modifier.fillMaxWidth(share).fillMaxSize().background(MaterialTheme.colorScheme.primary))
                    }
                }
                if (s.proportionalWrist) Text("Wrist follows effort is on" + (s.wristEffortPct?.let { ": last effort $it% of your usual" } ?: ""), style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = vm::resetSimulatedWrist) { Text("Straighten") }
            }
        }
    }
}
