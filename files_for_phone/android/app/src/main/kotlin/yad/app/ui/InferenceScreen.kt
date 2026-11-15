@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package yad.app.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import yad.app.InferenceViewModel
import yad.app.PI_GROUP_SIZE
import yad.app.PI_THRESHOLD
import yad.app.Source
import yad.app.UiState
import yad.app.displayName
import yad.core.HandCommand
import kotlin.math.roundToInt

private val StopRed = Color(0xFFD32F2F)
private val ResumeGreen = Color(0xFF2E7D32)
private val WarnAmber = Color(0xFFFFA000)

@Composable
fun InferenceScreen(vm: InferenceViewModel) {
    val s by vm.state.collectAsStateWithLifecycle()
    Column(
        Modifier
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Project Yad", style = MaterialTheme.typography.headlineSmall)
        Text(s.status, style = MaterialTheme.typography.bodySmall)
        s.selfTest?.let {
            Text(it, color = if (s.selfTestPassed == true) ResumeGreen else StopRed, fontWeight = FontWeight.Bold)
        }
        ModelPicker(s, vm::selectModel, vm::deleteModel)
        SourcePicker(s, vm::selectSource)

        GestureCard(s)
        StopButton(s.stopped, vm::toggleStop)
        ManualButtons(vm::manual)
        HandSection(s, vm::setRealHand)
        RunControls(s, onStart = vm::start, onStop = vm::stopRun, onRealTime = vm::setRealTime)
        if (s.source == Source.ARMBAND) CheckSection(s, vm::startCheck, vm::startCalibration)
        if (s.source == Source.ARMBAND) ArmbandSection(s, vm::setAcceptBroadcast, vm::requestEegMode, vm::captureRawPackets, vm::diagnosticsText)
        Sensitivity(s, vm)
        Stats(s)
        Benchmark(s, vm::benchmark)
        CommandLog(s)
    }
}

@Composable
private fun ModelPicker(s: UiState, onSelect: (String) -> Unit, onDelete: (String) -> Unit) {
    Text("User model", style = MaterialTheme.typography.labelMedium)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        s.models.forEach { name ->
            FilterChip(selected = name == s.model, onClick = { onSelect(name) }, label = { Text(displayName(name)) })
        }
    }
    val current = s.model
    if (current != null && '@' in current) {
        OutlinedButton(onClick = { onDelete(current) }) { Text("Delete ${displayName(current)}") }
    }
}

@Composable
private fun SourcePicker(s: UiState, onSelect: (Source) -> Unit) {
    Text("Signal", style = MaterialTheme.typography.labelMedium)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(selected = s.source == Source.REPLAY, onClick = { onSelect(Source.REPLAY) }, label = { Text("Recording (replay)") })
        FilterChip(selected = s.source == Source.ARMBAND, onClick = { onSelect(Source.ARMBAND) }, label = { Text("Armband (live)") })
    }
}

@Composable
private fun GestureCard(s: UiState) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Decision", style = MaterialTheme.typography.labelMedium)
            Text(s.decided?.name ?: "—", fontSize = 44.sp, fontWeight = FontWeight.Bold)
            LinearProgressIndicator(progress = { s.confidence.toFloat() }, modifier = Modifier.fillMaxWidth())
            Text("confidence ${(s.confidence * 100).roundToInt()}%   ·   decision every ${s.decisionEveryMs} ms")
            Spacer(Modifier.height(8.dp))
            val truth = if (s.source == Source.REPLAY) "   ·   recorded as: ${s.trueGesture?.name ?: "—"}" else ""
            Text("this window: ${s.windowGesture?.name ?: "—"}$truth")
            Text(
                if (s.realHand) "real hand: ${s.handState}" else "simulated hand: ${s.simulatedPosition?.name ?: "not moved yet"}",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun StopButton(stopped: Boolean, onToggle: () -> Unit) {
    Button(
        onClick = onToggle,
        modifier = Modifier.fillMaxWidth().height(72.dp),
        colors = ButtonDefaults.buttonColors(containerColor = if (stopped) ResumeGreen else StopRed),
    ) {
        Text(if (stopped) "STOPPED — tap to resume" else "STOP", fontSize = 24.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun ManualButtons(onCommand: (HandCommand) -> Unit) {
    Text("Manual (always works, even when stopped)", style = MaterialTheme.typography.labelMedium)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        HandCommand.entries.forEach { cmd ->
            OutlinedButton(onClick = { onCommand(cmd) }, modifier = Modifier.weight(1f)) { Text(cmd.name) }
        }
    }
}

@Composable
private fun HandSection(s: UiState, onRealHand: (Boolean) -> Unit) {
    val context = LocalContext.current
    val perms = arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN)
    var denied by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        denied = !result.values.all { it }
        if (!denied) onRealHand(true)
    }
    HorizontalDivider()
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Switch(checked = s.realHand, onCheckedChange = { on ->
            if (!on) onRealHand(false)
            else if (perms.all { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }) onRealHand(true)
            else launcher.launch(perms)
        })
        Column {
            Text("Real hand (Bluetooth)", fontWeight = FontWeight.Bold)
            Text(if (s.realHand) s.handState else "off — commands go to the simulated hand", style = MaterialTheme.typography.bodySmall)
        }
    }
    if (s.realHand) Text("The hand moves. STOP blocks the classifier; manual buttons still move it.", color = WarnAmber)
    if (denied) Text("Bluetooth permission was refused, so the real hand can't be used.", color = StopRed)
}

@Composable
private fun RunControls(s: UiState, onStart: () -> Unit, onStop: () -> Unit, onRealTime: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        val what = if (s.source == Source.REPLAY) "replay" else "armband"
        Button(onClick = if (s.running) onStop else onStart, enabled = s.model != null) {
            Text(if (s.running) "Stop $what" else "Start $what")
        }
        if (s.source == Source.REPLAY) {
            Switch(checked = s.realTime, onCheckedChange = onRealTime)
            Text(if (s.realTime) "real time" else "as fast as possible")
        }
    }
}

@Composable
private fun ArmbandSection(
    s: UiState,
    onAcceptBroadcast: (Boolean) -> Unit,
    onRequestEeg: () -> Unit,
    onCapture: () -> Unit,
    diagnostics: () -> String,
) {
    val a = s.armband
    val captureMessage = s.captureMessage
    val clipboard = LocalClipboardManager.current
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Armband", style = MaterialTheme.typography.titleMedium)
            a.error?.let { Text(it, color = StopRed) }
            if (a.running || a.datagrams > 0) {
                Text("listening: ${a.mode}")
                a.format?.let { Text("format: $it") }
                Text("phone on WiFi: ${a.wifi ?: "—"}   armband: ${a.sender ?: "waiting for data…"}")
                Text("samples/s: ${"%.0f".format(a.samplesPerSecond)} (expect 500)   lost: ${a.lostSamples}   battery: ${a.battery?.let { "$it%" } ?: "—"}")
                Text("delay behind live: ${s.lagMs} ms" + if (s.lagMs > 500) "  ← falling behind" else "",
                    color = if (s.lagMs > 500) StopRed else Color.Unspecified)
                if (a.rejectedPackets > 0) Text("rejected packets (not from the armband): ${a.rejectedPackets}", color = WarnAmber)
                Text("EEG mode: requested ${a.eegRequests}×, " + if (a.eegAcknowledged) "acknowledged ✓" else "no acknowledgement yet")
                OutlinedButton(onClick = onRequestEeg, enabled = a.sender != null) { Text("Request EEG mode") }
                HorizontalDivider()
                Text("Diagnostics", fontWeight = FontWeight.Bold)
                Text("packets: ${a.datagrams} (${"%.0f".format(a.packetsPerSecond)}/s)   EMG: ${a.emgPackets}   other: ${a.otherPackets}   samples: ${a.samples}", fontSize = 12.sp)
                a.sendError?.let { Text("EEG request failed: $it", color = StopRed, fontSize = 12.sp) }
                if (a.bySize.isNotEmpty()) {
                    Text("non-EMG packets by size (top 8):", fontSize = 12.sp, color = WarnAmber)
                    a.bySize.forEach { (size, v) ->
                        Text("$size B × ${v.first}: ${v.second.take(47)}", fontFamily = FontFamily.Monospace, fontSize = 10.sp)
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onCapture) { Text("Save 5 s raw packets") }
                    OutlinedButton(onClick = { clipboard.setText(AnnotatedString(diagnostics())) }) { Text("Copy diagnostics") }
                }
                captureMessage?.let { Text(it, fontSize = 12.sp) }
                if (a.running && a.datagrams == 0L && !s.acceptBroadcast) {
                    Text("No data yet. If the armband is on and this stays empty, it may broadcast: try the switch below.", color = WarnAmber)
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Switch(checked = s.acceptBroadcast, onCheckedChange = onAcceptBroadcast)
                Text("Also accept broadcast packets", style = MaterialTheme.typography.bodySmall)
            }
            Text(
                "Off: listens only on the phone's WiFi address. On: listens on port 4210 for any address on the WiFi " +
                    "network, which includes broadcasts. Either way, only data from the armband's WiFi network, and only " +
                    "from the first device that streams, is used.",
                style = MaterialTheme.typography.bodySmall,
            )
            HorizontalDivider()
            Text("Signal level vs your recording (rest your arm)", fontWeight = FontWeight.Bold)
            val live = s.liveLevels
            val ref = s.restReference
            if (live == null || ref == null) {
                Text("waiting for data…", style = MaterialTheme.typography.bodySmall)
            } else {
                s.levelVerdict?.let { Text(it) }
                Text("live µV:     " + live.joinToString(" ") { "%5.1f".format(it) }, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                Text("recorded µV: " + ref.joinToString(" ") { "%5.1f".format(it) }, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
            }
        }
    }
}

@Composable
private fun CheckSection(s: UiState, onStart: () -> Unit, onCalibrate: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Calibrate & check", style = MaterialTheme.typography.titleMedium)
            val prompt = s.checkPrompt
            if (prompt != null) {
                Text(s.checkStep ?: "", style = MaterialTheme.typography.labelMedium)
                Text(prompt, fontSize = 30.sp, fontWeight = FontWeight.Bold)
            } else {
                Text("Calibrate (~1 min): hold each gesture for 10 s; the phone then retrains your model with it " +
                    "(about 3–5 min on the phone) and saves it as a new version. Do this each time you put the armband on.",
                    style = MaterialTheme.typography.bodySmall)
                Button(onClick = onCalibrate, enabled = s.running) { Text("Calibrate") }
                Text("Guided check (~35 s): hold each gesture for 6 s and see how well each is recognised.",
                    style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = onStart, enabled = s.running) { Text("Start guided check") }
            }
            s.checkResults.forEach { Text(it, fontFamily = FontFamily.Monospace, fontSize = 13.sp) }
            s.checkStatus?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            s.calibrationStatus?.let { Text(it, fontWeight = FontWeight.Bold) }
        }
    }
}

@Composable
private fun Sensitivity(s: UiState, vm: InferenceViewModel) {
    HorizontalDivider()
    Text("Sensitivity", style = MaterialTheme.typography.titleMedium)
    Text("Confidence threshold: ${(s.threshold * 100).roundToInt()}%")
    Slider(
        value = s.threshold.toFloat(), onValueChange = { vm.setThreshold((it * 100).roundToInt() / 100.0) },
        valueRange = 0.3f..0.95f, steps = 12,
    )
    Text("Smoothing: ${s.groupSize} windows (${s.decisionEveryMs} ms per decision)")
    Slider(
        value = s.groupSize.toFloat(), onValueChange = { vm.setGroupSize(it.roundToInt()) },
        valueRange = 1f..12f, steps = 10,
    )
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Switch(checked = s.sendOnlyOnChange, onCheckedChange = vm::setSendOnlyOnChange)
        Text("send only when the gesture changes")
    }
    OutlinedButton(onClick = vm::piPreset) {
        Text("Pi settings (${PI_GROUP_SIZE} windows, ${(PI_THRESHOLD * 100).roundToInt()}%)")
    }
}

@Composable
private fun Stats(s: UiState) {
    HorizontalDivider()
    Text("Stats", style = MaterialTheme.typography.titleMedium)
    Text("windows: ${s.windows}" + if (s.source == Source.REPLAY && s.windows > 0) "   correct: ${"%.1f".format(100.0 * s.windowCorrect / s.windows)}%" else "")
    if (s.source == Source.REPLAY) {
        val t = s.scoring
        Text("decisions where one gesture was held: ${t.steadyDecided}   correct: ${"%.1f".format(100 * s.steadyAccuracy)}%")
        Text("groups spanning a gesture change: ${t.transitionGroups} (not scored)", style = MaterialTheme.typography.bodySmall)
        Text("groups below the threshold (no decision): ${t.groups - t.decided}", style = MaterialTheme.typography.bodySmall)
    }
    Text("CNN: ${"%.1f".format(s.cnnMs)} ms/window   head: ${"%.2f".format(s.headMs)} ms   (budget 200 ms)")
}

@Composable
private fun Benchmark(s: UiState, onRun: () -> Unit) {
    HorizontalDivider()
    OutlinedButton(onClick = onRun, enabled = !s.benchmarkRunning) {
        Text(if (s.benchmarkRunning) "Benchmark running…" else "Benchmark CNN engines")
    }
    s.benchmark.forEach { Text(it, fontFamily = FontFamily.Monospace, fontSize = 11.sp) }
}

@Composable
private fun CommandLog(s: UiState) {
    HorizontalDivider()
    Text("Commands to the hand", style = MaterialTheme.typography.titleMedium)
    if (s.log.isEmpty()) Text("none yet", style = MaterialTheme.typography.bodySmall)
    s.log.forEach { e ->
        Text("${e.time}  ${e.source.padEnd(10)} ${e.command.name.padEnd(5)}  ${e.outcome}", fontFamily = FontFamily.Monospace, fontSize = 12.sp)
    }
}
