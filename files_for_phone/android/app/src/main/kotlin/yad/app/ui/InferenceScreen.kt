@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package yad.app.ui

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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import yad.app.InferenceViewModel
import yad.app.PI_GROUP_SIZE
import yad.app.PI_THRESHOLD
import yad.app.UiState
import yad.core.HandCommand
import kotlin.math.roundToInt

private val StopRed = Color(0xFFD32F2F)
private val ResumeGreen = Color(0xFF2E7D32)

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
        Text("Project Yad — replay", style = MaterialTheme.typography.headlineSmall)
        Text(s.status, style = MaterialTheme.typography.bodySmall)
        s.selfTest?.let {
            Text(it, color = if (s.selfTestPassed == true) ResumeGreen else StopRed, fontWeight = FontWeight.Bold)
        }
        ModelPicker(s, vm::selectModel)

        GestureCard(s)
        StopButton(s.stopped, vm::toggleStop)
        ManualButtons(vm::manual)
        PlaybackControls(s, onPlay = vm::play, onPause = vm::pause, onRealTime = vm::setRealTime)
        Sensitivity(s, vm)
        Stats(s)
        Benchmark(s.benchmark, vm::benchmark)
        CommandLog(s)
    }
}

@Composable
private fun ModelPicker(s: UiState, onSelect: (String) -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        s.models.forEach { name ->
            FilterChip(selected = name == s.model, onClick = { onSelect(name) }, label = { Text(name) })
        }
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
            Text("this window: ${s.windowGesture?.name ?: "—"}   ·   recorded as: ${s.trueGesture?.name ?: "—"}")
            Text("simulated hand: ${s.handPosition?.name ?: "not moved yet"}", style = MaterialTheme.typography.bodySmall)
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
private fun PlaybackControls(s: UiState, onPlay: () -> Unit, onPause: () -> Unit, onRealTime: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Button(onClick = if (s.playing) onPause else onPlay, enabled = s.model != null) {
            Text(if (s.playing) "Pause replay" else "Play replay")
        }
        Switch(checked = s.realTime, onCheckedChange = onRealTime)
        Text(if (s.realTime) "real time" else "as fast as possible")
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
    val winAcc = if (s.windows > 0) 100.0 * s.windowCorrect / s.windows else 0.0
    val decAcc = if (s.decisions > 0) 100.0 * s.decisionCorrect / s.decisions else 0.0
    Text("windows: ${s.windows}   correct: ${"%.1f".format(winAcc)}%")
    Text("decisions: ${s.decisions}   correct: ${"%.1f".format(decAcc)}%")
    Text("CNN: ${"%.1f".format(s.cnnMs)} ms/window   head: ${"%.2f".format(s.headMs)} ms   (budget 200 ms)")
}

@Composable
private fun Benchmark(result: String?, onRun: () -> Unit) {
    HorizontalDivider()
    OutlinedButton(onClick = onRun) { Text("Benchmark CNN (1 / 2 / 4 threads)") }
    result?.let { Text(it, fontFamily = FontFamily.Monospace) }
}

@Composable
private fun CommandLog(s: UiState) {
    HorizontalDivider()
    Text("Commands to the hand", style = MaterialTheme.typography.titleMedium)
    if (s.log.isEmpty()) Text("none yet", style = MaterialTheme.typography.bodySmall)
    s.log.forEach { e ->
        val note = if (e.blocked) "  (blocked by STOP)" else ""
        Text("${e.time}  ${e.source.padEnd(10)} ${e.command.name}$note", fontFamily = FontFamily.Monospace, fontSize = 12.sp)
    }
}
