@file:OptIn(ExperimentalLayoutApi::class)

package yad.app.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import yad.app.InferenceViewModel
import yad.app.UiState
import yad.core.HandCommand
import yad.core.HandModel
import yad.core.HandPreset
import yad.core.HandPresets
import yad.core.PresetAction
import kotlin.math.roundToInt

private val Connected = Color(0xFF2E7D32)
private val Waiting = Color(0xFFFFA000)
private val Danger = Color(0xFFC62828)

/**
 * Controls for the new hand (controller v2): everything its own manager app offers that the hand's
 * code runs, plus presets made of those commands, and what it doesn't run yet shown as placeholders.
 */
@Composable
fun NewHandPanel(s: UiState, vm: InferenceViewModel) {
    val model = s.handModel
    var confirmCalibrate by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf<Int?>(null) }      // preset index; presets.size = a new one
    var editMode by remember { mutableStateOf(false) }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { vm.importPresets(it) }

    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("New hand", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                StatusPill(s)
            }

            Section("Whole hand") {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = { vm.manual(HandCommand.OPEN) }, Modifier.weight(1f).height(60.dp)) { Text("OPEN", fontSize = 20.sp) }
                    Button(onClick = { vm.manual(HandCommand.CLOSE) }, Modifier.weight(1f).height(60.dp)) { Text("CLOSE", fontSize = 20.sp) }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilledTonalButton(onClick = { vm.manual(HandCommand.STEP) }, Modifier.weight(1f)) { Text("Close a bit") }
                    FilledTonalButton(onClick = { confirmCalibrate = true }, Modifier.weight(1f)) { Text("Calibrate…") }
                    OutlinedButton(
                        onClick = { vm.manual(HandCommand.STOP) }, Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Danger),
                    ) { Text("Stop motors") }
                }
            }

            model.speedRange?.let { range ->
                var speed by remember(model) { mutableStateOf(s.handSpeed.toFloat()) }
                Section("Speed  ${speed.roundToInt()}") {
                    Slider(
                        value = speed, onValueChange = { speed = it },
                        valueRange = range.first.toFloat()..range.last.toFloat(), steps = (range.last - range.first) / 10 - 1,
                        onValueChangeFinished = { vm.setHandSpeed(speed.roundToInt()) },
                    )
                }
            }

            if (model.fingers.isNotEmpty()) {
                Section("Fingers   0% open · 100% closed") {
                    model.fingers.forEachIndexed { i, name ->
                        var pos by remember(model) { mutableStateOf(s.fingerPositions.getOrElse(i) { 0 }.toFloat()) }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(name, Modifier.width(64.dp))
                            Slider(
                                value = pos, onValueChange = { pos = it }, valueRange = 0f..100f, steps = 9,
                                onValueChangeFinished = { vm.setFingerPosition(i + 1, pos.roundToInt()) },
                                modifier = Modifier.weight(1f),
                            )
                            Text("${pos.roundToInt()}%", Modifier.width(44.dp))
                        }
                    }
                }
            }

            Section("Presets") {
                s.presets.withIndex().chunked(2).forEach { pair ->
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        pair.forEach { (i, p) ->
                            PresetTile(p, model, editMode, Modifier.weight(1f)) { if (editMode) editing = i else vm.runPreset(p) }
                        }
                        if (pair.size == 1) Spacer(Modifier.weight(1f))
                    }
                }
                FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = { editMode = !editMode }) { Text(if (editMode) "Done" else "Edit") }
                    if (editMode) TextButton(onClick = { editing = s.presets.size }) { Text("+ New") }
                    TextButton(onClick = { importer.launch("*/*") }) { Text("Import…") }
                    TextButton(onClick = vm::exportPresets) { Text("Export") }
                    TextButton(onClick = vm::resetPresets) { Text("Defaults") }
                }
                if (editMode) Text("Tap a preset to change it.", style = MaterialTheme.typography.bodySmall)
                s.presetMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }

            if (model.notYetSupported.isNotEmpty()) {
                Section("Not supported by this hand yet") {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        model.notYetSupported.forEach { AssistChip(onClick = {}, enabled = false, label = { Text(it) }) }
                    }
                    Text("Its manager app offers these, but the code on the hand ignores them; they'll work here once the hand supports them.",
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }

    if (confirmCalibrate) AlertDialog(
        onDismissRequest = { confirmCalibrate = false },
        title = { Text("Calibrate the hand?") },
        text = { Text("Each finger drives fully open and fully closed until it stops, then opens. Keep the fingers clear.") },
        confirmButton = { TextButton(onClick = { confirmCalibrate = false; vm.manual(HandCommand.CALIBRATE) }) { Text("Calibrate") } },
        dismissButton = { TextButton(onClick = { confirmCalibrate = false }) { Text("Cancel") } },
    )

    editing?.let { index ->
        PresetEditor(
            preset = s.presets.getOrNull(index), model = model,
            onSave = { p ->
                val list = s.presets.toMutableList()
                if (index < list.size) list[index] = p else list.add(p)
                vm.savePresets(list); editing = null
            },
            onDelete = if (index < s.presets.size && s.presets.size > 1) ({ vm.savePresets(s.presets.filterIndexed { i, _ -> i != index }); editing = null }) else null,
            onDismiss = { editing = null },
        )
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        content()
    }
}

@Composable
private fun StatusPill(s: UiState) {
    val (text, color) = when {
        !s.realHand -> "simulated" to Color.Gray
        s.handState == "connected" -> "connected" to Connected
        s.handState.startsWith("failed") -> "not connected" to Danger
        else -> "waiting for the hand" to Waiting
    }
    Surface(color = color.copy(alpha = 0.18f), shape = RoundedCornerShape(50)) {
        Text(text, color = color, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp))
    }
}

@Composable
private fun PresetTile(p: HandPreset, model: HandModel, editMode: Boolean, modifier: Modifier, onClick: () -> Unit) {
    Card(
        modifier.clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = if (editMode) MaterialTheme.colorScheme.tertiaryContainer else MaterialTheme.colorScheme.secondaryContainer),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(p.name, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            Text(p.actions.joinToString(", ") { HandPresets.describe(it, model.fingers) }, style = MaterialTheme.typography.bodySmall, maxLines = 2)
        }
    }
}

/** Edits one preset as a single action; presets with several actions can come from an imported file. */
@Composable
private fun PresetEditor(preset: HandPreset?, model: HandModel, onSave: (HandPreset) -> Unit, onDelete: (() -> Unit)?, onDismiss: () -> Unit) {
    val first = preset?.actions?.firstOrNull()
    val kinds = listOf("Open", "Close", "Close a bit", "Calibrate", "All fingers to", "One finger to")
    var name by remember { mutableStateOf(preset?.name ?: "New preset") }
    var kind by remember {
        mutableIntStateOf(when (first) {
            is PresetAction.Command -> when (first.command) { HandCommand.CLOSE -> 1; HandCommand.STEP -> 2; HandCommand.CALIBRATE -> 3; else -> 0 }
            is PresetAction.AllFingers -> 4
            is PresetAction.Finger -> 5
            null -> 4
        })
    }
    var percent by remember { mutableStateOf(((first as? PresetAction.AllFingers)?.percent ?: (first as? PresetAction.Finger)?.percent ?: 50).toFloat()) }
    var finger by remember { mutableIntStateOf((first as? PresetAction.Finger)?.finger ?: 1) }
    val several = (preset?.actions?.size ?: 0) > 1

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (preset == null) "New preset" else "Edit preset") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it.take(20) }, label = { Text("Name") }, singleLine = true)
                if (several) Text("This preset has ${preset!!.actions.size} steps; saving keeps only the one chosen here.", style = MaterialTheme.typography.bodySmall)
                kinds.forEachIndexed { i, k ->
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.clickable { kind = i }) {
                        RadioButton(selected = kind == i, onClick = { kind = i })
                        Text(k)
                    }
                }
                if (kind == 5) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    model.fingers.forEachIndexed { i, f -> FilterChip(selected = finger == i + 1, onClick = { finger = i + 1 }, label = { Text(f) }) }
                }
                if (kind >= 4) {
                    Text("${percent.roundToInt()}%  (0 = open, 100 = closed)")
                    Slider(value = percent, onValueChange = { percent = it }, valueRange = 0f..100f, steps = 19)
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val action = when (kind) {
                    0 -> PresetAction.Command(HandCommand.OPEN)
                    1 -> PresetAction.Command(HandCommand.CLOSE)
                    2 -> PresetAction.Command(HandCommand.STEP)
                    3 -> PresetAction.Command(HandCommand.CALIBRATE)
                    4 -> PresetAction.AllFingers(percent.roundToInt())
                    else -> PresetAction.Finger(finger, percent.roundToInt())
                }
                onSave(HandPreset(name.ifBlank { "Preset" }, listOf(action)))
            }) { Text("Save") }
        },
        dismissButton = {
            Row {
                if (onDelete != null) TextButton(onClick = onDelete, colors = ButtonDefaults.textButtonColors(contentColor = Danger)) { Text("Delete") }
                TextButton(onClick = onDismiss) { Text("Cancel") }
            }
        },
    )
}
