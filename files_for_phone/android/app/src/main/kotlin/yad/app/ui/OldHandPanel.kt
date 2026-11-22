@file:OptIn(ExperimentalLayoutApi::class)

package yad.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
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
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import yad.app.InferenceViewModel
import yad.app.UiState
import yad.core.HandCommand
import yad.core.Haifa3dProtocol
import yad.core.Haifa3dProtocol.Motor
import yad.core.Haifa3dProtocol.Movement
import kotlin.math.roundToInt

/**
 * Controls for the old hand (Haifa3D v1): everything the Haifa3D app offers — whole-hand moves,
 * hold-to-move live control, actions built from movements, the 12 presets stored on the hand,
 * its configuration values and battery level.
 */
@Composable
fun OldHandPanel(s: UiState, vm: InferenceViewModel) {
    var editing by remember { mutableStateOf<Int?>(null) }        // movement index; size = a new one
    var confirmReset by remember { mutableStateOf<Int?>(null) }    // 0 presets, 1 configuration
    var saveSlot by remember { mutableIntStateOf(0) }

    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Old hand", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                AssistChip(onClick = vm::v1ReadBattery, label = { Text("battery ${s.v1Battery?.let { "$it%" } ?: "?"}") })
                StatusPill(s)
            }

            PanelSection("Whole hand") {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = { vm.manual(HandCommand.OPEN) }, Modifier.weight(1f).height(60.dp)) { Text("OPEN", fontSize = 20.sp) }
                    Button(onClick = { vm.manual(HandCommand.CLOSE) }, Modifier.weight(1f).height(60.dp)) { Text("CLOSE", fontSize = 20.sp) }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    FilledTonalButton(onClick = { vm.manual(HandCommand.LEFT) }, Modifier.weight(1f)) { Text("◀ Wrist left") }
                    FilledTonalButton(onClick = { vm.manual(HandCommand.RIGHT) }, Modifier.weight(1f)) { Text("Wrist right ▶") }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    FilledTonalButton(onClick = { vm.manual(HandCommand.POINT) }, Modifier.weight(1f)) { Text("☝ Point") }
                    FilledTonalButton(onClick = { vm.manual(HandCommand.PEACE) }, Modifier.weight(1f)) { Text("✌ Peace") }
                }
            }

            PanelSection("Live control — hold to move") {
                HoldRow("Wrist", "◀ Left", "Right ▶", Motor.TURN, vm)
                listOf(Motor.FINGER1, Motor.FINGER2, Motor.FINGER3, Motor.FINGER4).forEach { HoldRow(it.title, "Open", "Close", it, vm) }
            }

            PanelSection("Action builder") {
                if (s.v1Movements.isEmpty()) Text("No movements yet. An action runs its movements one after another.", style = MaterialTheme.typography.bodySmall)
                s.v1Movements.forEachIndexed { i, m ->
                    Card(Modifier.fillMaxWidth().clickable { editing = i }, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
                        Text("${i + 1}. ${m.describe()}", Modifier.padding(10.dp), style = MaterialTheme.typography.bodyMedium)
                    }
                }
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { editing = s.v1Movements.size }) { Text("+ Movement") }
                    Button(onClick = vm::v1RunMovements, enabled = s.v1Movements.isNotEmpty()) { Text("Run") }
                    TextButton(onClick = { vm.v1SetMovements(emptyList()) }, enabled = s.v1Movements.isNotEmpty()) { Text("Clear") }
                }
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Save as preset #")
                    SlotPicker(saveSlot) { saveSlot = it }
                    OutlinedButton(onClick = { vm.v1SavePreset(saveSlot) }, enabled = s.v1Movements.isNotEmpty()) { Text("Save to hand") }
                }
            }

            PanelSection("Presets on the hand (12)") {
                OutlinedButton(onClick = vm::v1ReadPresets) { Text("Read from hand") }
                s.v1Presets.forEachIndexed { n, desc ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("#$n", Modifier.width(36.dp), fontWeight = FontWeight.Bold)
                        Text(desc ?: "—", Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, maxLines = 2)
                        TextButton(onClick = { vm.v1LoadPreset(n) }) { Text("Edit") }
                        FilledTonalButton(onClick = { vm.v1TriggerPreset(n) }) { Text("Run") }
                    }
                }
            }

            PanelSection("Configuration") {
                OutlinedButton(onClick = vm::v1ReadConfig) { Text("Read from hand") }
                Haifa3dProtocol.CONFIG.forEach { c -> ConfigRow(c, s.v1Config[c.id], vm) }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { confirmReset = 0 }, colors = ButtonDefaults.outlinedButtonColors(contentColor = Danger)) { Text("Reset presets") }
                    OutlinedButton(onClick = { confirmReset = 1 }, colors = ButtonDefaults.outlinedButtonColors(contentColor = Danger)) { Text("Reset configuration") }
                }
            }

            s.v1Message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

            PanelSection("Not in this app yet") {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    AssistChip(onClick = {}, enabled = false, label = { Text("Voice commands (Haifa3D app's \"speak to action\")") })
                }
            }
        }
    }

    editing?.let { i ->
        MovementEditor(
            s.v1Movements.getOrNull(i),
            onSave = { m -> vm.v1SetMovements(s.v1Movements.toMutableList().also { if (i < it.size) it[i] = m else it.add(m) }); editing = null },
            onDelete = if (i < s.v1Movements.size) ({ vm.v1SetMovements(s.v1Movements.filterIndexed { k, _ -> k != i }); editing = null }) else null,
            onDismiss = { editing = null },
        )
    }

    confirmReset?.let { which ->
        AlertDialog(
            onDismissRequest = { confirmReset = null },
            title = { Text(if (which == 0) "Reset the presets?" else "Reset the configuration?") },
            text = { Text("This puts the hand's ${if (which == 0) "12 presets" else "configuration values"} back to factory defaults. It can't be undone.") },
            confirmButton = { TextButton(onClick = { vm.v1Reset(which); confirmReset = null }) { Text("Reset", color = Danger) } },
            dismissButton = { TextButton(onClick = { confirmReset = null }) { Text("Cancel") } },
        )
    }
}

/** A row of two press-and-hold buttons for one motor. */
@Composable
private fun HoldRow(title: String, back: String, forward: String, motor: Motor, vm: InferenceViewModel) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(title, Modifier.width(72.dp))
        HoldButton(back, Modifier.weight(1f), onDown = { vm.v1Hold(motor, closing = false) }, onUp = { vm.v1Release(motor) })
        HoldButton(forward, Modifier.weight(1f), onDown = { vm.v1Hold(motor, closing = true) }, onUp = { vm.v1Release(motor) })
    }
}

@Composable
private fun HoldButton(text: String, modifier: Modifier, onDown: () -> Unit, onUp: () -> Unit) {
    var pressed by remember { mutableStateOf(false) }
    Surface(
        modifier.height(44.dp).pointerInput(Unit) {
            detectTapGestures(onPress = {
                pressed = true; onDown()
                try { tryAwaitRelease() } finally { pressed = false; onUp() }
            })
        },
        shape = RoundedCornerShape(50),
        color = if (pressed) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer,
    ) {
        Box(contentAlignment = Alignment.Center) {
            Text(text, color = if (pressed) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondaryContainer)
        }
    }
}

@Composable
private fun SlotPicker(slot: Int, onPick: (Int) -> Unit) {
    var open by remember { mutableStateOf(false) }
    OutlinedButton(onClick = { open = true }) { Text("$slot") }
    if (open) AlertDialog(
        onDismissRequest = { open = false },
        title = { Text("Preset slot") },
        text = {
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                (0 until Haifa3dProtocol.PRESETS).forEach { n -> FilterChip(selected = n == slot, onClick = { onPick(n); open = false }, label = { Text("#$n") }) }
            }
        },
        confirmButton = { TextButton(onClick = { open = false }) { Text("Close") } },
    )
}

@Composable
private fun ConfigRow(c: Haifa3dProtocol.ConfigValue, value: Int?, vm: InferenceViewModel) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Column(Modifier.weight(1f)) {
            Text(c.title, style = MaterialTheme.typography.bodyMedium)
            Text(c.short, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (c.isSwitch) {
            Switch(checked = value == 1, onCheckedChange = { vm.v1WriteConfig(c.id, if (it) 1 else 0) }, enabled = value != null)
        } else {
            var text by remember(value) { mutableStateOf(value?.toString() ?: "") }
            OutlinedTextField(
                value = text, onValueChange = { t -> text = t.filter { it.isDigit() }.take(3) }, singleLine = true,
                modifier = Modifier.width(84.dp), placeholder = { Text("?") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            )
            TextButton(onClick = { text.toIntOrNull()?.let { vm.v1WriteConfig(c.id, it) } }, enabled = text.toIntOrNull() != null && text != value?.toString()) { Text("Set") }
        }
    }
}

/** One movement: which motors, which way each, how long, which torque limit. */
@Composable
private fun MovementEditor(m: Movement?, onSave: (Movement) -> Unit, onDelete: (() -> Unit)?, onDismiss: () -> Unit) {
    var motors by remember { mutableStateOf(m?.motors ?: setOf(Motor.FINGER1, Motor.FINGER2, Motor.FINGER3, Motor.FINGER4)) }
    var closing by remember { mutableStateOf(m?.closing ?: motors) }
    var high by remember { mutableStateOf(m?.highTorque?.isNotEmpty() ?: false) }
    var time by remember { mutableStateOf((m?.timeUnits ?: 20).toFloat()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (m == null) "New movement" else "Edit movement") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Motor.entries.forEach { mo ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(checked = mo in motors, onCheckedChange = { on -> motors = if (on) motors + mo else motors - mo })
                        Text(mo.title, Modifier.width(72.dp))
                        val (a, b) = if (mo == Motor.TURN) "right" to "left" else "open" to "close"
                        FilterChip(selected = mo !in closing, onClick = { closing = closing - mo }, label = { Text(a) }, enabled = mo in motors)
                        FilterChip(selected = mo in closing, onClick = { closing = closing + mo }, label = { Text(b) }, enabled = mo in motors,
                            modifier = Modifier.padding(start = 6.dp))
                    }
                }
                Text("Stop after ${time.roundToInt() * Haifa3dProtocol.MS_PER_TIME_UNIT} ms (or at the torque limit)")
                Slider(value = time, onValueChange = { time = it }, valueRange = 1f..100f)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = high, onCheckedChange = { high = it })
                    Text(if (high) "  High torque (grips harder)" else "  Low torque (gentle)")
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onSave(Movement(motors, closing intersect motors, if (high) motors else emptySet(), time.roundToInt()))
            }, enabled = motors.isNotEmpty()) { Text("Save") }
        },
        dismissButton = {
            Row {
                if (onDelete != null) TextButton(onClick = onDelete, colors = ButtonDefaults.textButtonColors(contentColor = Danger)) { Text("Delete") }
                TextButton(onClick = onDismiss) { Text("Cancel") }
            }
        },
    )
}
