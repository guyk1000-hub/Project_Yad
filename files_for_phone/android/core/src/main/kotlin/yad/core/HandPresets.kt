package yad.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** One step of a preset, built only from what the hand understands. */
@Serializable
sealed interface PresetAction {
    @Serializable @SerialName("command")
    data class Command(val command: HandCommand) : PresetAction

    /** Every finger to [percent] (0 = open, 100 = closed). */
    @Serializable @SerialName("all")
    data class AllFingers(val percent: Int) : PresetAction

    /** One finger (numbered from 1) to [percent]. */
    @Serializable @SerialName("finger")
    data class Finger(val finger: Int, val percent: Int) : PresetAction
}

/** A named button that runs its actions in order (the "Direct Commands" of the hand's manager app). */
@Serializable
data class HandPreset(val name: String, val actions: List<PresetAction>)

object HandPresets {
    /** The manager app's defaults (its DefCommands.json), made of commands this hand really runs. */
    val DEFAULTS: List<HandPreset> = listOf(
        HandPreset("Calibrate", listOf(PresetAction.Command(HandCommand.CALIBRATE))),
        HandPreset("Open", listOf(PresetAction.Command(HandCommand.OPEN))),
        HandPreset("Close", listOf(PresetAction.Command(HandCommand.CLOSE))),
    ) + listOf(50, 25, 75, 10, 90).map { HandPreset("All $it%", listOf(PresetAction.AllFingers(it))) }

    private val json = Json { prettyPrint = true; classDiscriminator = "type" }

    fun toJson(presets: List<HandPreset>): String = json.encodeToString(presets)

    /** Parses and checks a presets file; throws IllegalArgumentException with a readable reason. */
    fun fromJson(text: String, fingers: Int = 4): List<HandPreset> {
        val list = try {
            json.decodeFromString<List<HandPreset>>(text)
        } catch (e: Exception) {
            throw IllegalArgumentException("not a presets file (${e.message?.lineSequence()?.firstOrNull()})")
        }
        require(list.isNotEmpty()) { "no presets in the file" }
        for (p in list) {
            require(p.name.isNotBlank() && p.actions.isNotEmpty()) { "preset \"${p.name}\" is empty" }
            for (a in p.actions) when (a) {
                is PresetAction.AllFingers -> require(a.percent in 0..100) { "${p.name}: ${a.percent}% outside 0..100" }
                is PresetAction.Finger -> require(a.finger in 1..fingers && a.percent in 0..100) { "${p.name}: finger ${a.finger} / ${a.percent}%" }
                is PresetAction.Command -> {}
            }
        }
        return list
    }

    fun describe(a: PresetAction, fingerNames: List<String>): String = when (a) {
        is PresetAction.Command -> a.command.title.lowercase().replaceFirstChar { it.uppercase() }
        is PresetAction.AllFingers -> "all fingers to ${a.percent}%"
        is PresetAction.Finger -> "${fingerNames.getOrElse(a.finger - 1) { "finger ${a.finger}" }} to ${a.percent}%"
    }
}
