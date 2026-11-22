package yad.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * What a model was built from: one person's recordings, in order. A full recording starts a new
 * lineage (that recording only); a calibration extends its parent's lineage with one more
 * recording. Lineages never mix people or gesture sets.
 *
 * [pcData]: the lineage includes the person's original PC recording (the bundled models).
 */
@Serializable
data class ModelLineage(
    val person: String,
    val recordings: List<String> = emptyList(),
    val pcData: Boolean = false,
    val parent: String? = null,
) {
    fun extendedWith(recording: String, parent: String) = copy(recordings = recordings + recording, parent = parent)

    fun toJson(): String = json.encodeToString(this)

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        fun fromJson(text: String): ModelLineage = json.decodeFromString(text)

        /** A bundled PC model: its person's PC recording. */
        fun bundled(person: String) = ModelLineage(person, pcData = true)

        /**
         * For models made before lineages were stored: read the recordings from the model's own note
         * ("trained on the phone from zero: 2 recordings (rec_a.bin, rec_b.bin), …"). The 0.5
         * calibrations ("calibrated on the phone: …") were built on the PC recording.
         */
        fun fromLegacyNote(person: String, note: String): ModelLineage {
            if (note.startsWith("calibrated on the phone")) return bundled(person)
            val files = Regex("""\(([^)]*\.bin[^)]*)\)""").find(note)?.groupValues?.get(1)
                ?.split(",")?.map { it.trim() }?.filter { it.endsWith(".bin") }.orEmpty()
            return ModelLineage(person, files)
        }
    }
}
