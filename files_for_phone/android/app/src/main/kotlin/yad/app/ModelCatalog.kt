package yad.app

import android.content.Context
import yad.core.MlpHead
import yad.core.ReplayRecording
import yad.core.RmsOver
import yad.core.TrainingSet
import java.io.File

/**
 * The user models: the ones bundled in the APK (copied from files_for_phone/models at build time;
 * offered when head_<user>.json and replay_<user>.bin exist), plus versions calibrated on this
 * phone, stored as files/models/<user>@<yyyyMMdd_HHmm>.json.
 *
 * A calibrated version uses its user's bundled replay and training data.
 */
class ModelCatalog(private val context: Context) {
    private val calibratedDir = File(context.filesDir, "models").apply { mkdirs() }

    private val bundled: List<String> by lazy {
        val files = context.assets.list("").orEmpty().toSet()
        files.filter { it.startsWith("head_") && it.endsWith(".json") }
            .map { it.removePrefix("head_").removeSuffix(".json") }
            .filter { "replay_$it.bin" in files }
            .sorted()
    }

    /** Bundled models, then calibrated versions (newest first within each user). */
    val names: List<String>
        get() {
            val calibrated = calibratedDir.listFiles { f -> f.name.endsWith(".json") }.orEmpty()
                .map { it.name.removeSuffix(".json") }
                .filter { userOf(it) in bundled }
                .sortedDescending()
            return bundled + calibrated
        }

    fun userOf(name: String): String = name.substringBefore('@')

    fun isCalibrated(name: String) = '@' in name

    /** Newest calibrated version of [user], if any. */
    fun latestFor(user: String): String? = names.filter { isCalibrated(it) && userOf(it) == user }.maxOrNull()

    fun head(name: String): MlpHead = MlpHead.fromJson(
        if (isCalibrated(name)) File(calibratedDir, "$name.json").readText()
        else context.assets.open("head_$name.json").bufferedReader().use { it.readText() }
    )

    fun replay(name: String): ReplayRecording =
        ReplayRecording.parse(context.assets.open("replay_${userOf(name)}.bin").use { it.readBytes() })

    /** The user's original recording as head inputs, or null if it isn't bundled. */
    fun trainingData(name: String): Pair<TrainingSet, RmsOver>? = try {
        TrainingSet.parse(context.assets.open("train_${userOf(name)}.bin").use { it.readBytes() })
    } catch (_: java.io.IOException) {
        null
    }

    fun saveCalibrated(name: String, headJson: String) = File(calibratedDir, "$name.json").writeText(headJson)

    fun delete(name: String) {
        if (isCalibrated(name)) File(calibratedDir, "$name.json").delete()
    }
}
