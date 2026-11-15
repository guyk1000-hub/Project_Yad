package yad.app

import android.content.Context
import yad.core.MlpHead
import yad.core.ReplayRecording
import java.io.File

/**
 * The user models: the ones bundled in the APK (copied from files_for_phone/models at build time;
 * offered when head_<user>.json and replay_<user>.bin exist), plus versions made on this phone,
 * stored as files/models/<name>.json:
 * - <user>@<yyyyMMdd_HHmm[ss]>: calibrated (the PC recording + a short phone recording);
 * - <user>-phone@<yyyyMMdd_HHmmss>: trained from zero on the user's phone recordings.
 * Names are never reused: saving refuses to overwrite.
 *
 * The phone recordings live in files/recordings/<user>/rec_<user>_<time>.bin, each with its head
 * inputs cached next to it (.feat). A phone-made version uses its user's bundled replay.
 */
class ModelCatalog(private val context: Context) {
    companion object {
        const val PHONE_SUFFIX = "-phone"
    }

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
                .sortedByDescending { it.substringAfter('@') }
            return bundled + calibrated
        }

    fun userOf(name: String): String = name.substringBefore('@').removeSuffix(PHONE_SUFFIX)

    /** Made on this phone (calibrated or trained from zero), so it can be deleted. */
    fun isPhoneMade(name: String) = '@' in name

    /** Newest phone-made version of [user], if any. */
    fun latestFor(user: String): String? =
        names.filter { isPhoneMade(it) && userOf(it) == user }.maxByOrNull { it.substringAfter('@') }

    /** Newest version of [user] trained from zero on phone recordings, if any. */
    fun latestPhoneTrained(user: String): String? =
        names.filter { it.startsWith("$user$PHONE_SUFFIX@") }.maxByOrNull { it.substringAfter('@') }

    fun head(name: String): MlpHead = MlpHead.fromJson(
        if (isPhoneMade(name)) File(calibratedDir, "$name.json").readText()
        else context.assets.open("head_$name.json").bufferedReader().use { it.readText() }
    )

    fun replay(name: String): ReplayRecording =
        ReplayRecording.parse(context.assets.open("replay_${userOf(name)}.bin").use { it.readBytes() })

    fun saveCalibrated(name: String, headJson: String) {
        val f = File(calibratedDir, "$name.json")
        check(!f.exists()) { "a model named $name already exists" }
        f.writeText(headJson)
    }

    fun recordingsDir(user: String) = File(context.filesDir, "recordings/$user").apply { mkdirs() }

    /** The user's phone recordings, oldest first. */
    fun recordings(user: String): List<File> =
        recordingsDir(user).listFiles { f -> f.name.endsWith(".bin") }.orEmpty().sortedBy { it.name }

    /** Stores a recording under [fileName] (made unique if taken); returns the file. */
    fun saveRecording(user: String, fileName: String, bytes: ByteArray): File {
        val dir = recordingsDir(user)
        val base = fileName.removeSuffix(".bin")
        var f = File(dir, "$base.bin")
        var k = 2
        while (f.exists()) f = File(dir, "${base}_${k++}.bin")
        f.writeBytes(bytes)
        return f
    }

    fun deleteRecording(f: File) {
        f.delete()
        File(f.path.removeSuffix(".bin") + ".feat").delete()
    }

    fun delete(name: String) {
        if (isPhoneMade(name)) File(calibratedDir, "$name.json").delete()
    }
}
