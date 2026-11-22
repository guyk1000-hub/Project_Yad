package yad.app

import android.content.Context
import yad.core.MlpHead
import yad.core.ModelNames
import yad.core.ReplayRecording
import java.io.File

/**
 * The user models: the ones bundled in the APK (copied from files_for_phone/models at build time;
 * offered when head_<user>.json and replay_<user>.bin exist), plus versions made on this phone,
 * stored as files/models/<name>.json:
 * - <user>@<yyyyMMdd_HHmm[ss]>: calibrated (the PC recording + a short phone recording);
 * - <user>-<label>@<yyyyMMdd_HHmmss>: trained from zero on the user's phone recordings.
 * (see yad.core.ModelNames). Names are never reused: saving refuses to overwrite.
 *
 * The phone recordings live in files/recordings/<user>/rec_<user>_<time>.bin, each with its head
 * inputs cached next to it (.feat). A phone-made version uses its user's bundled replay, if any.
 *
 * People: everyone with a bundled model, plus anyone added on the phone (a recordings folder). A new
 * person has no model until their first recording is trained.
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
                .sortedByDescending { it.substringAfter('@') }
            return bundled + calibrated
        }

    fun userOf(name: String): String = ModelNames.userOf(name)

    /**
     * The people to choose from: added on this phone, plus bundled users. Bundled variants of a
     * person's PC models (tomer_long, tomer_250) are left to the engineering view.
     */
    val people: List<String>
        get() {
            val added = File(context.filesDir, "recordings").listFiles { f -> f.isDirectory }.orEmpty().map { it.name }
            return (bundled.filter { '_' !in it } + added).distinct().sortedBy { it.lowercase() }
        }

    fun hasBundledModel(person: String) = person in bundled

    /** A person's name: letters, digits, '_' (spaces become '_'). Returns the stored name, or null if invalid / taken. */
    fun addPerson(name: String): String? {
        val clean = name.trim().replace(Regex("\\s+"), "_").filter { it.isLetterOrDigit() || it == '_' }.take(24)
        if (clean.isEmpty() || people.any { it.equals(clean, ignoreCase = true) }) return null
        recordingsDir(clean)
        return clean
    }

    /** The model to use for [person]: their newest phone-made one, else their bundled one, else none yet. */
    fun modelFor(person: String): String? = latestFor(person) ?: person.takeIf { it in bundled }

    /** Made on this phone (calibrated or trained from zero), so it can be deleted. */
    fun isPhoneMade(name: String) = '@' in name

    /** Newest phone-made version of [user], if any. */
    fun latestFor(user: String): String? =
        names.filter { isPhoneMade(it) && userOf(it) == user }.maxByOrNull { it.substringAfter('@') }

    /** Newest version of [user] trained from zero on phone recordings, if any. */
    fun latestPhoneTrained(user: String): String? =
        names.filter { ModelNames.isPhoneTrained(it) && userOf(it) == user }.maxByOrNull { it.substringAfter('@') }

    fun head(name: String): MlpHead = MlpHead.fromJson(
        if (isPhoneMade(name)) File(calibratedDir, "$name.json").readText()
        else context.assets.open("head_$name.json").bufferedReader().use { it.readText() }
    )

    /** The person's bundled recording to replay, or null (people added on the phone have none). */
    fun replay(name: String): ReplayRecording? = try {
        ReplayRecording.parse(context.assets.open("replay_${userOf(name)}.bin").use { it.readBytes() })
    } catch (_: java.io.IOException) {
        null
    }

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
