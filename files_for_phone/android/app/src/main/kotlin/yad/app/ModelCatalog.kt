package yad.app

import android.content.Context
import yad.core.MlpHead
import yad.core.ReplayRecording

/**
 * The models and recordings bundled in the APK (copied from files_for_phone/models at build time).
 * A model is offered when both head_<name>.json and replay_<name>.bin exist.
 */
class ModelCatalog(private val context: Context) {
    val names: List<String> by lazy {
        val files = context.assets.list("").orEmpty().toSet()
        files.filter { it.startsWith("head_") && it.endsWith(".json") }
            .map { it.removePrefix("head_").removeSuffix(".json") }
            .filter { "replay_$it.bin" in files }
            .sorted()
    }

    fun head(name: String): MlpHead =
        MlpHead.fromJson(context.assets.open("head_$name.json").bufferedReader().use { it.readText() })

    fun replay(name: String): ReplayRecording =
        ReplayRecording.parse(context.assets.open("replay_$name.bin").use { it.readBytes() })
}
