package yad.app

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import yad.core.EmgWindow
import yad.core.FeatureExtractor
import yad.core.MlpHead
import yad.core.ScalarFeatures
import kotlin.math.abs
import kotlin.math.max

/**
 * Runs on the phone at startup: TFLite CNN + tomer head on the golden windows (assets/golden),
 * compared with the Python/Keras reference. The same check as export_cnn.py, but on this device.
 */
object SelfTest {
    data class Result(val passed: Boolean, val summary: String)

    fun run(context: Context, extractor: FeatureExtractor, tomerHead: MlpHead): Result {
        fun load(name: String): JsonElement =
            Json.parseToJsonElement(context.assets.open("golden/$name").bufferedReader().use { it.readText() })

        val cnn = load("cnn_features.json")
        val headGolden = load("head_tomer.json")
        val windows = cnn.jsonObject.getValue("windows").jsonArray.map { w ->
            val rows = w.jsonArray
            val len = rows[0].jsonArray.size
            val data = FloatArray(rows.size * len)
            rows.forEachIndexed { c, row -> row.jsonArray.forEachIndexed { t, v -> data[c * len + t] = v.jsonPrimitive.double.toFloat() } }
            EmgWindow(data, rows.size, len)
        }
        val wantFeatures = cnn.jsonObject.getValue("features").jsonArray.map { r -> r.jsonArray.map { it.jsonPrimitive.double } }
        val wantProbs = headGolden.jsonObject.getValue("probs").jsonArray.map { r -> r.jsonArray.map { it.jsonPrimitive.double } }
        val labels = headGolden.jsonObject.getValue("labels").jsonArray.map { it.jsonPrimitive.int }

        var featErr = 0.0
        var probErr = 0.0
        var agree = 0
        for ((i, w) in windows.withIndex()) {
            val f = extractor.extract(w)
            val scale = wantFeatures[i].maxOf { abs(it) }
            for (j in f.indices) featErr = max(featErr, abs(f[j] - wantFeatures[i][j]) / scale)
            val p = tomerHead.predictProba(ScalarFeatures.headInput(f, w, tomerHead.rmsOver))
            for (k in p.indices) probErr = max(probErr, abs(p[k] - wantProbs[i][k]))
            if (tomerHead.classes[p.indices.maxBy { p[it] }] == labels[i]) agree++
        }
        val passed = featErr < 1e-4 && agree == windows.size
        val summary = "On-device check ${if (passed) "PASSED" else "FAILED"}: " +
            "CNN rel. error %.1e, probabilities ±%.1e, %d/%d windows correct".format(featErr, probErr, agree, windows.size)
        return Result(passed, summary)
    }
}
