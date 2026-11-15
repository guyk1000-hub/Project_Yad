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
 * Golden windows (assets/golden) with the Python/Keras reference outputs: the same check as
 * export_cnn.py, run on this phone. Used at startup and for every benchmark engine.
 */
class GoldenCheck(context: Context) {
    data class Result(val featureError: Double, val probError: Double, val correct: Int, val total: Int) {
        val passed: Boolean get() = featureError < 1e-4 && correct == total
        val summary: String
            get() = "CNN rel. error %.1e, probabilities ±%.1e, %d/%d windows correct".format(featureError, probError, correct, total)
    }

    private fun load(context: Context, name: String): JsonElement =
        Json.parseToJsonElement(context.assets.open("golden/$name").bufferedReader().use { it.readText() })

    val windows: List<EmgWindow>
    private val wantFeatures: List<List<Double>>
    private val wantProbs: List<List<Double>>
    private val labels: List<Int>

    init {
        val cnn = load(context, "cnn_features.json").jsonObject
        val head = load(context, "head_tomer.json").jsonObject
        windows = cnn.getValue("windows").jsonArray.map { w ->
            val rows = w.jsonArray
            val len = rows[0].jsonArray.size
            val data = FloatArray(rows.size * len)
            rows.forEachIndexed { c, row -> row.jsonArray.forEachIndexed { t, v -> data[c * len + t] = v.jsonPrimitive.double.toFloat() } }
            EmgWindow(data, rows.size, len)
        }
        wantFeatures = cnn.getValue("features").jsonArray.map { r -> r.jsonArray.map { it.jsonPrimitive.double } }
        wantProbs = head.getValue("probs").jsonArray.map { r -> r.jsonArray.map { it.jsonPrimitive.double } }
        labels = head.getValue("labels").jsonArray.map { it.jsonPrimitive.int }
    }

    /** [tomerHead] must be head_tomer.json: the golden probabilities come from that model. */
    fun run(extractor: FeatureExtractor, tomerHead: MlpHead): Result {
        var featErr = 0.0
        var probErr = 0.0
        var correct = 0
        for ((i, w) in windows.withIndex()) {
            val f = extractor.extract(w)
            val scale = wantFeatures[i].maxOf { abs(it) }
            for (j in f.indices) featErr = max(featErr, abs(f[j] - wantFeatures[i][j]) / scale)
            val p = tomerHead.predictProba(ScalarFeatures.headInput(f, w, tomerHead.rmsOver))
            for (k in p.indices) probErr = max(probErr, abs(p[k] - wantProbs[i][k]))
            if (tomerHead.classes[p.indices.maxBy { p[it] }] == labels[i]) correct++
        }
        return Result(featErr, probErr, correct, windows.size)
    }
}
