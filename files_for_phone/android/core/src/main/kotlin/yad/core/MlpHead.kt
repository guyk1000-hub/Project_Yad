package yad.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.tanh

/**
 * A user's classifier: StandardScaler + sklearn MLPClassifier, loaded from the JSON written
 * by files_for_phone/tools/export_head.py. Computes the same probabilities as sklearn's
 * predict_proba.
 */
class MlpHead private constructor(
    val modelName: String,
    val classes: IntArray,
    val classNames: Map<Int, String>,
    val rmsOver: RmsOver,
    private val mean: DoubleArray,
    private val scale: DoubleArray,
    private val layers: List<Layer>,
    private val hidden: (Double) -> Double,
) {
    /** weights[i * nOut + j]: out[j] = bias[j] + sum_i in[i] * weights[i][j] */
    private class Layer(val nIn: Int, val nOut: Int, val weights: DoubleArray, val bias: DoubleArray)

    val inputSize: Int get() = mean.size

    fun predictProba(input: FloatArray): DoubleArray {
        require(input.size == inputSize) { "$modelName expects $inputSize features, got ${input.size}" }
        var a = DoubleArray(inputSize) { (input[it].toDouble() - mean[it]) / scale[it] }
        for ((li, layer) in layers.withIndex()) {
            val out = layer.bias.copyOf()
            for (i in 0 until layer.nIn) {
                val ai = a[i]
                val row = i * layer.nOut
                for (j in 0 until layer.nOut) out[j] += ai * layer.weights[row + j]
            }
            if (li < layers.lastIndex) for (j in out.indices) out[j] = hidden(out[j])
            a = out
        }
        return softmax(a)
    }

    private fun softmax(z: DoubleArray): DoubleArray {
        var m = Double.NEGATIVE_INFINITY
        for (v in z) m = max(m, v)
        val e = DoubleArray(z.size) { exp(z[it] - m) }
        val s = e.sum()
        for (i in e.indices) e[i] /= s
        return e
    }

    @Serializable
    private data class HeadJson(
        val format: String,
        val format_version: Int,
        val model_name: String,
        val input: Input,
        val classes: List<Int>,
        val class_names: Map<String, String>,
        val scaler: Scaler,
        val hidden_activation: String,
        val output_activation: String,
        val layers: List<LayerJson>,
    )

    @Serializable
    private data class Input(val n_features: Int, val scalar_rms_over: String)

    @Serializable
    private data class Scaler(val mean: List<Double>, val scale: List<Double>)

    @Serializable
    private data class LayerJson(val weights: List<List<Double>>, val bias: List<Double>)

    companion object {
        private const val FORMAT = "yad-head"
        private const val FORMAT_VERSION = 1
        private val json = Json { ignoreUnknownKeys = true }

        fun fromJson(text: String): MlpHead {
            val h = json.decodeFromString<HeadJson>(text)
            require(h.format == FORMAT && h.format_version == FORMAT_VERSION) {
                "not a $FORMAT v$FORMAT_VERSION file (got ${h.format} v${h.format_version})"
            }
            require(h.output_activation == "softmax") { "unsupported output activation ${h.output_activation}" }
            val hidden: (Double) -> Double = when (h.hidden_activation) {
                "tanh" -> ::tanh
                "relu" -> { x -> max(x, 0.0) }
                "logistic" -> { x -> 1 / (1 + exp(-x)) }
                "identity" -> { x -> x }
                else -> throw IllegalArgumentException("unsupported hidden activation ${h.hidden_activation}")
            }
            require(h.scaler.mean.size == h.input.n_features && h.scaler.scale.size == h.input.n_features) {
                "scaler size doesn't match n_features=${h.input.n_features}"
            }
            var nIn = h.input.n_features
            val layers = h.layers.map { l ->
                require(l.weights.size == nIn) { "layer expects $nIn inputs, has ${l.weights.size}" }
                val nOut = l.bias.size
                val w = DoubleArray(nIn * nOut)
                l.weights.forEachIndexed { i, row ->
                    require(row.size == nOut) { "ragged weight row" }
                    row.forEachIndexed { j, v -> w[i * nOut + j] = v }
                }
                Layer(nIn, nOut, w, l.bias.toDoubleArray()).also { nIn = nOut }
            }
            require(nIn == h.classes.size) { "output size $nIn != ${h.classes.size} classes" }
            return MlpHead(
                modelName = h.model_name,
                classes = h.classes.toIntArray(),
                classNames = h.class_names.mapKeys { it.key.toInt() },
                rmsOver = RmsOver.fromJson(h.input.scalar_rms_over),
                mean = h.scaler.mean.toDoubleArray(),
                scale = h.scaler.scale.toDoubleArray(),
                layers = layers,
                hidden = hidden,
            )
        }
    }
}
