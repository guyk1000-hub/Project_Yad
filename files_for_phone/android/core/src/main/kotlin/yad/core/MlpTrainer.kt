package yad.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

/** Head inputs (1026 features) with labels and per-sample weights. */
class TrainingSet(val features: List<FloatArray>, val labels: IntArray, val weights: DoubleArray = DoubleArray(labels.size) { 1.0 }) {
    init {
        require(features.size == labels.size && labels.size == weights.size) { "sizes differ" }
    }

    val size: Int get() = labels.size

    operator fun plus(other: TrainingSet) =
        TrainingSet(features + other.features, labels + other.labels, weights + other.weights)

    /** Same data with every weight multiplied by [factor]. */
    fun weighted(factor: Double) = TrainingSet(features, labels, DoubleArray(size) { weights[it] * factor })

    companion object {
        private val MAGIC = "YADT".toByteArray(Charsets.US_ASCII)

        /** Reads a file written by files_for_phone/tools/export_train_features.py. */
        fun parse(bytes: ByteArray): Pair<TrainingSet, RmsOver> {
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { b.get(it) }
            require(magic.contentEquals(MAGIC)) { "not a training-features file" }
            require(b.int == 1) { "unsupported version" }
            val n = b.int
            val d = b.int
            val rmsOver = RmsOver.entries[b.int]
            require(bytes.size == 20 + 8 * n + 4 * n * d) { "truncated training-features file" }
            val labels = IntArray(n) { b.int }
            repeat(n) { b.int }                     // takes: not needed for training
            val floats = b.asFloatBuffer()
            val features = List(n) { FloatArray(d).also { floats.get(it) } }
            return TrainingSet(features, labels) to rmsOver
        }
    }
}

/**
 * Trains the same classifier as training_data/fine_tune.py on the phone: StandardScaler +
 * MLP (32, 16) with tanh, softmax output, cross-entropy + L2 (alpha = 0.01) on the weights,
 * optimised with L-BFGS, as sklearn's MLPClassifier(solver="lbfgs") does. Sample weights behave
 * like repeating samples. The result is a head in the same JSON format as export_head.py.
 */
class MlpTrainer(
    val hidden: IntArray = intArrayOf(32, 16),
    val alpha: Double = 0.01,
    val maxIterations: Int = 300,
    val gradientTolerance: Double = 1e-4,
    val seed: Int = 42,
) {
    class Result(
        val mean: DoubleArray,
        val scale: DoubleArray,
        val weights: List<DoubleArray>,   // weights[l][i * nOut + j]
        val biases: List<DoubleArray>,
        val sizes: IntArray,              // layer sizes, input first
        val classes: IntArray,
        val iterations: Int,
        val loss: Double,
    ) {
        /** sklearn-compatible head JSON (format yad-head v1), loadable with MlpHead.fromJson. */
        fun toHeadJson(modelName: String, rmsOver: RmsOver, classNames: Map<Int, String>, note: String): String {
            val json = buildJsonObject {
                put("format", "yad-head")
                put("format_version", 1)
                put("model_name", modelName)
                putJsonObject("source") { put("trained_on", note) }
                putJsonObject("input") {
                    put("n_features", sizes[0])
                    put("scalar_rms_over", rmsOver.jsonName)
                }
                put("classes", JsonArray(classes.map { JsonPrimitive(it) }))
                putJsonObject("class_names") { classes.forEach { put(it.toString(), classNames[it] ?: it.toString()) } }
                putJsonObject("scaler") {
                    put("mean", JsonArray(mean.map { JsonPrimitive(it) }))
                    put("scale", JsonArray(scale.map { JsonPrimitive(it) }))
                }
                put("hidden_activation", "tanh")
                put("output_activation", "softmax")
                putJsonArray("layers") {
                    for (l in weights.indices) {
                        val nIn = sizes[l]
                        val nOut = sizes[l + 1]
                        add(buildJsonObject {
                            put("weights", buildJsonArray {
                                for (i in 0 until nIn) add(JsonArray(List(nOut) { j -> JsonPrimitive(weights[l][i * nOut + j]) }))
                            })
                            put("bias", JsonArray(biases[l].map { JsonPrimitive(it) }))
                        })
                    }
                }
            }
            return json.toString()
        }
    }

    /** [onProgress] gets (iteration, loss) and returns false to cancel. */
    fun train(data: TrainingSet, onProgress: (Int, Double) -> Boolean = { _, _ -> true }): Result {
        val classes = data.labels.distinct().sorted().toIntArray()
        val classIndex = classes.withIndex().associate { (i, c) -> c to i }
        val n = data.size
        val d = data.features[0].size
        val w = data.weights
        val wSum = w.sum()

        // Weighted StandardScaler (equal to sklearn's on repeated samples).
        val mean = DoubleArray(d)
        for (s in 0 until n) { val f = data.features[s]; for (k in 0 until d) mean[k] += w[s] * f[k] }
        for (k in 0 until d) mean[k] /= wSum
        val variance = DoubleArray(d)
        for (s in 0 until n) { val f = data.features[s]; for (k in 0 until d) { val e = f[k] - mean[k]; variance[k] += w[s] * e * e } }
        val scale = DoubleArray(d) { k -> sqrt(variance[k] / wSum).let { if (it < 1e-12) 1.0 else it } }
        val x = Array(n) { s -> val f = data.features[s]; DoubleArray(d) { k -> (f[k] - mean[k]) / scale[k] } }
        val y = IntArray(n) { classIndex.getValue(data.labels[it]) }

        val sizes = intArrayOf(d, *hidden, classes.size)
        val net = Net(sizes)
        val params = net.initial(Random(seed))
        val objective = { p: DoubleArray, grad: DoubleArray -> net.lossAndGradient(p, x, y, w, wSum, alpha, grad) }
        val (best, iterations, loss) = Lbfgs(maxIterations, gradientTolerance).minimize(params, objective, onProgress)
        val (ws, bs) = net.unpack(best)
        return Result(mean, scale, ws, bs, sizes, classes, iterations, loss)
    }

    /** Fully connected tanh network with softmax output; parameters packed as [W0, b0, W1, b1, ...]. */
    private class Net(val sizes: IntArray) {
        val layers = sizes.size - 1
        val wOffset = IntArray(layers)
        val bOffset = IntArray(layers)
        val count: Int

        init {
            var o = 0
            for (l in 0 until layers) {
                wOffset[l] = o; o += sizes[l] * sizes[l + 1]
                bOffset[l] = o; o += sizes[l + 1]
            }
            count = o
        }

        /** Glorot-uniform init, as sklearn does for tanh. */
        fun initial(r: Random): DoubleArray {
            val p = DoubleArray(count)
            for (l in 0 until layers) {
                val bound = sqrt(6.0 / (sizes[l] + sizes[l + 1]))
                for (i in 0 until sizes[l] * sizes[l + 1]) p[wOffset[l] + i] = r.nextDouble(-bound, bound)
                for (j in 0 until sizes[l + 1]) p[bOffset[l] + j] = r.nextDouble(-bound, bound)
            }
            return p
        }

        fun unpack(p: DoubleArray): Pair<List<DoubleArray>, List<DoubleArray>> = Pair(
            List(layers) { l -> p.copyOfRange(wOffset[l], wOffset[l] + sizes[l] * sizes[l + 1]) },
            List(layers) { l -> p.copyOfRange(bOffset[l], bOffset[l] + sizes[l + 1]) },
        )

        /** Weighted mean cross-entropy + alpha/2 * |W|^2 / sum(w) (sklearn's loss); fills [grad]. */
        fun lossAndGradient(p: DoubleArray, x: Array<DoubleArray>, y: IntArray, w: DoubleArray, wSum: Double,
                            alpha: Double, grad: DoubleArray): Double {
            grad.fill(0.0)
            var loss = 0.0
            val acts = Array(layers + 1) { DoubleArray(sizes[it]) }
            val deltas = Array(layers + 1) { DoubleArray(sizes[it]) }
            for (s in x.indices) {
                acts[0] = x[s]
                for (l in 0 until layers) {
                    val nIn = sizes[l]; val nOut = sizes[l + 1]
                    val a = acts[l]; val z = acts[l + 1]
                    for (j in 0 until nOut) z[j] = p[bOffset[l] + j]
                    val wo = wOffset[l]
                    for (i in 0 until nIn) {
                        val ai = a[i]
                        if (ai == 0.0) continue
                        val row = wo + i * nOut
                        for (j in 0 until nOut) z[j] += ai * p[row + j]
                    }
                    if (l < layers - 1) for (j in 0 until nOut) z[j] = tanh(z[j])
                }
                // softmax + cross-entropy
                val out = acts[layers]
                var m = Double.NEGATIVE_INFINITY
                for (v in out) m = max(m, v)
                var sum = 0.0
                for (j in out.indices) { out[j] = exp(out[j] - m); sum += out[j] }
                for (j in out.indices) out[j] /= sum
                val ws = w[s] / wSum
                loss -= ws * ln(max(out[y[s]], 1e-15))
                val dOut = deltas[layers]
                for (j in out.indices) dOut[j] = ws * (out[j] - if (j == y[s]) 1.0 else 0.0)
                // backward
                for (l in layers - 1 downTo 0) {
                    val nIn = sizes[l]; val nOut = sizes[l + 1]
                    val a = acts[l]; val dz = deltas[l + 1]
                    val wo = wOffset[l]; val bo = bOffset[l]
                    for (j in 0 until nOut) grad[bo + j] += dz[j]
                    val da = deltas[l]
                    if (l > 0) da.fill(0.0)
                    for (i in 0 until nIn) {
                        val ai = a[i]
                        val row = wo + i * nOut
                        var acc = 0.0
                        for (j in 0 until nOut) {
                            grad[row + j] += ai * dz[j]
                            acc += p[row + j] * dz[j]
                        }
                        if (l > 0) da[i] = acc * (1 - ai * ai)   // tanh'
                    }
                }
            }
            // L2 on weights only (not biases), scaled by 1/sum(w) like sklearn's 1/n_samples
            var l2 = 0.0
            for (l in 0 until layers) for (i in 0 until sizes[l] * sizes[l + 1]) {
                val v = p[wOffset[l] + i]
                l2 += v * v
                grad[wOffset[l] + i] += alpha * v / wSum
            }
            return loss + 0.5 * alpha * l2 / wSum
        }
    }
}

/** Limited-memory BFGS with a backtracking (Armijo) line search. */
class Lbfgs(val maxIterations: Int, val gradientTolerance: Double, val memory: Int = 10) {
    data class Outcome(val x: DoubleArray, val iterations: Int, val loss: Double)

    fun minimize(x0: DoubleArray, f: (DoubleArray, DoubleArray) -> Double, onProgress: (Int, Double) -> Boolean): Outcome {
        val n = x0.size
        var x = x0.copyOf()
        var g = DoubleArray(n)
        var fx = f(x, g)
        val sHist = ArrayDeque<DoubleArray>()
        val yHist = ArrayDeque<DoubleArray>()
        val rhoHist = ArrayDeque<Double>()
        var iter = 0
        while (iter < maxIterations) {
            if (g.maxOf { abs(it) } < gradientTolerance) break
            // two-loop recursion: direction = -H g
            val q = g.copyOf()
            val alphas = DoubleArray(sHist.size)
            for (k in sHist.indices.reversed()) {
                alphas[k] = rhoHist[k] * dot(sHist[k], q)
                axpy(-alphas[k], yHist[k], q)
            }
            if (sHist.isNotEmpty()) {
                val gamma = dot(sHist.last(), yHist.last()) / dot(yHist.last(), yHist.last())
                for (i in q.indices) q[i] *= gamma
            } else {
                val gn = sqrt(dot(g, g))
                for (i in q.indices) q[i] /= max(gn, 1e-12)
            }
            for (k in sHist.indices) {
                val beta = rhoHist[k] * dot(yHist[k], q)
                axpy(alphas[k] - beta, sHist[k], q)
            }
            val dir = DoubleArray(n) { -q[it] }
            var slope = dot(g, dir)
            if (slope >= 0) {                        // not a descent direction: restart along -g
                sHist.clear(); yHist.clear(); rhoHist.clear()
                for (i in dir.indices) dir[i] = -g[i]
                slope = -dot(g, g)
            }
            // backtracking line search
            var step = 1.0
            val xNew = DoubleArray(n)
            val gNew = DoubleArray(n)
            var fNew: Double
            var tries = 0
            while (true) {
                for (i in 0 until n) xNew[i] = x[i] + step * dir[i]
                fNew = f(xNew, gNew)
                if (fNew <= fx + 1e-4 * step * slope || tries >= 30) break
                step *= 0.5
                tries++
            }
            if (fNew > fx) break                     // no progress possible
            val s = DoubleArray(n) { xNew[it] - x[it] }
            val yv = DoubleArray(n) { gNew[it] - g[it] }
            val sy = dot(s, yv)
            if (sy > 1e-10) {
                if (sHist.size == memory) { sHist.removeFirst(); yHist.removeFirst(); rhoHist.removeFirst() }
                sHist.addLast(s); yHist.addLast(yv); rhoHist.addLast(1.0 / sy)
            }
            val improvement = fx - fNew
            x = xNew.copyOf(); g = gNew.copyOf(); fx = fNew
            iter++
            if (!onProgress(iter, fx)) break
            if (improvement <= 1e-12 * max(1.0, abs(fx))) break
        }
        return Outcome(x, iter, fx)
    }

    private fun dot(a: DoubleArray, b: DoubleArray): Double { var s = 0.0; for (i in a.indices) s += a[i] * b[i]; return s }
    private fun axpy(a: Double, x: DoubleArray, y: DoubleArray) { for (i in x.indices) y[i] += a * x[i] }
}
