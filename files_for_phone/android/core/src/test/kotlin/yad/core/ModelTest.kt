package yad.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelTest {
    private val scalar = Golden.json("scalar_features.json")
    private val cnn = Golden.json("cnn_features.json")
    private val headGolden = Golden.json("head_tomer.json")

    @Test
    fun `asym and mag match both feature versions`() {
        val windows = scalar["windows"].jsonArray.map { it.window() }
        for (mode in RmsOver.entries) {
            val v = scalar["versions"][mode.jsonName]
            val asym = v["asym"].doubles()
            val mag = v["mag"].doubles()
            var worst = 0.0
            for ((i, w) in windows.withIndex()) {
                val got = ScalarFeatures.compute(w, mode)
                // Python computes in float32; allow float32 rounding, scaled to the signal size.
                val tol = 1e-5 * max(1.0, mag[i])
                assertTrue(abs(got.asym - asym[i]) <= tol, "$mode asym[$i]: ${got.asym} vs ${asym[i]}")
                assertTrue(abs(got.mag - mag[i]) <= tol, "$mode mag[$i]: ${got.mag} vs ${mag[i]}")
                worst = maxOf(worst, abs(got.asym - asym[i]) / max(1.0, mag[i]), abs(got.mag - mag[i]) / max(1.0, mag[i]))
            }
            println("scalar features $mode: ${windows.size} windows, max relative diff = $worst")
        }
    }

    @Test
    fun `tomer head matches sklearn predict_proba`() {
        val head = MlpHead.fromJson(Golden.modelText("head_tomer.json"))
        assertEquals(RmsOver.fromJson(headGolden["scalar_rms_over"].jsonPrimitive.content), head.rmsOver)
        val inputs = headGolden["inputs"].jsonArray.map { it.floats() }
        val probs = headGolden["probs"].matrix()
        val labels = headGolden["labels"].ints()
        var worst = 0.0
        for (i in inputs.indices) {
            val p = head.predictProba(inputs[i])
            worst = maxOf(worst, maxAbsDiff(p, probs[i]))
            assertEquals(probs[i].argmax(), p.argmax(), "window $i")
            assertEquals(labels[i], head.classes[p.argmax()], "window $i label")
        }
        println("head_tomer: ${inputs.size} windows, max |Δp| = $worst")
        assertTrue(worst < 1e-5, "max |Δp| = $worst")
    }

    @Test
    fun `every exported head loads`() {
        val files = Golden.modelFiles("head_")
        assertTrue(files.isNotEmpty())
        for (f in files) {
            val h = MlpHead.fromJson(f.readText())
            println("${f.name}: ${h.inputSize} inputs, ${h.classes.size} classes, rms over ${h.rmsOver}")
            assertEquals(1026, h.inputSize)
        }
    }

    /** Window -> fake CNN (golden Keras features) -> asym/mag -> head: the full path minus TFLite. */
    @Test
    fun `pipeline classify matches the Python reference`() {
        val windows = cnn["windows"].jsonArray.map { it.window() }
        val features = cnn["features"].jsonArray.map { it.floats() }
        val fake = object : FeatureExtractor {
            override val outputSize = 1024
            override fun extract(window: EmgWindow) = features[windows.indexOfFirst { it.data.contentEquals(window.data) }]
        }
        val head = MlpHead.fromJson(Golden.modelText("head_tomer.json"))
        val pipeline = Pipeline(FilterChain.pi(), Windower(), fake, head, GroupAverageDecisionPolicy())
        val wantInputs = headGolden["inputs"].jsonArray.map { it.floats() }
        val wantProbs = headGolden["probs"].matrix()
        for ((i, w) in windows.withIndex()) {
            val input = ScalarFeatures.headInput(fake.extract(w), w, head.rmsOver)
            for (j in 0 until 1024) assertEquals(wantInputs[i][j], input[j], "cnn feature $j of window $i")
            val r = pipeline.classify(w)
            assertTrue(maxAbsDiff(r.probabilities, wantProbs[i]) < 1e-5, "window $i probabilities")
        }
    }
}
