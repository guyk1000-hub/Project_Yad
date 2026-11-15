package yad.core

import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MlpTrainerTest {
    /** Synthetic 3-class data: class = which of 3 noisy centres a point comes from. */
    private fun blobs(n: Int, d: Int, r: Random): TrainingSet {
        val centres = List(3) { DoubleArray(d) { r.nextDouble(-3.0, 3.0) } }
        val labels = IntArray(n) { it % 3 }
        val feats = labels.map { c -> FloatArray(d) { k -> (centres[c][k] + r.nextDouble(-1.0, 1.0)).toFloat() } }
        return TrainingSet(feats, labels)
    }

    @Test
    fun `trains to high accuracy and the result loads as a head`() {
        val train = blobs(300, 20, Random(1))
        val result = MlpTrainer(hidden = intArrayOf(8, 4), maxIterations = 200).train(train)
        val head = MlpHead.fromJson(result.toHeadJson("synthetic", RmsOver.CHANNELS, emptyMap(), "unit test"))
        val correct = train.features.indices.count { s ->
            val p = head.predictProba(train.features[s])
            head.classes[p.indices.maxBy { p[it] }] == train.labels[s]
        }
        println("synthetic: ${result.iterations} iterations, loss ${result.loss}, accuracy ${correct / 300.0}")
        assertTrue(correct >= 297, "accuracy $correct/300")
    }

    @Test
    fun `sample weights act like repeating samples`() {
        val r = Random(2)
        val base = blobs(60, 6, r)
        val repeated = TrainingSet(base.features + base.features.take(20), base.labels + base.labels.take(20))
        val weights = DoubleArray(60) { if (it < 20) 2.0 else 1.0 }
        val weighted = TrainingSet(base.features, base.labels, weights)
        val t = MlpTrainer(hidden = intArrayOf(5, 4), maxIterations = 50)
        val a = t.train(repeated)
        val b = t.train(weighted)
        val diff = a.weights[0].indices.maxOf { abs(a.weights[0][it] - b.weights[0][it]) }
        assertTrue(diff < 1e-6, "weights differ by $diff")
        assertTrue(abs(a.mean[0] - b.mean[0]) < 1e-9 && abs(a.scale[0] - b.scale[0]) < 1e-9)
    }

    @Test
    fun `training features file parses`() {
        val f = File(System.getProperty("yad.models.dir")).resolve("train_tomer.bin")
        val (set, rms) = TrainingSet.parse(f.readBytes())
        assertEquals(1450, set.size)
        assertEquals(1026, set.features[0].size)
        assertEquals(RmsOver.TIME, rms)
    }
}

/**
 * Kotlin trainer vs sklearn on real data: old recording + calibration windows from the
 * 14 Nov guided check (hop 50, weighted to ~37% of the total), tested on the last 3 s of each
 * gesture. Reference (sklearn, 300 iterations): guy 77.0%, tomer 98.7%. The data comes from
 * tools/out/scratch/calib_recipe.py and isn't committed, so the test skips without it.
 */
class TrainerVsSklearnTest {
    private val scratch = File(System.getProperty("yad.models.dir")).resolve("../tools/out/scratch")

    private fun load(name: String) = TrainingSet.parse(scratch.resolve(name).readBytes()).first

    @Test
    fun `matches sklearn accuracy on the 14 Nov checks`() {
        if (!scratch.resolve("golden_calib_guy.bin").isFile) { println("no golden calibration data, skipping"); return }
        for ((user, reference) in listOf("guy" to 0.770, "tomer" to 0.987)) {
            val old = TrainingSet.parse(File(System.getProperty("yad.models.dir")).resolve("train_$user.bin").readBytes()).first
            val calib = load("golden_calib_$user.bin")
            val test = load("golden_test_$user.bin")
            val share = 0.37
            val w = share / (1 - share) * old.size / calib.size
            val t0 = System.nanoTime()
            val r = MlpTrainer(maxIterations = 300).train(old + calib.weighted(w))
            val secs = (System.nanoTime() - t0) / 1e9
            val head = MlpHead.fromJson(r.toHeadJson(user, RmsOver.CHANNELS, emptyMap(), "test"))
            val correct = test.features.indices.count { s ->
                val p = head.predictProba(test.features[s]); head.classes[p.indices.maxBy { p[it] }] == test.labels[s]
            }
            val acc = correct.toDouble() / test.size
            println("$user: kotlin %.1f%% vs sklearn %.1f%%, %d iterations, %.0f s on this PC".format(100 * acc, 100 * reference, r.iterations, secs))
            assertTrue(acc >= reference - 0.05, "$user: $acc vs sklearn $reference")
        }
    }
}
