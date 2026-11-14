package yad.core

import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals

class FilterTest {
    private val g = Golden.json("filters.json")

    @Test
    fun `coefficients match the Pi's filters`() {
        val chain = FilterChain.pi(g["sampling_rate"].jsonPrimitive.double.toInt())
        val expected = g["filters"].jsonArray
        assertEquals(expected.size, chain.filters.size)
        for ((f, e) in chain.filters.zip(expected)) {
            val got = doubleArrayOf(f.a0, f.a1, f.a2, f.b1, f.b2)
            val want = listOf("a0", "a1", "a2", "b1", "b2").map { e[it].jsonPrimitive.double }.toDoubleArray()
            assertEquals(0.0, maxAbsDiff(got, want), "coefficients of ${f.type}")
        }
    }

    @Test
    fun `streaming filter output matches the Pi bit for bit`() {
        val raw = g["raw"].matrix()
        val want = g["filtered"].matrix()
        val chain = FilterChain.pi()
        var worst = 0.0
        for ((i, sample) in raw.withIndex()) {
            worst = maxOf(worst, maxAbsDiff(chain.process(sample.copyOf()), want[i]))
        }
        println("filters: ${raw.size} samples, max |diff| = $worst")
        assertEquals(0.0, worst)
    }

    @Test
    fun `Pi-style windows equal the filtered stream cast to float32`() {
        val raw = g["raw"].matrix()
        val want = g["filtered"].matrix()
        val len = g["window_len"].jsonPrimitive.double.toInt()
        val chain = FilterChain.pi()
        val windower = Windower(length = len)
        val windows = raw.mapNotNull { windower.push(chain.process(it.copyOf())) }
        assertEquals(raw.size / len, windows.size)
        for ((w, win) in windows.withIndex()) {
            for (ch in 0 until 8) for (t in 0 until len) {
                assertEquals(want[w * len + t][ch].toFloat(), win[ch, t], "window $w ch $ch t $t")
            }
        }
    }
}
