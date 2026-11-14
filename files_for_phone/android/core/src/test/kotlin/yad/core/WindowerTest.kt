package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class WindowerTest {
    private fun sample(v: Double) = DoubleArray(2) { ch -> v * 10 + ch }

    @Test
    fun `non-overlapping windows like the Pi`() {
        val w = Windower(channels = 2, length = 4)
        val out = (0 until 12).mapNotNull { w.push(sample(it.toDouble())) }
        assertEquals(3, out.size)
        assertEquals(listOf(40f, 50f, 60f, 70f), (0 until 4).map { out[1][0, it] })
        assertEquals(listOf(81f, 91f, 101f, 111f), (0 until 4).map { out[2][1, it] })
    }

    @Test
    fun `overlapping windows with a smaller hop`() {
        val w = Windower(channels = 2, length = 4, hop = 2)
        val out = (0 until 10).mapNotNull { w.push(sample(it.toDouble())) }
        // first at sample 3, then every 2 samples: 3, 5, 7, 9
        assertEquals(4, out.size)
        assertEquals(listOf(20f, 30f, 40f, 50f), (0 until 4).map { out[1][0, it] })
        assertEquals(listOf(60f, 70f, 80f, 90f), (0 until 4).map { out[3][0, it] })
    }

    @Test
    fun `no window before the first one is full`() {
        val w = Windower(channels = 2, length = 4)
        repeat(3) { assertNull(w.push(sample(it.toDouble()))) }
    }
}
