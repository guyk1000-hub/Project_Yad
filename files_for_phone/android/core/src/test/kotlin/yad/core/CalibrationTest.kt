package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CalibrationTest {
    @Test
    fun `windows stay inside one gesture and skip its first second`() {
        val fs = 500
        // 1 s getting ready (-1), then 3 s of gesture 1, then 3 s of gesture 2
        val labels = List(fs) { -1 } + List(3 * fs) { 1 } + List(3 * fs) { 2 }
        val raw = labels.indices.map { n -> DoubleArray(8) { ch -> (n % 7) * 10.0 + ch } }
        val w = Calibration.windows(raw, labels)
        // per gesture: starts from +1 s to (3 s - 100 samples), every 25 -> (1000 - 100) / 25 + 1 = 37
        assertEquals(37, w.count { it.second == 1 })
        assertEquals(37, w.count { it.second == 2 })
        assertTrue(w.none { it.second == -1 })
    }

    @Test
    fun `new data gets the configured share of the weight`() {
        val w = Calibration.newWeight(1450, 200)
        assertEquals(Calibration.NEW_SHARE, 200 * w / (1450 + 200 * w), 1e-12)
    }
}
