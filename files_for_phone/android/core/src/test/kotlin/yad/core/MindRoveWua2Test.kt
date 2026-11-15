package yad.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MindRoveWua2Test {
    /** The first 32 bytes of a real packet from the armband (diagnostics, 2026-11-14). */
    private val realPrefix = "0e 70 02 00 34 a4 02 00 8f 4f fc ff dc 51 fe ff 4b 65 fc ff 7f cc ff ff d1 a5 03 00 6b 80 fd ff"
        .split(" ").map { it.toInt(16).toByte() }.toByteArray()

    @Test
    fun `real packet prefix decodes to the 8 EMG channels`() {
        val packet = ByteArray(216).also { realPrefix.copyInto(it) }
        val s = MindRoveWua2.decode(packet)
        assertEquals(2, s.size)
        val raw = intArrayOf(159758, 173108, -241777, -110116, -236213, -13185, 239057, -163733)
        for (ch in 0 until 8) assertEquals(raw[ch] * 0.045, s[0].emgUv[ch], "channel ${ch + 1}")
    }

    @Test
    fun `every field lands at the offset WUA2_PACKET reads`() {
        val buf = ByteBuffer.allocate(216).order(ByteOrder.LITTLE_ENDIAN)
        for (n in 0 until 2) {
            val base = n * 108
            for (ch in 0 until 8) buf.putInt(base + ch * 4, (n + 1) * 1000 + ch)
            for (r in 0 until 10) buf.putInt(base + 32 + r * 4, 500 + r)
            buf.putInt(base + 72, 3_525)          // mV -> (3.525 - 2.8) * 100 / 1.45 = 50 %
            buf.putInt(base + 76, 41 + n)         // counter
            for (a in 0 until 3) buf.putInt(base + 80 + a * 4, 16384 * (a + 1))
            for (g in 0 until 3) buf.putInt(base + 92 + g * 4, -100 * (g + 1))
            buf.putInt(base + 104, 7)
        }
        val s = MindRoveWua2.decode(buf.array())
        for (n in 0 until 2) {
            for (ch in 0 until 8) assertEquals(((n + 1) * 1000 + ch) * 0.045, s[n].emgUv[ch])
            assertEquals((0 until 10).map { 500 + it }, s[n].resistance.toList())
            assertTrue(abs(s[n].batteryPercent - 50.0) < 1e-9, "battery ${s[n].batteryPercent}")
            assertEquals(41L + n, s[n].counter)
            assertTrue(abs(s[n].accelG[0] - 16384 * 6.1035e-5) < 1e-12)
            assertEquals(-100 * 0.01526, s[n].gyroDps[0])
            assertEquals(7, s[n].other)
        }
    }

    @Test
    fun `other sizes are not WUA2`() {
        assertEquals(0, MindRoveWua2.decode(ByteArray(100)).size)
        assertEquals(0, MindRoveWua2.decode(ByteArray(217), 217).size)
    }
}
