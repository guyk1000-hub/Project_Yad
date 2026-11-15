package yad.core

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GuidedCheckTest {
    /**
     * Runs a full check on a synthetic stream through the real filters and windower, with a fake
     * classifier that is right only for CLOSE. Writes the file to build/ so the Python reader
     * (tools/analyze_check.py --selftest) can be checked against the same bytes.
     */
    @Test
    fun `scores each gesture and writes a readable file`() {
        val classes = intArrayOf(0, 1, 2, 3, 4)
        val check = GuidedCheck("test", "time", classes)
        val filters = FilterChain.pi()
        val windower = Windower()
        val total = check.phases.sumOf { it.seconds } * check.sampleRate
        var phase = 0
        var phaseEnd = check.phases[0].seconds * check.sampleRate
        for (n in 0 until total) {
            if (n == phaseEnd) { phase++; phaseEnd += check.phases[phase].seconds * check.sampleRate }
            val raw = DoubleArray(8) { ch -> 1000.0 * ch + 50 * sin(n * 0.3 + ch) }
            check.addSample(raw.copyOf(), phase)
            val w = windower.push(filters.process(raw)) ?: continue
            val label = check.phases[phase].label
            val predicted = if (label == Gesture.CLOSE.id) label else Gesture.REST.id
            check.addWindow(phase, predicted, DoubleArray(5) { if (it == predicted) 0.9 else 0.025 }, w.length)
        }
        val r = check.results()
        println(r.joinToString("\n"))
        assertTrue(r[1].startsWith("CLOSE  100%"), r[1])
        assertTrue(r[2].startsWith("OPEN     0%"), r[2])
        assertTrue(r[0].startsWith("REST   100%"), r[0])
        // 6 s per gesture = 30 windows; the first second (5 windows) isn't scored.
        assertTrue(r[1].contains("(25/25)"), r[1])

        val bytes = check.toBytes()
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("YADC", String(bytes, 0, 4))
        b.position(4)
        assertEquals(1, b.int)
        val headerLen = b.int
        val header = String(bytes, 12, headerLen)
        assertTrue(header.contains("\"scalar_rms_over\":\"time\""), header)
        b.position(12 + headerLen)
        assertEquals(total, b.int)

        val out = File("build/guided_check_sample.bin")
        out.parentFile.mkdirs()
        out.writeBytes(bytes)
        println("wrote ${out.absolutePath}")
    }
}
