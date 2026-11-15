package yad.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RecordingTest {
    @Test
    fun `a recording repeats the gestures per take with breaks between`() {
        val r = GuidedCheck("tomer", "time", intArrayOf(0, 1, 2, 3, 4), phaseSeconds = 10, takes = 3)
        assertEquals(1 + 3 * 5 + 2, r.phases.size)
        assertEquals(listOf(-1, 0, 1, 2, 3, 4, -1, 0), r.phases.take(8).map { it.label })
        assertEquals("OPEN (take 3/3)", r.phases.last { it.label == Gesture.OPEN.id }.title)
        assertEquals(3 + 150 + 10, r.phases.sumOf { it.seconds })
        assertEquals("OPEN", GuidedCheck("t", "time", intArrayOf(0, 1, 2, 3, 4)).phases[3].title)
    }

    @Test
    fun `samples read back from the file`() {
        val r = GuidedCheck("tomer", "time", intArrayOf(0, 1, 2, 3, 4), takes = 2)
        for (n in 0 until 700) r.addSample(DoubleArray(8) { n * 10.0 + it }, if (n < 500) 0 else 1)
        val (raw, labels) = GuidedCheck.readSamples(r.toBytes())
        assertEquals(700, raw.size)
        assertEquals(r.rawLabelsList, labels)
        assertContentEquals(DoubleArray(8) { 6990.0 + it }, raw.last())
        assertFailsWith<IllegalArgumentException> { GuidedCheck.readSamples("nope".toByteArray()) }
    }

    @Test
    fun `cached features round trip`() {
        val set = TrainingSet(listOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(-4f, 5.5f, 0f)), intArrayOf(2, 4))
        val (back, rms) = TrainingSet.parse(set.toBytes(RmsOver.TIME))
        assertEquals(RmsOver.TIME, rms)
        assertContentEquals(set.labels, back.labels)
        assertContentEquals(set.features[1], back.features[1])
    }
}
