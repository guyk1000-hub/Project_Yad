package yad.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

class ReplayTest {
    @Test
    fun `replay file parses with balanced labels`() {
        val f = File(System.getProperty("yad.models.dir")).resolve("replay_tomer.bin")
        val r = ReplayRecording.parse(f.readBytes())
        assertEquals(435, r.size)
        assertEquals(List(5) { 87 }, r.labels.toList().groupingBy { it }.eachCount().toSortedMap().values.toList())
        assertEquals(8, r.windows[0].channels)
        assertEquals(100, r.windows[0].length)
    }
}
