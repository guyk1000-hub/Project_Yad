package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelLineageTest {
    @Test
    fun `a calibration extends exactly its parent's recordings`() {
        val first = ModelLineage("guy", listOf("rec_guy_1.bin"))
        val calibrated = first.extendedWith("rec_guy_2.bin", parent = "guy-morning@20261122_171436")
        assertEquals(listOf("rec_guy_1.bin", "rec_guy_2.bin"), calibrated.recordings)
        assertEquals("guy", calibrated.person)
        assertEquals(calibrated, ModelLineage.fromJson(calibrated.toJson()))
    }

    @Test
    fun `older models are read from their note`() {
        val note = "trained on the phone from zero: 2 recordings (rec_guy_20261122_170928.bin, rec_guy_20261122_171830.bin), 2492 windows, 300 iterations"
        assertEquals(listOf("rec_guy_20261122_170928.bin", "rec_guy_20261122_171830.bin"), ModelLineage.fromLegacyNote("guy", note).recordings)
        assertTrue(ModelLineage.fromLegacyNote("tomer", "calibrated on the phone: 885 new windows + 1450 from the original recording").pcData)
    }
}
