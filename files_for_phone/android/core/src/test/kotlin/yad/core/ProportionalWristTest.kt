package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ProportionalWristTest {
    @Test
    fun `stronger contraction turns the wrist further per decision`() {
        val w = ProportionalWrist(reference = 20.0)
        val light = w.timeUnits(w.ratio(12.0), 200, 1.0)
        val typical = w.timeUnits(w.ratio(20.0), 200, 1.0)
        val strong = w.timeUnits(w.ratio(40.0), 200, 1.0)
        assertTrue(light < typical && typical < strong, "$light $typical $strong")
        assertEquals(2, typical)      // half of the 4 units until the next decision
        assertEquals(4, strong)       // all of it: continuous turning
        assertEquals(1, light)        // never zero
        assertEquals(4, w.timeUnits(w.ratio(20.0), 200, 2.0))   // sensitivity scales it
    }

    @Test
    fun `without recordings it learns the user's typical effort live`() {
        val w = ProportionalWrist(reference = null)
        repeat(10) { w.ratio(10.0) }
        assertEquals(2.0, w.ratio(20.0), 1e-9)
    }

    @Test
    fun `reference comes from the LEFT and RIGHT windows of the recordings`() {
        val set = TrainingSet(listOf(floatArrayOf(0f, 5f), floatArrayOf(0f, 30f), floatArrayOf(0f, 10f)), intArrayOf(Gesture.CLOSE.id, Gesture.LEFT.id, Gesture.RIGHT.id))
        assertEquals(30.0, ProportionalWrist.reference(listOf(set))!!, 1e-9)
        val step = ProportionalWrist.step(left = true, units = 3)
        val m = Haifa3dProtocol.decodeAction(step)!!.single()
        assertEquals(setOf(Haifa3dProtocol.Motor.TURN), m.motors)
        assertEquals(setOf(Haifa3dProtocol.Motor.TURN), m.closing)
        assertEquals(3, m.timeUnits)
    }
}
