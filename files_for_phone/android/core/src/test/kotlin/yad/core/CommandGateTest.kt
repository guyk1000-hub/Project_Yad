package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class CommandGateTest {
    @Test
    fun `classifier commands pass like the Pi, rest sends nothing`() {
        val gate = CommandGate()
        assertEquals(HandCommand.CLOSE, gate.fromClassifier(Gesture.CLOSE))
        assertEquals(HandCommand.CLOSE, gate.fromClassifier(Gesture.CLOSE))  // the Pi repeats too
        assertNull(gate.fromClassifier(Gesture.REST))
    }

    @Test
    fun `stop blocks the classifier but not manual commands`() {
        val gate = CommandGate()
        gate.stop()
        for (g in Gesture.entries) assertNull(gate.fromClassifier(g), "$g while stopped")
        assertEquals(HandCommand.OPEN, gate.manual(HandCommand.OPEN))
        gate.resume()
        assertEquals(HandCommand.LEFT, gate.fromClassifier(Gesture.LEFT))
    }

    @Test
    fun `send only on change suppresses repeats`() {
        val gate = CommandGate(sendOnlyOnChange = true)
        assertEquals(HandCommand.OPEN, gate.fromClassifier(Gesture.OPEN))
        assertNull(gate.fromClassifier(Gesture.OPEN))
        assertEquals(HandCommand.CLOSE, gate.fromClassifier(Gesture.CLOSE))
        gate.stop(); gate.resume()
        assertEquals(HandCommand.CLOSE, gate.fromClassifier(Gesture.CLOSE), "resume forgets the last command")
    }
}
