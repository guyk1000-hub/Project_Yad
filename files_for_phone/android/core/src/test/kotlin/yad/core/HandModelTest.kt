package yad.core

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HandModelTest {
    @Test
    fun `every hand encodes exactly the commands it offers`() {
        for (c in HandCommand.entries) {
            assertEquals(HandModel.HAIFA3D_V1.supports(c), Haifa3dProtocol.encode(c) != null, "v1 $c")
            assertEquals(HandModel.CONTROLLER_V2.supports(c), HandControllerV2Protocol.encode(c) != null, "v2 $c")
        }
    }

    @Test
    fun `new hand bytes follow the Command enum in defines h`() {
        assertContentEquals(byteArrayOf(1, 0, 0), HandControllerV2Protocol.encode(HandCommand.CLOSE))   // FORWARD
        assertContentEquals(byteArrayOf(2, 0, 0), HandControllerV2Protocol.encode(HandCommand.OPEN))    // BACKWARD
        assertContentEquals(byteArrayOf(0, 0, 0), HandControllerV2Protocol.encode(HandCommand.STOP))
        assertContentEquals(byteArrayOf(5, 0, 0), HandControllerV2Protocol.encode(HandCommand.STEP))
        assertContentEquals(byteArrayOf(7, 0, 0), HandControllerV2Protocol.encode(HandCommand.CALIBRATE))
        assertContentEquals(byteArrayOf(8, 0, 75), HandControllerV2Protocol.speed(75))                 // VELOCITY, byte 2
        assertFailsWith<IllegalArgumentException> { HandControllerV2Protocol.speed(200) }
    }

    @Test
    fun `gestures the new hand can't do send nothing`() {
        assertNull(HandControllerV2Protocol.encode(Gesture.LEFT.command!!))
        assertNotNull(HandControllerV2Protocol.encode(Gesture.CLOSE.command!!))
        assertTrue(HandModel.HAIFA3D_V1.speedRange == null)
    }
}
