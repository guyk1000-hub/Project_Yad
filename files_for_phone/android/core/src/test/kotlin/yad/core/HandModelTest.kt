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

class HandPresetsTest {
    @Test
    fun `finger position bytes`() {
        assertContentEquals(byteArrayOf(6, 2, 40), HandControllerV2Protocol.position(2, 40))   // [POSITION, M2, 40%]
        assertFailsWith<IllegalArgumentException> { HandControllerV2Protocol.position(5, 40) }
        assertFailsWith<IllegalArgumentException> { HandControllerV2Protocol.position(1, 101) }
    }

    @Test
    fun `presets survive export and import`() {
        val custom = HandPresets.DEFAULTS + HandPreset("Point", listOf(PresetAction.AllFingers(100), PresetAction.Finger(1, 0)))
        val back = HandPresets.fromJson(HandPresets.toJson(custom))
        assertEquals(custom, back)
        assertEquals(8, HandPresets.DEFAULTS.size)
        assertEquals("Index to 0%", HandPresets.describe(PresetAction.Finger(1, 0), HandModel.CONTROLLER_V2.fingers))
    }

    @Test
    fun `bad preset files are refused with a reason`() {
        assertFailsWith<IllegalArgumentException> { HandPresets.fromJson("{}") }
        assertFailsWith<IllegalArgumentException> { HandPresets.fromJson("""[{"name":"x","actions":[{"type":"all","percent":300}]}]""") }
        assertFailsWith<IllegalArgumentException> { HandPresets.fromJson("""[{"name":"x","actions":[{"type":"finger","finger":9,"percent":10}]}]""") }
    }
}
