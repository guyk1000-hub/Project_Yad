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

class Haifa3dV1Test {
    private val P = Haifa3dProtocol

    @Test
    fun `the four Pi commands come out byte for byte the same`() {
        // files_for_pi/protocol.py: [5] + [torque, time, motors, direction]
        assertContentEquals(byteArrayOf(5, 0b11111000.toByte(), 20, 0b01111000, 0), P.encode(HandCommand.OPEN))
        assertContentEquals(byteArrayOf(5, 0b11111000.toByte(), 20, 0b01111000, 0b11111000.toByte()), P.encode(HandCommand.CLOSE))
        assertContentEquals(byteArrayOf(5, 0b11111000.toByte(), 20, 0b10000000.toByte(), 0b10000000.toByte()), P.encode(HandCommand.LEFT))
        assertContentEquals(byteArrayOf(5, 0b11111000.toByte(), 20, 0b10000000.toByte(), 0), P.encode(HandCommand.RIGHT))
    }

    @Test
    fun `actions with several movements round trip`() {
        val tap = listOf(
            Haifa3dProtocol.Movement(setOf(Haifa3dProtocol.Motor.FINGER1), setOf(Haifa3dProtocol.Motor.FINGER1), emptySet(), 6),
            Haifa3dProtocol.Movement(setOf(Haifa3dProtocol.Motor.FINGER1), emptySet(), emptySet(), 6),
        )
        val bytes = P.action(tap)
        assertEquals(9, bytes.size)
        assertEquals(9, bytes[0].toInt())
        assertEquals(tap, P.decodeAction(bytes))
        assertNull(P.decodeAction(byteArrayOf(1, 2, 3)))
    }

    @Test
    fun `uuids and live control`() {
        assertEquals("e0198001-7544-42c1-100b-b24344b6aa70", P.presetUuid(11))
        assertEquals("e0198003-7544-42c1-1013-b24344b6aa70", P.configUuid(19))
        assertEquals("e0198003-7544-42c1-0101-b24344b6aa70", P.configResetUuid(1))
        // finger 2 closing, wrist right: 1 unit, low torque
        val live = P.liveStep(mapOf(Haifa3dProtocol.Motor.FINGER2 to true, Haifa3dProtocol.Motor.TURN to false))
        assertContentEquals(byteArrayOf(5, 0, 1, 0b10100000.toByte(), 0b00100000), live)
    }
}
