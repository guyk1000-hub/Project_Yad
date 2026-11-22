package yad.core

/** Class ids used by every model (files_for_pi/realtime_pi_classify.py GESTURE_MAP). */
enum class Gesture(val id: Int) {
    REST(0), CLOSE(1), OPEN(2), RIGHT(3), LEFT(4);

    /** What the hand should do for this gesture; REST sends nothing, like the Pi. */
    val command: HandCommand?
        get() = when (this) {
            REST -> null
            CLOSE -> HandCommand.CLOSE
            OPEN -> HandCommand.OPEN
            RIGHT -> HandCommand.RIGHT
            LEFT -> HandCommand.LEFT
        }

    companion object {
        fun fromId(id: Int): Gesture? = entries.firstOrNull { it.id == id }
    }
}

/** What the hand should do, independent of any hand's protocol (each driver encodes it its own way). */
enum class HandCommand(val title: String) {
    OPEN("OPEN"), CLOSE("CLOSE"), LEFT("LEFT"), RIGHT("RIGHT"),
    STOP("STOP"), STEP("CLOSE A BIT"), CALIBRATE("CALIBRATE"),
}

/**
 * The hands the app can drive, each with what it supports. The app shows only these commands;
 * anything else (e.g. a LEFT gesture on the new hand) is logged and not sent.
 */
enum class HandModel(val title: String, val commands: List<HandCommand>, val speedRange: IntRange?, val defaultSpeed: Int) {
    HAIFA3D_V1("Old hand (Haifa3D v1)", listOf(HandCommand.OPEN, HandCommand.CLOSE, HandCommand.LEFT, HandCommand.RIGHT), null, 0),
    CONTROLLER_V2(
        "New hand (controller v2)",
        listOf(HandCommand.OPEN, HandCommand.CLOSE, HandCommand.STOP, HandCommand.STEP, HandCommand.CALIBRATE),
        HandControllerV2Protocol.SPEED_RANGE, HandControllerV2Protocol.DEFAULT_SPEED,
    );

    fun supports(command: HandCommand) = command in commands
}

/** The Haifa3D hand (first version), from files_for_pi/protocol.py. */
object Haifa3dProtocol {
    /** BLE characteristic the Pi writes commands to (protocol.DIRECT_UUID). */
    const val DIRECT_EXECUTE_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"

    /** The hand the Pi connects to (basic_commands_pi.py). No pairing needed. */
    const val DEFAULT_ADDRESS = "B8:D6:1A:40:EF:D6"

    /** "Direct execute": [5] + 4 bytes; null for what this hand can't do. */
    fun encode(command: HandCommand): ByteArray? {
        val bytes = when (command) {
            HandCommand.OPEN -> intArrayOf(5, 0b11111000, 20, 0b01111000, 0b00000000)
            HandCommand.CLOSE -> intArrayOf(5, 0b11111000, 20, 0b01111000, 0b11111000)
            HandCommand.LEFT -> intArrayOf(5, 0b11111000, 20, 0b10000000, 0b10000000)
            HandCommand.RIGHT -> intArrayOf(5, 0b11111000, 20, 0b10000000, 0b00000000)
            else -> return null
        }
        return ByteArray(bytes.size) { bytes[it].toByte() }
    }
}

/**
 * The new hand (Haifa3D Hand-Controller-System v2.0, used as it is; we don't change it). The hand
 * is the BLE central: the phone acts as its "manager", advertising [SERVICE_UUID]. The hand
 * connects, subscribes to [CHARACTERISTIC_UUID] and runs each notification (managerInterface.cpp):
 * byte 0 = command (the Command enum in defines.h), byte 2 = speed for VELOCITY.
 * The UUIDs are the manager ones in the hand's Hand2_configuration.json.
 */
object HandControllerV2Protocol {
    const val SERVICE_UUID = "0000abcd-0000-1000-8000-00805f9b34fb"
    const val CHARACTERISTIC_UUID = "0000dcba-0000-1000-8000-00805f9b34fb"

    // defines.h: enum Command { STOP, FORWARD, BACKWARD, CLOCKWISE, COUNTER_CLOCKWISE, STEP, POSITION, CALIBRATE, VELOCITY, ... }
    private const val STOP = 0
    private const val FORWARD = 1      // all fingers fully closed
    private const val BACKWARD = 2     // all fingers fully open
    private const val STEP = 5         // all fingers 10% more closed
    private const val CALIBRATE = 7    // re-learn each finger's limits, then open
    private const val VELOCITY = 8

    /** The hand starts at 50 (managerInterface.cpp). Kept below 128: the motor driver takes a signed byte. */
    val SPEED_RANGE = 10..100
    const val DEFAULT_SPEED = 50

    /** 3 bytes for every command (the hand reads up to byte 2); null for what this hand can't do. */
    fun encode(command: HandCommand): ByteArray? = when (command) {
        HandCommand.OPEN -> message(BACKWARD)
        HandCommand.CLOSE -> message(FORWARD)
        HandCommand.STOP -> message(STOP)
        HandCommand.STEP -> message(STEP)
        HandCommand.CALIBRATE -> message(CALIBRATE)
        HandCommand.LEFT, HandCommand.RIGHT -> null    // no wrist on this hand
    }

    fun speed(value: Int): ByteArray {
        require(value in SPEED_RANGE) { "speed $value outside $SPEED_RANGE" }
        return message(VELOCITY, value)
    }

    private fun message(command: Int, value: Int = 0) = byteArrayOf(command.toByte(), 0, value.toByte())
}

/** Where commands go: the real hand over BLE, or a simulated one. */
interface HandSink {
    suspend fun send(command: HandCommand)
}
