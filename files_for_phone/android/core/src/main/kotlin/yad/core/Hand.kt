package yad.core

import kotlinx.serialization.Serializable

/**
 * Class ids used by every model: 0–4 as files_for_pi/realtime_pi_classify.py GESTURE_MAP, then the
 * finger gestures, which only models trained on the "with fingers" gesture set know.
 */
enum class Gesture(val id: Int, val prompt: String) {
    REST(0, "Relax your hand"),
    CLOSE(1, "Make a fist"),
    OPEN(2, "Open your hand wide"),
    RIGHT(3, "Turn your wrist right"),
    LEFT(4, "Turn your wrist left"),
    POINT(5, "Point with your index finger"),
    PEACE(6, "Peace sign: index and middle finger up");

    /** What the hand should do for this gesture; REST sends nothing, like the Pi. */
    val command: HandCommand?
        get() = when (this) {
            REST -> null
            CLOSE -> HandCommand.CLOSE
            OPEN -> HandCommand.OPEN
            RIGHT -> HandCommand.RIGHT
            LEFT -> HandCommand.LEFT
            POINT -> HandCommand.POINT
            PEACE -> HandCommand.PEACE
        }

    companion object {
        fun fromId(id: Int): Gesture? = entries.firstOrNull { it.id == id }
    }
}

/** What the hand should do, independent of any hand's protocol (each driver encodes it its own way). */
@Serializable
enum class HandCommand(val title: String) {
    OPEN("OPEN"), CLOSE("CLOSE"), LEFT("LEFT"), RIGHT("RIGHT"),
    STOP("STOP"), STEP("CLOSE A BIT"), CALIBRATE("CALIBRATE"),

    /** Absolute poses: every finger is driven to its place, so the result is the same from open or closed. */
    POINT("POINT"), PEACE("PEACE"),
}

/** Which gestures a recording asks for (and so which a model trained on it knows). */
enum class GestureSet(val title: String, val gestures: List<Gesture>) {
    BASIC("Hand (5 gestures)", listOf(Gesture.REST, Gesture.CLOSE, Gesture.OPEN, Gesture.RIGHT, Gesture.LEFT)),
    FINGERS("Hand + fingers (7)", BASIC.gestures + listOf(Gesture.POINT, Gesture.PEACE)),
}

/**
 * The hands the app can drive, each with what it supports. The app shows only these commands;
 * anything else (e.g. a LEFT gesture on the new hand) is logged and not sent.
 */
enum class HandModel(
    val title: String,
    val commands: List<HandCommand>,
    val speedRange: IntRange?,
    val defaultSpeed: Int,
    /** Fingers that can be moved one by one (POSITION), numbered from 1; empty if the hand can't. */
    val fingers: List<String> = emptyList(),
    /** What the hand's own manager app offers but this hand's code ignores: shown, not sent. */
    val notYetSupported: List<String> = emptyList(),
) {
    HAIFA3D_V1(
        "Old hand (Haifa3D v1)",
        listOf(HandCommand.OPEN, HandCommand.CLOSE, HandCommand.LEFT, HandCommand.RIGHT, HandCommand.POINT, HandCommand.PEACE), null, 0,
    ),
    CONTROLLER_V2(
        "New hand (controller v2)",
        listOf(HandCommand.OPEN, HandCommand.CLOSE, HandCommand.POINT, HandCommand.PEACE, HandCommand.STOP, HandCommand.STEP, HandCommand.CALIBRATE),
        HandControllerV2Protocol.SPEED_RANGE, HandControllerV2Protocol.DEFAULT_SPEED,
        fingers = listOf("Index", "Middle", "Ring", "Pinky"),
        notYetSupported = listOf("Wrist (LEFT / RIGHT)", "Subscribe", "Special commands 1–8", "Packet", "Screen packet"),
    );

    fun supports(command: HandCommand) = command in commands
}

/**
 * The Haifa3D hand (first version). Protocol from Haifa3D's public haifa3d-hand-app (Protocol.md,
 * the app's BLE services) and files_for_pi/protocol.py. The phone connects to the hand (no pairing).
 *
 * An action is [1 + 4n] followed by n movements of 4 bytes each: torque stop (bit set = high
 * torque), time stop (units of 50 ms), motors activated, motors direction. Bits: 7 = turn
 * (wrist), 6..3 = fingers 1..4. Direction bit 1 = close (fingers) / left (turn).
 */
object Haifa3dProtocol {
    const val DIRECT_EXECUTE_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"
    const val TRIGGER_UUID = "e0198002-7544-42c1-0001-b24344b6aa70"
    const val BATTERY_SERVICE_UUID = "0000180f-0000-1000-8000-00805f9b34fb"
    const val BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    const val PRESETS = 12
    const val MS_PER_TIME_UNIT = 50

    /** The hand the Pi connects to (basic_commands_pi.py). No pairing needed. */
    const val DEFAULT_ADDRESS = "B8:D6:1A:40:EF:D6"

    fun presetUuid(n: Int): String {
        require(n in 0 until PRESETS) { "preset $n outside 0..${PRESETS - 1}" }
        return "e0198001-7544-42c1-10%02x-b24344b6aa70".format(n)
    }

    fun configUuid(id: Int): String {
        require(id in 0..20) { "config id $id outside 0..20" }
        return "e0198003-7544-42c1-10%02x-b24344b6aa70".format(id)
    }

    /** 0 = reset presets, 1 = reset configuration (write 1 to trigger). */
    fun configResetUuid(which: Int): String {
        require(which in 0..1)
        return "e0198003-7544-42c1-01%02x-b24344b6aa70".format(which)
    }

    enum class Motor(val title: String, val bit: Int) {
        TURN("Wrist", 7), FINGER1("Finger 1", 6), FINGER2("Finger 2", 5), FINGER3("Finger 3", 4), FINGER4("Finger 4", 3);

        val mask get() = 1 shl bit
    }

    /** One movement: [motors] move ([closing]: direction bit set) until [timeUnits] × 50 ms or the torque limit. */
    @Serializable
    data class Movement(
        val motors: Set<Motor>,
        val closing: Set<Motor> = emptySet(),
        val highTorque: Set<Motor> = emptySet(),
        val timeUnits: Int = 20,
    ) {
        init {
            require(timeUnits in 0..255) { "time $timeUnits outside 0..255" }
        }

        fun toBytes(): ByteArray = byteArrayOf(mask(highTorque), timeUnits.toByte(), mask(motors), mask(closing))

        fun describe(): String {
            if (motors.isEmpty()) return "nothing (${timeUnits * MS_PER_TIME_UNIT} ms)"
            val parts = motors.sortedBy { -it.bit }.map { m ->
                val dir = if (m == Motor.TURN) (if (m in closing) "left" else "right") else (if (m in closing) "close" else "open")
                "${m.title} $dir"
            }
            return parts.joinToString(", ") + " · ${timeUnits * MS_PER_TIME_UNIT} ms · " +
                (if (highTorque.containsAll(motors)) "high" else if (highTorque.isEmpty()) "low" else "mixed") + " torque"
        }

        companion object {
            private fun mask(set: Set<Motor>) = set.fold(0) { acc, m -> acc or m.mask }.toByte()

            fun fromBytes(b: ByteArray, at: Int): Movement {
                fun set(x: Byte) = Motor.entries.filter { (x.toInt() and it.mask) != 0 }.toSet()
                return Movement(set(b[at + 2]), set(b[at + 3]), set(b[at]), b[at + 1].toInt() and 0xFF)
            }
        }
    }

    fun action(movements: List<Movement>): ByteArray {
        require(movements.isNotEmpty() && 1 + 4 * movements.size <= 255) { "an action needs 1..63 movements" }
        return byteArrayOf((1 + 4 * movements.size).toByte()) + movements.flatMap { it.toBytes().toList() }
    }

    /** Reads an action (e.g. a preset read from the hand); null if the bytes aren't one. */
    fun decodeAction(b: ByteArray): List<Movement>? {
        if (b.size < 5 || (b.size - 1) % 4 != 0) return null
        return (0 until (b.size - 1) / 4).map { Movement.fromBytes(b, 1 + 4 * it) }
    }

    private val FINGERS = setOf(Motor.FINGER1, Motor.FINGER2, Motor.FINGER3, Motor.FINGER4)
    private val ALL = Motor.entries.toSet()

    /** The Pi's four commands: high torque on every motor, 1 s. */
    fun encode(command: HandCommand): ByteArray? = when (command) {
        HandCommand.OPEN -> action(listOf(Movement(FINGERS, emptySet(), ALL, 20)))
        HandCommand.CLOSE -> action(listOf(Movement(FINGERS, ALL, ALL, 20)))   // the Pi also sets the (unused) wrist direction bit
        HandCommand.LEFT -> action(listOf(Movement(setOf(Motor.TURN), setOf(Motor.TURN), ALL, 20)))
        HandCommand.RIGHT -> action(listOf(Movement(setOf(Motor.TURN), emptySet(), ALL, 20)))
        // One movement, every finger driven to its end stop in its own direction: same pose from open or closed.
        HandCommand.POINT -> action(listOf(Movement(FINGERS, setOf(Motor.FINGER2, Motor.FINGER3, Motor.FINGER4), ALL, 20)))
        HandCommand.PEACE -> action(listOf(Movement(FINGERS, setOf(Motor.FINGER3, Motor.FINGER4), ALL, 20)))
        else -> null
    }

    /** Live control, like the Haifa3D app: one 50 ms low-torque step for each held motor, repeated while held. */
    fun liveStep(held: Map<Motor, Boolean>): ByteArray =
        action(listOf(Movement(held.keys, held.filterValues { it }.keys, emptySet(), 1)))

    /** Configuration values (Protocol.md; the Haifa3D app's ConfigurationService). One byte each. */
    data class ConfigValue(val id: Int, val short: String, val title: String, val isSwitch: Boolean = false)

    val CONFIG = listOf(
        ConfigValue(0, "LTV", "Low torque value"),
        ConfigValue(1, "HTV", "High torque value"),
        ConfigValue(2, "TS", "Torque measure start (ms)"),
        ConfigValue(3, "WW", "Window width filter (samples)"),
        ConfigValue(4, "LTS", "Low torque slope"),
        ConfigValue(5, "HTS", "High torque slope"),
        ConfigValue(6, "TF0", "Threshold factor, motor 0 (%)"),
        ConfigValue(7, "TF1", "Threshold factor, motor 1 (%)"),
        ConfigValue(8, "TF2", "Threshold factor, motor 2 (%)"),
        ConfigValue(9, "TF3", "Threshold factor, motor 3 (%)"),
        ConfigValue(10, "TF4", "Threshold factor, motor 4 (%)"),
        ConfigValue(18, "DV", "Threshold value unit"),
        ConfigValue(19, "DEB", "Debug output over USB", isSwitch = true),
    )
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
    private const val POSITION = 6     // [6, finger, percent]
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
        HandCommand.POINT, HandCommand.PEACE -> null   // several messages: see messages()
    }

    /** Everything to send for [command], in order; null if this hand can't do it. Poses set every finger. */
    fun messages(command: HandCommand): List<ByteArray>? = when (command) {
        HandCommand.POINT -> pose(0, 100, 100, 100)
        HandCommand.PEACE -> pose(0, 0, 100, 100)
        else -> encode(command)?.let { listOf(it) }
    }

    /** Positions for fingers 1..4 (index, middle, ring, pinky), 0 = open, 100 = closed. */
    private fun pose(vararg percent: Int) = percent.mapIndexed { i, p -> position(i + 1, p) }

    /** One finger (1 = M1 … 4 = M4) to [percent] of its travel, 0 = open, 100 = closed. */
    fun position(finger: Int, percent: Int): ByteArray {
        require(finger in 1..4) { "finger $finger outside 1..4" }
        require(percent in 0..100) { "position $percent outside 0..100" }
        return message(POSITION, percent, finger)
    }

    fun speed(value: Int): ByteArray {
        require(value in SPEED_RANGE) { "speed $value outside $SPEED_RANGE" }
        return message(VELOCITY, value)
    }

    private fun message(command: Int, value: Int = 0, arg: Int = 0) = byteArrayOf(command.toByte(), arg.toByte(), value.toByte())
}

/** Where commands go: the real hand over BLE, or a simulated one. */
interface HandSink {
    suspend fun send(command: HandCommand)
}
