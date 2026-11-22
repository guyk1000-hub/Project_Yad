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
enum class HandCommand { OPEN, CLOSE, LEFT, RIGHT }

/** The Haifa3D hand (first version), from files_for_pi/protocol.py. */
object Haifa3dProtocol {
    /** BLE characteristic the Pi writes commands to (protocol.DIRECT_UUID). */
    const val DIRECT_EXECUTE_UUID = "e0198000-7544-42c1-0001-b24344b6aa70"

    /** The hand the Pi connects to (basic_commands_pi.py). No pairing needed. */
    const val DEFAULT_ADDRESS = "B8:D6:1A:40:EF:D6"

    /** "Direct execute": [5] + 4 bytes. */
    fun encode(command: HandCommand): ByteArray {
        val bytes = when (command) {
            HandCommand.OPEN -> intArrayOf(5, 0b11111000, 20, 0b01111000, 0b00000000)
            HandCommand.CLOSE -> intArrayOf(5, 0b11111000, 20, 0b01111000, 0b11111000)
            HandCommand.LEFT -> intArrayOf(5, 0b11111000, 20, 0b10000000, 0b10000000)
            HandCommand.RIGHT -> intArrayOf(5, 0b11111000, 20, 0b10000000, 0b00000000)
        }
        return ByteArray(bytes.size) { bytes[it].toByte() }
    }
}

/** Where commands go: the real hand over BLE, or a simulated one. */
interface HandSink {
    suspend fun send(command: HandCommand)
}
