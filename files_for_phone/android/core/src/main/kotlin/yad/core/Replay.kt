package yad.core

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A recording exported by files_for_phone/tools/export_replay.py: consecutive filtered windows
 * with their gesture labels, played back instead of the armband.
 */
class ReplayRecording(val labels: IntArray, val windows: List<EmgWindow>) {
    val size: Int get() = windows.size

    companion object {
        private val MAGIC = "YADR".toByteArray(Charsets.US_ASCII)
        private const val VERSION = 1

        fun parse(bytes: ByteArray): ReplayRecording {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { buf.get(it) }
            require(magic.contentEquals(MAGIC)) { "not a replay file" }
            val version = buf.int
            require(version == VERSION) { "unsupported replay version $version" }
            val n = buf.int
            val channels = buf.int
            val length = buf.int
            require(bytes.size == 20 + 4 * n + 4 * n * channels * length) { "truncated replay file" }
            val labels = IntArray(n) { buf.int }
            val floats = buf.asFloatBuffer()
            val windows = List(n) {
                val data = FloatArray(channels * length)
                floats.get(data)
                EmgWindow(data, channels, length)
            }
            return ReplayRecording(labels, windows)
        }
    }
}
