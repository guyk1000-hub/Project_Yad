package yad.core

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decoder for the MindRove armband's 216-byte "WUA2" packets, the format this project's armband
 * firmware sends. The MindRove Android SDK can't decode it (it only knows the newer MES_2..5
 * packets); the Python SDK the Pi uses can. This decoder follows that one: WUA2_PACKET::interpret
 * in the mindrove 5.2.3 wheel's libMindRoveWrapper.so.
 *
 * A packet is 2 samples of 108 bytes, little-endian int32 fields:
 *   0..28    8 EMG channels, x 0.045 -> µV (the same scale the Pi's recordings use)
 *   32..68   10 resistance channels (raw)
 *   72       battery, mV -> percent: (mV / 1000 - 2.8) * 100 / 1.45
 *   76       "other" channel (raw; behaves like a sample counter)
 *   80..88   accelerometer x/y/z, x 6.1035e-5 -> g
 *   92..100  gyroscope x/y/z, x 0.01526 -> °/s
 *   104      "other" channel (raw)
 */
object MindRoveWua2 {
    const val PACKET_SIZE = 216
    const val SAMPLE_SIZE = 108
    const val SAMPLES_PER_PACKET = PACKET_SIZE / SAMPLE_SIZE
    const val EMG_SCALE_UV = 0.045
    private const val ACCEL_SCALE_G = 6.1035e-5
    private const val GYRO_SCALE_DPS = 0.01526

    class Sample(
        val emgUv: DoubleArray,
        val resistance: IntArray,
        val batteryPercent: Double,
        val counter: Long,
        val accelG: DoubleArray,
        val gyroDps: DoubleArray,
        val other: Int,
    )

    fun isWua2(length: Int) = length == PACKET_SIZE

    /** Decodes one packet; returns an empty list if [length] isn't a WUA2 packet. */
    fun decode(data: ByteArray, length: Int = data.size, offset: Int = 0): List<Sample> {
        if (!isWua2(length)) return emptyList()
        val b = ByteBuffer.wrap(data, offset, length).order(ByteOrder.LITTLE_ENDIAN)
        return List(SAMPLES_PER_PACKET) { n ->
            val base = offset + n * SAMPLE_SIZE
            fun i32(at: Int) = b.getInt(base + at)
            Sample(
                emgUv = DoubleArray(8) { ch -> i32(ch * 4) * EMG_SCALE_UV },
                resistance = IntArray(10) { i32(32 + it * 4) },
                batteryPercent = (i32(72).toLong().and(0xFFFFFFFFL) / 1000.0 - 2.8) * 100 / 1.45,
                counter = i32(76).toLong() and 0xFFFFFFFFL,
                accelG = DoubleArray(3) { i32(80 + it * 4) * ACCEL_SCALE_G },
                gyroDps = DoubleArray(3) { i32(92 + it * 4) * GYRO_SCALE_DPS },
                other = i32(104),
            )
        }
    }
}
