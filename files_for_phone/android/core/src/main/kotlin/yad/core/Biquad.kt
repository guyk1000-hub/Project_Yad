package yad.core

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tan

enum class FilterType { LOWPASS, HIGHPASS, BANDPASS, NOTCH, PEAK, LOWSHELF, HIGHSHELF }

/**
 * Port of BiquadMultiChan in training_data/utils.py: one biquad section with independent
 * state per channel. The arithmetic follows the Python expression by expression, so
 * results are identical in double precision (verified against tools/golden/filters.json).
 *
 * @param fc cutoff as a fraction of the sampling rate (e.g. 50.0 / 500)
 */
class BiquadMultiChan(
    val channels: Int,
    val type: FilterType,
    val fc: Double,
    val q: Double,
    val peakGainDb: Double,
) {
    var a0 = 0.0; private set
    var a1 = 0.0; private set
    var a2 = 0.0; private set
    var b1 = 0.0; private set
    var b2 = 0.0; private set

    private val z1 = DoubleArray(channels)
    private val z2 = DoubleArray(channels)

    init {
        calcBiquad()
    }

    fun process(input: Double, channel: Int): Double {
        val output = input * a0 + z1[channel]
        z1[channel] = input * a1 + z2[channel] - b1 * output
        z2[channel] = input * a2 - b2 * output
        return output
    }

    fun reset() {
        z1.fill(0.0)
        z2.fill(0.0)
    }

    private fun calcBiquad() {
        val v = 10.0.pow(abs(peakGainDb) / 20.0)
        val k = tan(PI * fc)
        val norm: Double
        when (type) {
            FilterType.LOWPASS -> {
                norm = 1 / (1 + k / q + k * k)
                a0 = k * k * norm
                a1 = 2 * a0
                a2 = a0
                b1 = 2 * (k * k - 1) * norm
                b2 = (1 - k / q + k * k) * norm
            }
            FilterType.HIGHPASS -> {
                norm = 1 / (1 + k / q + k * k)
                a0 = 1 * norm
                a1 = -2 * a0
                a2 = a0
                b1 = 2 * (k * k - 1) * norm
                b2 = (1 - k / q + k * k) * norm
            }
            FilterType.BANDPASS -> {
                norm = 1.0 / (1.0 + k / q + k * k)
                a0 = (k / q) * norm
                a1 = 0.0
                a2 = -a0
                b1 = 2.0 * (k * k - 1.0) * norm
                b2 = (1.0 - k / q + k * k) * norm
            }
            FilterType.NOTCH -> {
                norm = 1.0 / (1.0 + k / q + k * k)
                a0 = (1 + k * k) * norm
                a1 = 2.0 * (k * k - 1) * norm
                a2 = a0
                b1 = a1
                b2 = (1.0 - k / q + k * k) * norm
            }
            FilterType.PEAK -> if (peakGainDb >= 0) {
                norm = 1 / (1 + (1 / q) * k + k * k)
                a0 = (1 + (v / q) * k + k * k) * norm
                a1 = 2 * (k * k - 1) * norm
                a2 = (1 - (v / q) * k + k * k) * norm
                b1 = a1
                b2 = (1 - (1 / q) * k + k * k) * norm
            } else {
                norm = 1 / (1 + (v / q) * k + k * k)
                a0 = (1 + (1 / q) * k + k * k) * norm
                a1 = 2 * (k * k - 1) * norm
                a2 = (1 - (1 / q) * k + k * k) * norm
                b1 = a1
                b2 = (1 - (v / q) * k + k * k) * norm
            }
            FilterType.LOWSHELF -> if (peakGainDb >= 0) {
                norm = 1 / (1 + sqrt(2.0) * k + k * k)
                a0 = (1 + sqrt(2 * v) * k + v * k * k) * norm
                a1 = 2 * (v * k * k - 1) * norm
                a2 = (1 - sqrt(2 * v) * k + v * k * k) * norm
                b1 = 2 * (k * k - 1) * norm
                b2 = (1 - sqrt(2.0) * k + k * k) * norm
            } else {
                norm = 1 / (1 + sqrt(2 * v) * k + v * k * k)
                a0 = (1 + sqrt(2.0) * k + k * k) * norm
                a1 = 2 * (k * k - 1) * norm
                a2 = (1 - sqrt(2.0) * k + k * k) * norm
                b1 = 2 * (v * k * k - 1) * norm
                b2 = (1 - sqrt(2 * v) * k + v * k * k) * norm
            }
            FilterType.HIGHSHELF -> if (peakGainDb >= 0) {
                norm = 1 / (1 + sqrt(2.0) * k + k * k)
                a0 = (v + sqrt(2 * v) * k + k * k) * norm
                a1 = 2 * (k * k - v) * norm
                a2 = (v - sqrt(2 * v) * k + k * k) * norm
                b1 = 2 * (k * k - 1) * norm
                b2 = (1 - sqrt(2.0) * k + k * k) * norm
            } else {
                norm = 1 / (v + sqrt(2 * v) * k + k * k)
                a0 = (1 + sqrt(2.0) * k + k * k) * norm
                a1 = 2 * (k * k - 1) * norm
                a2 = (1 - sqrt(2.0) * k + k * k) * norm
                b1 = 2 * (k * k - v) * norm
                b2 = (v - sqrt(2 * v) * k + k * k) * norm
            }
        }
    }
}

/** A cascade of biquads applied to every channel of each incoming sample, in order. */
class FilterChain(val filters: List<BiquadMultiChan>) {
    val channels: Int = filters.first().channels

    init {
        require(filters.all { it.channels == channels }) { "all filters must have the same channel count" }
    }

    /** Filters one multi-channel sample in place and returns it. */
    fun process(sample: DoubleArray): DoubleArray {
        for (ch in 0 until channels) {
            var x = sample[ch]
            for (f in filters) x = f.process(x, ch)
            sample[ch] = x
        }
        return sample
    }

    fun reset() = filters.forEach { it.reset() }

    companion object {
        /** The chain in files_for_pi/realtime_pi_classify.py: DC high-pass, 50 Hz notch, 100 Hz low-pass. */
        fun pi(samplingRate: Int = 500, channels: Int = 8) = FilterChain(
            listOf(
                BiquadMultiChan(channels, FilterType.HIGHPASS, 4.5 / samplingRate, 0.5, 0.0),
                BiquadMultiChan(channels, FilterType.NOTCH, 50.0 / samplingRate, 4.0, 0.0),
                BiquadMultiChan(channels, FilterType.LOWPASS, 100.0 / samplingRate, 0.5, 0.0),
            )
        )
    }
}
