package yad.core

import kotlin.math.sqrt

/**
 * How asym/mag take the RMS of an [8, L] window. Models were trained with both versions,
 * and each exported head says which one it needs (input.scalar_rms_over).
 */
enum class RmsOver(val jsonName: String) {
    /** RMS over the 8 channels at each time step (axis=1): what the Pi computes today. */
    CHANNELS("channels"),

    /** RMS over time for each channel (axis=2): what mlp_model_tomer was trained with. */
    TIME("time");

    companion object {
        fun fromJson(name: String) = entries.firstOrNull { it.jsonName == name }
            ?: throw IllegalArgumentException("unknown scalar_rms_over '$name'")
    }
}

/**
 * The two hand-made features appended to the CNN embedding:
 *   asym = mean(rms[4..7]) - mean(rms[0..3]),  mag = mean(rms)
 * Indices 0..3 / 4..7 are applied to whatever the RMS vector is, exactly like the Python
 * code (with CHANNELS that means time steps 0..7, not electrodes).
 */
object ScalarFeatures {
    private val LEFT = 0..3
    private val RIGHT = 4..7

    data class Values(val asym: Float, val mag: Float)

    fun compute(w: EmgWindow, mode: RmsOver): Values {
        val rms = when (mode) {
            RmsOver.CHANNELS -> DoubleArray(w.length) { t ->
                var s = 0.0
                for (ch in 0 until w.channels) { val x = w[ch, t].toDouble(); s += x * x }
                sqrt(s / w.channels)
            }
            RmsOver.TIME -> DoubleArray(w.channels) { ch ->
                var s = 0.0
                for (t in 0 until w.length) { val x = w[ch, t].toDouble(); s += x * x }
                sqrt(s / w.length)
            }
        }
        val left = LEFT.sumOf { rms[it] } / LEFT.count()
        val right = RIGHT.sumOf { rms[it] } / RIGHT.count()
        return Values(asym = (right - left).toFloat(), mag = rms.average().toFloat())
    }

    /** [cnn embedding, asym, mag]: the head's input, in the order the scalers were fit on. */
    fun headInput(cnn: FloatArray, w: EmgWindow, mode: RmsOver): FloatArray {
        val v = compute(w, mode)
        return FloatArray(cnn.size + 2).also {
            cnn.copyInto(it)
            it[cnn.size] = v.asym
            it[cnn.size + 1] = v.mag
        }
    }
}
