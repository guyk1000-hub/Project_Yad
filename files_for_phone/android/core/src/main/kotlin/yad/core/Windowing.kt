package yad.core

/**
 * One window of filtered EMG, laid out like the Pi's model input [8, L, 1]:
 * data[ch * length + t]. float32, because the Pi casts to float32 after filtering.
 */
class EmgWindow(val data: FloatArray, val channels: Int = 8, val length: Int = 100) {
    init {
        require(data.size == channels * length) { "expected ${channels * length} values, got ${data.size}" }
    }

    operator fun get(channel: Int, t: Int): Float = data[channel * length + t]
}

/**
 * Cuts a stream of filtered samples into windows of [length] samples: the first window as
 * soon as [length] samples have arrived, then one every [hop] samples.
 *
 * hop == length reproduces the Pi (consecutive, non-overlapping windows); a smaller hop
 * gives overlapping windows for the lower-latency mode.
 */
class Windower(val channels: Int = 8, val length: Int = 100, val hop: Int = length) {
    init {
        require(hop in 1..length) { "hop must be in 1..$length" }
    }

    private val ring = Array(channels) { DoubleArray(length) }
    private var head = 0        // next write position = oldest sample once the ring is full
    private var filled = 0
    private var emitted = false
    private var sinceEmit = 0

    /** Adds one filtered sample; returns a window when one is due, else null. */
    fun push(sample: DoubleArray): EmgWindow? {
        for (ch in 0 until channels) ring[ch][head] = sample[ch]
        head = (head + 1) % length
        if (filled < length) {
            filled++
            if (filled < length) return null
        } else {
            sinceEmit++
            if (emitted && sinceEmit < hop) return null
        }
        emitted = true
        sinceEmit = 0
        val out = FloatArray(channels * length)
        for (ch in 0 until channels) {
            for (t in 0 until length) out[ch * length + t] = ring[ch][(head + t) % length].toFloat()
        }
        return EmgWindow(out, channels, length)
    }

    fun reset() {
        ring.forEach { it.fill(0.0) }
        head = 0
        filled = 0
        emitted = false
        sinceEmit = 0
    }
}
