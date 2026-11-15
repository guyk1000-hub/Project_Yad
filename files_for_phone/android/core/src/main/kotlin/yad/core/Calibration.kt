package yad.core

/**
 * Turns a calibration recording (raw samples + the gesture held for each) into training windows,
 * and sets how much they count against the user's original recording.
 *
 * Recipe, chosen offline on the 14 Nov guided checks (FINDINGS.md / REPORT_NOTES.md):
 * - windows every [HOP] samples, entirely inside one gesture, starting >= 1 s into it;
 * - the new windows weigh [NEW_SHARE] of the training total, the old recording the rest;
 * - 300 L-BFGS iterations (sklearn 300 vs full training: tomer 99% either way, guy 77-80% vs 90%,
 *   so more data per gesture matters more than more iterations).
 */
object Calibration {
    const val PHASE_SECONDS = 10
    const val HOP = 25
    const val NEW_SHARE = 0.37
    const val ITERATIONS = 300

    /** Filtered windows with labels. The filters run over the whole stream, getting-ready part included. */
    fun windows(raw: List<DoubleArray>, labels: List<Int>, sampleRate: Int = 500, length: Int = 100,
                hop: Int = HOP): List<Pair<EmgWindow, Int>> {
        val filters = FilterChain.pi(sampleRate)
        val filtered = raw.map { filters.process(it.copyOf()) }
        val phaseStart = IntArray(labels.size)
        for (i in 1 until labels.size) phaseStart[i] = if (labels[i] != labels[i - 1]) i else phaseStart[i - 1]
        val out = ArrayList<Pair<EmgWindow, Int>>()
        var s = 0
        while (s + length <= labels.size) {
            val label = labels[s]
            if (label >= 0 && labels[s + length - 1] == label && s >= phaseStart[s] + sampleRate) {
                val data = FloatArray(8 * length)
                for (ch in 0 until 8) for (t in 0 until length) data[ch * length + t] = filtered[s + t][ch].toFloat()
                out.add(EmgWindow(data, 8, length) to label)
            }
            s += hop
        }
        return out
    }

    /** Weight per new window so the new data is [NEW_SHARE] of the total weight. */
    fun newWeight(oldCount: Int, newCount: Int): Double =
        if (oldCount == 0) 1.0 else NEW_SHARE / (1 - NEW_SHARE) * oldCount / newCount
}
