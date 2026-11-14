package yad.core

/** A gesture the pipeline decided on, with the averaged class probabilities behind it. */
data class Decision(val classIndex: Int, val probabilities: DoubleArray) {
    val confidence: Double get() = probabilities[classIndex]
}

interface DecisionPolicy {
    /** Feed one window's class probabilities; returns a decision when one is made. */
    fun offer(probabilities: DoubleArray): Decision?
    fun reset()
}

/**
 * The Pi's policy (training_data/inference_mix.py, _inference_worker): average consecutive,
 * non-overlapping groups of [groupSize] windows; emit the top class if its averaged
 * probability is >= [threshold]. Defaults are what the Pi runs (its config sets neither).
 */
class GroupAverageDecisionPolicy(
    val groupSize: Int = 6,
    val threshold: Double = 0.7,
) : DecisionPolicy {
    private val pending = ArrayList<DoubleArray>(groupSize)

    override fun offer(probabilities: DoubleArray): Decision? {
        pending.add(probabilities)
        if (pending.size < groupSize) return null
        val avg = DoubleArray(probabilities.size)
        for (p in pending) for (i in avg.indices) avg[i] += p[i]
        for (i in avg.indices) avg[i] /= groupSize
        pending.clear()
        var top = 0
        for (i in avg.indices) if (avg[i] > avg[top]) top = i
        return if (avg[top] >= threshold) Decision(top, avg) else null
    }

    override fun reset() = pending.clear()
}
