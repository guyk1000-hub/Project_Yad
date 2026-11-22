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

/**
 * Fast mode: decides after every window. The newest [windows] windows must all have the same top
 * class, each with probability >= [threshold]; then that class is decided (and keeps being
 * decided while it holds). Compared with groups, it doesn't wait for a group boundary.
 *
 * Switch guard: going straight from one moving gesture to another (not via [restIndex]) needs one
 * window more, so a gesture that only flashes by during a change (e.g. OPEN on the way to CLOSE)
 * doesn't move the hand.
 */
class AgreeDecisionPolicy(
    val windows: Int = 3,
    val threshold: Double = 0.7,
    val switchGuard: Boolean = true,
    val restIndex: Int? = null,
) : DecisionPolicy {
    init {
        require(windows >= 1)
    }

    private val recent = ArrayDeque<DoubleArray>()
    private var lastDecided: Int? = null

    override fun offer(probabilities: DoubleArray): Decision? {
        recent.addLast(probabilities)
        if (recent.size > windows + 1) recent.removeFirst()
        val top = argmax(probabilities)
        var run = 0
        for (p in recent.reversed()) {
            if (argmax(p) == top && p[top] >= threshold) run++ else break
        }
        val last = lastDecided
        val guarded = switchGuard && last != null && last != top && last != restIndex && top != restIndex
        val needed = windows + if (guarded) 1 else 0
        if (run < needed) return null
        val used = recent.toList().takeLast(needed)
        val avg = DoubleArray(probabilities.size)
        for (p in used) for (i in avg.indices) avg[i] += p[i] / needed
        lastDecided = top
        return Decision(top, avg)
    }

    override fun reset() {
        recent.clear()
        lastDecided = null
    }

    private fun argmax(p: DoubleArray): Int {
        var t = 0
        for (i in p.indices) if (p[i] > p[t]) t = i
        return t
    }
}
