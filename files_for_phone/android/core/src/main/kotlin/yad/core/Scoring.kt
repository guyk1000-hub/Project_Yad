package yad.core

import kotlin.math.sqrt

/**
 * Scores decisions on labelled data (replays). Each group of [groupSize] windows is scored
 * against the gesture held for most of the group, not just the last window, and groups that
 * span a gesture change are counted separately: there, "the right answer" is ambiguous.
 *
 * Mirrors GroupAverageDecisionPolicy's grouping: feed every window, in order.
 */
class GroupScorer(val groupSize: Int) {
    data class Totals(
        val groups: Int = 0,
        val decided: Int = 0,
        val correct: Int = 0,
        val transitionGroups: Int = 0,
        val transitionDecided: Int = 0,
    ) {
        /** Decisions in groups where one gesture was held throughout. */
        val steadyDecided: Int get() = decided - transitionDecided
    }

    var totals = Totals()
        private set
    private var steadyCorrect = 0
    private val labels = ArrayList<Int>(groupSize)

    /** Accuracy over groups where one gesture was held for the whole group. */
    val steadyAccuracy: Double
        get() = if (totals.steadyDecided == 0) 0.0 else steadyCorrect.toDouble() / totals.steadyDecided

    /** Call once per window, with the decision the policy returned for it (usually null). */
    fun add(label: Int, decidedClass: Int?) {
        labels.add(label)
        if (labels.size < groupSize) {
            check(decidedClass == null) { "decision before the group was complete" }
            return
        }
        val majority = labels.groupingBy { it }.eachCount().maxBy { it.value }.key
        val transition = labels.any { it != labels[0] }
        val decided = decidedClass != null
        val correct = decided && decidedClass == majority
        if (decided && correct && !transition) steadyCorrect++
        totals = totals.copy(
            groups = totals.groups + 1,
            decided = totals.decided + if (decided) 1 else 0,
            correct = totals.correct + if (correct) 1 else 0,
            transitionGroups = totals.transitionGroups + if (transition) 1 else 0,
            transitionDecided = totals.transitionDecided + if (transition && decided) 1 else 0,
        )
        labels.clear()
    }
}

/**
 * Per-channel RMS of filtered windows, averaged over the last [windows] windows. Used to
 * compare the live armband's signal level with the user's recordings (the units check).
 */
class ChannelLevels(val channels: Int = 8, val windows: Int = 25) {
    private val history = ArrayDeque<DoubleArray>()

    fun add(w: EmgWindow) {
        history.addLast(DoubleArray(channels) { ch ->
            var s = 0.0
            for (t in 0 until w.length) { val x = w[ch, t].toDouble(); s += x * x }
            sqrt(s / w.length)
        })
        if (history.size > windows) history.removeFirst()
    }

    val count: Int get() = history.size

    fun mean(): DoubleArray = DoubleArray(channels) { ch -> history.sumOf { it[ch] } / history.size.coerceAtLeast(1) }

    fun clear() = history.clear()

    companion object {
        /** Mean per-channel RMS over the recording's rest windows (gesture 0): the reference level. */
        fun restReference(replay: ReplayRecording, channels: Int = 8): DoubleArray {
            val levels = ChannelLevels(channels, windows = Int.MAX_VALUE)
            replay.windows.zip(replay.labels.toList()).filter { it.second == Gesture.REST.id }.forEach { levels.add(it.first) }
            return levels.mean()
        }
    }
}
