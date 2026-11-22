package yad.core

import kotlin.math.roundToInt

/**
 * Wrist rotation proportional to effort (old hand): while LEFT or RIGHT is decided, the wrist turns
 * one step per decision, and each step is longer the harder the user contracts.
 *
 * Effort is the window's signal magnitude (the "mag" head input). It's compared with [reference],
 * the user's typical magnitude for LEFT/RIGHT in their recordings; without one, the recent median of
 * LEFT/RIGHT windows seen live is used. A step fills at most the time until the next decision, so
 * the wrist turns continuously at full effort and moves slowly at light effort.
 */
class ProportionalWrist(private val reference: Double?) {
    private val seen = ArrayDeque<Double>()

    /** Effort relative to the user's typical LEFT/RIGHT (1.0 = typical); also learns the live median. */
    fun ratio(effort: Double): Double {
        val ref = reference ?: median().takeIf { seen.size >= MIN_SEEN }
        seen.addLast(effort)
        if (seen.size > KEEP) seen.removeFirst()
        if (ref == null || ref <= 0) return 1.0
        return effort / ref
    }

    /**
     * Step length in 50 ms time units for one decision every [decisionMs]: typical effort at
     * sensitivity 1 → half of the time; 1.5× typical → all of it; never below one unit.
     */
    fun timeUnits(ratio: Double, decisionMs: Int, sensitivity: Double): Int {
        val maxUnits = (decisionMs / Haifa3dProtocol.MS_PER_TIME_UNIT).coerceIn(1, 255)
        val share = (ratio * sensitivity - 0.5).coerceIn(MIN_SHARE, 1.0)
        return (share * maxUnits).roundToInt().coerceIn(1, maxUnits)
    }

    private fun median(): Double? {
        if (seen.isEmpty()) return null
        val s = seen.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }

    companion object {
        const val MIN_SHARE = 0.15
        private const val KEEP = 100
        private const val MIN_SEEN = 5

        /** The user's typical LEFT/RIGHT magnitude from head inputs (the last value of each). */
        fun reference(sets: List<TrainingSet>): Double? {
            val mags = sets.flatMap { set ->
                set.labels.indices.filter { set.labels[it] == Gesture.LEFT.id || set.labels[it] == Gesture.RIGHT.id }
                    .map { set.features[it].last().toDouble() }
            }.sorted()
            if (mags.isEmpty()) return null
            return mags[mags.size / 2]
        }

        /** One wrist step for the old hand: low torque, so it stops gently at the end of travel. */
        fun step(left: Boolean, units: Int): ByteArray = Haifa3dProtocol.action(listOf(
            Haifa3dProtocol.Movement(setOf(Haifa3dProtocol.Motor.TURN), if (left) setOf(Haifa3dProtocol.Motor.TURN) else emptySet(), emptySet(), units),
        ))
    }
}
