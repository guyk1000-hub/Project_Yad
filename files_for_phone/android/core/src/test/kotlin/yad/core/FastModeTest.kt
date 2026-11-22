package yad.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FastModeTest {
    private fun p(top: Int, conf: Double = 0.9) = DoubleArray(5) { if (it == top) conf else (1 - conf) / 4 }

    @Test
    fun `decides as soon as N windows agree, not at a group boundary`() {
        val policy = AgreeDecisionPolicy(windows = 3, threshold = 0.7, restIndex = 0)
        assertNull(policy.offer(p(1)))
        assertNull(policy.offer(p(1)))
        assertEquals(1, policy.offer(p(1))?.classIndex)      // 3rd window
        assertEquals(1, policy.offer(p(1))?.classIndex)      // and while it holds
        assertNull(policy.offer(p(1, conf = 0.5)))           // below the threshold
    }

    @Test
    fun `a gesture flashing by during a change needs one window more`() {
        val policy = AgreeDecisionPolicy(windows = 2, threshold = 0.7, switchGuard = true, restIndex = 0)
        policy.offer(p(1)); assertNotNull(policy.offer(p(1)))  // CLOSE decided
        assertNull(policy.offer(p(2)))
        assertNull(policy.offer(p(2)))                         // CLOSE -> OPEN directly: 2 isn't enough
        assertEquals(2, policy.offer(p(2))?.classIndex)        // 3 is
        policy.offer(p(0)); assertEquals(0, policy.offer(p(0))?.classIndex)   // to REST: no guard
        policy.offer(p(1)); assertEquals(1, policy.offer(p(1))?.classIndex)   // from REST: no guard
    }

    @Test
    fun `the guided check reports the reaction time per gesture`() {
        val check = GuidedCheck("t", "time", intArrayOf(0, 1, 2, 3, 4))
        val sr = check.sampleRate
        var phase = 0
        var end = check.phases[0].seconds * sr
        for (n in 0 until check.phases.sumOf { it.seconds } * sr) {
            if (n == end) { phase++; end += check.phases[phase].seconds * sr }
            check.addSample(DoubleArray(8), phase)
            val sinceStart = n - (end - check.phases[phase].seconds * sr)
            // Decide the held gesture 0.8 s after each cue.
            if (sinceStart >= 400) check.addDecision(phase, check.phases[phase].label)
        }
        val r = check.reactionMs()
        assertEquals(5, r.size)
        assertTrue(r.all { it.second == 800 }, r.toString())
        assertTrue(check.reactionSummary().single().contains("median 800 ms"), check.reactionSummary().toString())
    }
}
