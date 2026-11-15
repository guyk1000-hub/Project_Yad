package yad.core

import java.io.File
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScoringTest {
    @Test
    fun `groups are scored against the majority gesture`() {
        val s = GroupScorer(groupSize = 3)
        // steady group, decided correctly
        s.add(1, null); s.add(1, null); s.add(1, 1)
        // transition 2,2,3: majority 2; decided 2 -> correct, but counted as a transition
        s.add(2, null); s.add(2, null); s.add(3, 2)
        // steady group, wrong decision
        s.add(4, null); s.add(4, null); s.add(4, 0)
        // steady group, no decision (below threshold)
        s.add(0, null); s.add(0, null); s.add(0, null)
        val t = s.totals
        assertEquals(4, t.groups)
        assertEquals(3, t.decided)
        assertEquals(2, t.correct)
        assertEquals(1, t.transitionGroups)
        assertEquals(2, t.steadyDecided)
        assertEquals(0.5, s.steadyAccuracy)
    }

    /**
     * The phone's tomer replay with its true labels. The old "last window" score showed ~96% on
     * the phone; every miss was a group that spans a gesture change.
     */
    @Test
    fun `tomer replay decisions are all correct where one gesture was held`() {
        val case = Golden.json("decision.json")["cases"].jsonArray
            .first { it["name"].jsonPrimitive.content == "replay_tomer_labelled" }
        val probs = case["per_window_probs"].matrix()
        val labels = case["labels"].ints()
        val policy = GroupAverageDecisionPolicy()
        val scorer = GroupScorer(policy.groupSize)
        for ((i, p) in probs.withIndex()) scorer.add(labels[i], policy.offer(p)?.classIndex)
        val t = scorer.totals
        println("replay_tomer: $t, steady accuracy ${scorer.steadyAccuracy}")
        assertTrue(t.steadyDecided > 50)
        assertEquals(1.0, scorer.steadyAccuracy)
    }

    @Test
    fun `rest reference comes from the rest windows only`() {
        val r = ReplayRecording.parse(File(System.getProperty("yad.models.dir")).resolve("replay_tomer.bin").readBytes())
        val ref = ChannelLevels.restReference(r)
        assertEquals(8, ref.size)
        assertTrue(ref.all { it > 0 && it.isFinite() }, ref.toList().toString())
        println("tomer rest RMS per channel (µV): ${ref.map { "%.1f".format(it) }}")
    }
}
