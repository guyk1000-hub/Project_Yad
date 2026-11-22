package yad.core

import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DecisionAndHandTest {

    @Test
    fun `decisions match the Pi's group-average policy`() {
        val g = Golden.json("decision.json")
        val groupSize = g["batch_size"].jsonPrimitive.int
        val threshold = g["threshold"].jsonPrimitive.double
        for (case in g["cases"].jsonArray) {
            val name = case["name"].jsonPrimitive.content
            val caseThreshold = case.jsonObject["threshold"]?.jsonPrimitive?.double ?: threshold
            val policy = GroupAverageDecisionPolicy(groupSize, caseThreshold)
            val got = case["per_window_probs"].matrix().withIndex().mapNotNull { (i, p) ->
                policy.offer(p)?.let { i to it }
            }
            val want = case["decisions"].jsonArray
            assertEquals(want.size, got.size, "$name: number of decisions")
            for ((k, w) in want.withIndex()) {
                assertEquals(w["after_window"].jsonPrimitive.int, got[k].first, "$name decision $k: window")
                assertEquals(w["gesture"].jsonPrimitive.int, got[k].second.classIndex, "$name decision $k: gesture")
                assertTrue(maxAbsDiff(w["avg"].doubles(), got[k].second.probabilities) < 1e-12, "$name decision $k: avg")
            }
            println("decision $name: ${got.size} decisions match")
        }
    }

    @Test
    fun `hand commands match protocol py`() {
        val g = Golden.json("hand_commands.json")
        assertEquals(g["characteristic_uuid"].jsonPrimitive.content, Haifa3dProtocol.DIRECT_EXECUTE_UUID)
        assertEquals(g["hand_address"].jsonPrimitive.content, Haifa3dProtocol.DEFAULT_ADDRESS)
        for ((id, spec) in g["gestures"].jsonObject) {
            val gesture = Gesture.fromId(id.toInt())!!
            assertEquals(spec["name"].jsonPrimitive.content.uppercase(), gesture.name)
            val payload = spec["payload"]
            if (payload.isNull()) {
                assertNull(gesture.command, "$gesture sends nothing")
            } else {
                val bytes = Haifa3dProtocol.encode(gesture.command!!)!!.map { it.toInt() and 0xFF }
                assertEquals(payload.ints().toList(), bytes, "$gesture payload")
            }
        }
    }
}
