package yad.core

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A short guided session: hold each gesture for [PHASE_SECONDS] while the live pipeline runs.
 * Records the raw (unfiltered) samples with the gesture being held, plus what the phone decided
 * for each window, so the same data can be run through the Pi's Python code on a PC
 * (files_for_phone/tools/analyze_check.py). Pure Kotlin, so the file format is tested on the PC.
 */
class GuidedCheck(val model: String, val rmsOver: String, val classes: IntArray, val sampleRate: Int = 500) {
    data class Phase(val label: Int, val title: String, val seconds: Int)

    val phases: List<Phase> = listOf(Phase(-1, "Get ready: relax your arm", READY_SECONDS)) +
        GESTURES.map { Phase(it.id, it.name, PHASE_SECONDS) }

    private val labels = ByteArrayOutputStream()
    private val samples = ByteArrayOutputStream()
    private val windows = ByteArrayOutputStream()
    var sampleCount = 0
        private set

    /** Ignore the first second of each gesture (moving into it). */
    val settleSamples = sampleRate

    /** Index of the first sample of each phase. */
    private val phaseStart = IntArray(phases.size) { -1 }

    // Per gesture id: windows scored / correct, and what the wrong answers were.
    private val scored = HashMap<Int, Int>()
    private val correct = HashMap<Int, Int>()
    private val wrong = HashMap<Int, HashMap<Int, Int>>()

    fun addSample(uv: DoubleArray, phaseIndex: Int) {
        if (phaseStart[phaseIndex] < 0) phaseStart[phaseIndex] = sampleCount
        labels.write(phases[phaseIndex].label)
        val b = ByteBuffer.allocate(8 * 8).order(ByteOrder.LITTLE_ENDIAN)
        uv.forEach { b.putDouble(it) }
        samples.write(b.array())
        sampleCount++
    }

    /**
     * One completed window, ending at the last sample added. Scored only if the whole window lies
     * inside one gesture phase and starts at least [settleSamples] (1 s) after the phase began.
     */
    fun addWindow(phaseIndex: Int, predictedClass: Int, probs: DoubleArray, windowLength: Int) {
        val end = sampleCount - 1
        val label = phases[phaseIndex].label
        val b = ByteBuffer.allocate(4 + 1 + 1 + 8 * probs.size).order(ByteOrder.LITTLE_ENDIAN)
        b.putInt(end); b.put(label.toByte()); b.put(predictedClass.toByte())
        probs.forEach { b.putDouble(it) }
        windows.write(b.array())
        val start = end - windowLength + 1
        if (label < 0 || start < phaseStart[phaseIndex] + settleSamples) return
        scored[label] = (scored[label] ?: 0) + 1
        if (predictedClass == label) correct[label] = (correct[label] ?: 0) + 1
        else wrong.getOrPut(label) { HashMap() }.merge(predictedClass, 1, Int::plus)
    }

    /** One line per gesture: accuracy and the most common wrong answer. */
    fun results(): List<String> = GESTURES.map { g ->
        val n = scored[g.id] ?: 0
        val ok = correct[g.id] ?: 0
        val confusion = wrong[g.id]?.maxByOrNull { it.value }?.let { (cls, k) ->
            "  (else mostly ${Gesture.fromId(cls)?.name ?: cls} ×$k)"
        } ?: ""
        if (n == 0) "${g.name.padEnd(6)} no windows" else "%-6s %3d%%  (%d/%d)%s".format(g.name, 100 * ok / n, ok, n, confusion)
    }

    /** File: b"YADC", int32 version, int32 header length, header JSON, then samples and windows. */
    fun toBytes(): ByteArray {
        val header = """{"model":"$model","scalar_rms_over":"$rmsOver","sampling_rate":$sampleRate,""" +
            """"window_length":100,"phase_seconds":$PHASE_SECONDS,"ready_seconds":$READY_SECONDS,""" +
            """"settle_samples":$settleSamples,"classes":[${classes.joinToString(",")}]}"""
        val h = header.toByteArray(Charsets.UTF_8)
        val out = ByteArrayOutputStream()
        val head = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
        head.put("YADC".toByteArray(Charsets.US_ASCII)); head.putInt(1); head.putInt(h.size)
        out.write(head.array()); out.write(h)
        val n = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
        out.write(n.putInt(sampleCount).array())
        out.write(labels.toByteArray())
        out.write(samples.toByteArray())
        val w = windows.toByteArray()
        val perWindow = 4 + 1 + 1 + 8 * classes.size
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(w.size / perWindow).array())
        out.write(w)
        return out.toByteArray()
    }

    companion object {
        val GESTURES = listOf(Gesture.REST, Gesture.CLOSE, Gesture.OPEN, Gesture.RIGHT, Gesture.LEFT)
        const val PHASE_SECONDS = 6
        const val READY_SECONDS = 3
    }
}
