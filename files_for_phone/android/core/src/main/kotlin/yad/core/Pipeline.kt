package yad.core

/** Window -> CNN embedding. On the phone this is the TFLite model; in tests, a fake. */
interface FeatureExtractor {
    val outputSize: Int
    fun extract(window: EmgWindow): FloatArray
}

/**
 * The signal path from raw armband samples to decisions:
 * filter -> window -> CNN + asym/mag -> head -> decision policy.
 * Not thread-safe; feed it from one thread.
 */
class Pipeline(
    private val filters: FilterChain,
    private val windower: Windower,
    private val extractor: FeatureExtractor,
    val head: MlpHead,
    private val policy: DecisionPolicy,
) {
    init {
        require(extractor.outputSize + 2 == head.inputSize) {
            "extractor gives ${extractor.outputSize} features but ${head.modelName} expects ${head.inputSize - 2} + 2"
        }
    }

    /** Result for each completed window. */
    data class WindowResult(val probabilities: DoubleArray, val decision: Decision?)

    /** Feeds one raw sample (8 channels, µV); returns a result when a window completes. */
    fun push(rawSample: DoubleArray): WindowResult? {
        val window = windower.push(filters.process(rawSample.copyOf())) ?: return null
        return classify(window)
    }

    /** Classifies an already-filtered window (also used to replay recordings, which are stored filtered). */
    fun classify(window: EmgWindow): WindowResult {
        val probs = head.predictProba(ScalarFeatures.headInput(extractor.extract(window), window, head.rmsOver))
        return WindowResult(probs, policy.offer(probs))
    }

    fun reset() {
        filters.reset()
        windower.reset()
        policy.reset()
    }
}
