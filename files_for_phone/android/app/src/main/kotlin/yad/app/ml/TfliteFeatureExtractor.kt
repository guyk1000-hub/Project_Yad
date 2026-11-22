package yad.app.ml

import android.content.Context
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import yad.core.EmgWindow
import yad.core.FeatureExtractor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** How TFLite runs the CNN. The benchmark tries several; playback uses [DEFAULT]. */
data class Engine(
    val label: String,
    val threads: Int = 1,
    val xnnpack: Boolean = true,
    val allowFp16: Boolean = false,
    val gpu: Boolean = false,
) {
    companion object {
        /** Fastest on the S23 FE (benchmark 14 Nov: 94.9 ms vs 111.5 ms for 1 thread), same fp32 results. */
        val DEFAULT = Engine("CPU, 4 threads, no XNNPACK", threads = 4, xnnpack = false)

        val BENCHMARK = listOf(
            Engine("CPU, 1 thread", threads = 1),
            Engine("CPU, 2 threads", threads = 2),
            Engine("CPU, 4 threads", threads = 4),
            DEFAULT,
            Engine("CPU, 1 thread, fp16 allowed", threads = 1, allowFp16 = true),
            Engine("CPU, 4 threads, fp16 allowed", threads = 4, allowFp16 = true),
            Engine("GPU, fp32", gpu = true),
            Engine("GPU, fp16 allowed", gpu = true, allowFp16 = true),
        )
    }
}

/**
 * The NaviFlame CNN up to dense_8, converted by files_for_phone/tools/export_cnn.py.
 * Input [1, 8, 100, 1] float32 in the same [channel][sample] order as EmgWindow.data.
 * Not thread-safe: use from one thread.
 */
class TfliteFeatureExtractor(model: MappedByteBuffer, val engine: Engine = Engine.DEFAULT) : FeatureExtractor, AutoCloseable {
    private val delegate: Delegate? = if (engine.gpu) {
        GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(engine.allowFp16))
    } else null

    private val interpreter = Interpreter(model, Interpreter.Options().apply {
        setNumThreads(engine.threads)
        setUseXNNPACK(engine.xnnpack)
        if (engine.allowFp16) setAllowFp16PrecisionForFp32(true)
        delegate?.let { addDelegate(it) }
    })
    private val inputShape = interpreter.getInputTensor(0).shape()
    override val outputSize: Int = interpreter.getOutputTensor(0).shape().last()

    private val input = ByteBuffer.allocateDirect(4 * inputShape.fold(1) { a, b -> a * b }).order(ByteOrder.nativeOrder())
    private val output = arrayOf(FloatArray(outputSize))

    override fun extract(window: EmgWindow): FloatArray {
        require(window.data.size * 4 == input.capacity()) { "window size doesn't match model input ${inputShape.toList()}" }
        input.rewind()
        input.asFloatBuffer().put(window.data)
        interpreter.run(input, output)
        return output[0].copyOf()
    }

    override fun close() {
        interpreter.close()
        (delegate as? AutoCloseable)?.close()
    }

    companion object {
        const val ASSET = "feature_extractor.tflite"

        fun mapAsset(context: Context, name: String = ASSET): MappedByteBuffer =
            context.assets.openFd(name).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}
