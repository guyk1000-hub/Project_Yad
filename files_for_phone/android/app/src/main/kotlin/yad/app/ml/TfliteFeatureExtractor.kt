package yad.app.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import yad.core.EmgWindow
import yad.core.FeatureExtractor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * The NaviFlame CNN up to dense_8, converted by files_for_phone/tools/export_cnn.py.
 * Input [1, 8, 100, 1] float32 in the same [channel][sample] order as EmgWindow.data.
 * Not thread-safe: use from one thread.
 */
class TfliteFeatureExtractor(model: MappedByteBuffer, val threads: Int) : FeatureExtractor, AutoCloseable {
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(threads))
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

    override fun close() = interpreter.close()

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
