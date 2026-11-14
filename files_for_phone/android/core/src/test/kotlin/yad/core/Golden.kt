package yad.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/** Access to files_for_phone/tools/golden and files_for_phone/models (paths set by Gradle). */
object Golden {
    private fun dir(prop: String) = File(
        System.getProperty(prop) ?: error("system property $prop not set — run the tests through Gradle")
    )

    /** Typed as JsonElement (not JsonObject) so the non-null `get` extensions below apply. */
    fun json(name: String): JsonElement {
        val f = dir("yad.golden.dir").resolve(name)
        check(f.isFile) { "missing $f — run files_for_phone/tools/make_golden.py" }
        return Json.parseToJsonElement(f.readText()).also { it.jsonObject }
    }

    fun modelText(name: String): String {
        val f = dir("yad.models.dir").resolve(name)
        check(f.isFile) { "missing $f — run files_for_phone/tools/export_head.py" }
        return f.readText()
    }

    fun modelFiles(prefix: String): List<File> =
        dir("yad.models.dir").listFiles { f -> f.name.startsWith(prefix) && f.name.endsWith(".json") }!!.sorted()
}

fun JsonElement.doubles(): DoubleArray = jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()
fun JsonElement.floats(): FloatArray = jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray()
fun JsonElement.matrix(): List<DoubleArray> = jsonArray.map { it.doubles() }
fun JsonElement.ints(): IntArray = jsonArray.map { it.jsonPrimitive.int }.toIntArray()
fun JsonElement.isNull() = this is JsonNull
operator fun JsonElement.get(key: String): JsonElement = jsonObject.getValue(key)
operator fun JsonElement.get(i: Int): JsonElement = (this as JsonArray)[i]

/** Golden window [8][100] -> EmgWindow. */
fun JsonElement.window(): EmgWindow {
    val rows = jsonArray
    val ch = rows.size
    val len = rows[0].jsonArray.size
    val data = FloatArray(ch * len)
    rows.forEachIndexed { c, row -> row.jsonArray.forEachIndexed { t, v -> data[c * len + t] = v.jsonPrimitive.double.toFloat() } }
    return EmgWindow(data, ch, len)
}

fun maxAbsDiff(a: DoubleArray, b: DoubleArray): Double {
    require(a.size == b.size) { "size ${a.size} != ${b.size}" }
    var m = 0.0
    for (i in a.indices) m = max(m, abs(a[i] - b[i]))
    return m
}

/** First index of the maximum, like numpy.argmax. */
fun DoubleArray.argmax(): Int {
    var best = 0
    for (i in indices) if (this[i] > this[best]) best = i
    return best
}
