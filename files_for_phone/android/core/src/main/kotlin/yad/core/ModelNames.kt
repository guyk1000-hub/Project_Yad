package yad.core

/**
 * Model names. Bundled PC models are plain user names ("tomer", "tomer_long"). Versions made on the
 * phone carry their user, an optional label the user chose, and when they were made:
 * - "<user>@<yyyyMMdd_HHmm[ss]>": calibrated (PC recording + a short phone recording, 0.5.x);
 * - "<user>-<label>@<yyyyMMdd_HHmmss>": trained from zero on phone recordings (label "phone" if none given).
 * The time keeps names unique, so a chosen label never replaces another model.
 */
object ModelNames {
    const val DEFAULT_LABEL = "phone"

    fun userOf(name: String): String = name.substringBefore('@').substringBefore('-')

    /** The user's label: letters, digits, '_' (spaces become '_'), at most 24 characters. */
    fun cleanLabel(label: String): String =
        label.trim().replace(Regex("\\s+"), "_").filter { it.isLetterOrDigit() || it == '_' }.take(24)
            .ifEmpty { DEFAULT_LABEL }

    fun phoneTrained(user: String, label: String, time: String) = "$user-${cleanLabel(label)}@$time"

    fun isPhoneTrained(name: String) = '@' in name && '-' in name.substringBefore('@')

    /** "tomer-morning@20261115_184045" -> "tomer · morning · 15/11 18:40"; bundled names unchanged. */
    fun display(name: String): String {
        val at = name.indexOf('@')
        val t = name.substring(at + 1)
        if (at < 0 || t.length < 13) return name
        val prefix = name.substring(0, at)
        val label = if ('-' in prefix) " · ${prefix.substringAfter('-')}" else ""
        return "${userOf(name)}$label · ${t.substring(6, 8)}/${t.substring(4, 6)} ${t.substring(9, 11)}:${t.substring(11, 13)}"
    }
}
