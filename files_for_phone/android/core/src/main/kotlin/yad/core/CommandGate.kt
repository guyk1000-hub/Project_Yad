package yad.core

/**
 * Decides what actually reaches the hand.
 *
 * - Classifier decisions pass only while not stopped. Like the Pi, every decision is sent,
 *   even if it repeats the previous one, unless [sendOnlyOnChange] is set.
 * - Manual commands always pass, even when stopped: the user is in control.
 *
 * Stop blocks the classifier immediately. It doesn't move the hand, and it doesn't disconnect.
 */
class CommandGate(@Volatile var sendOnlyOnChange: Boolean = false) {
    /** Volatile: Stop is pressed on the UI thread and must be seen at once by the classifier thread. */
    @Volatile
    var stopped: Boolean = false
        private set

    private var lastClassifierCommand: HandCommand? = null

    fun stop() {
        stopped = true
    }

    fun resume() {
        stopped = false
        lastClassifierCommand = null
    }

    /** Returns the command to send for a classifier decision, or null to send nothing. */
    fun fromClassifier(gesture: Gesture): HandCommand? {
        if (stopped) return null
        val cmd = gesture.command ?: return null
        if (sendOnlyOnChange && cmd == lastClassifierCommand) return null
        lastClassifierCommand = cmd
        return cmd
    }

    /** Manual buttons bypass the classifier and the stop. */
    fun manual(command: HandCommand): HandCommand = command
}
