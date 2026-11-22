package yad.app.hand

import kotlinx.coroutines.flow.StateFlow
import yad.core.HandSink

/**
 * A real hand the app can connect to. Each hand version has its own driver (its own transport and
 * command encoding); the rest of the app only sees this interface. See Haifa3dBleHand.
 */
interface HandDriver : HandSink {
    /** Shown in the app, e.g. "Haifa3D (v1)". */
    val label: String
    val state: StateFlow<HandState>
    suspend fun connect()
    fun disconnect()

    /** Movement speed, for hands that have one (HandModel.speedRange). */
    suspend fun setSpeed(value: Int) {}

    /** One finger (numbered from 1) to [percent], 0 = open; for hands with HandModel.fingers. */
    suspend fun setFingerPosition(finger: Int, percent: Int): Unit = error("this hand can't move single fingers")
}

sealed interface HandState {
    data object Disconnected : HandState
    data class Connecting(val step: String) : HandState
    data object Ready : HandState
    data class Failed(val reason: String) : HandState
}
