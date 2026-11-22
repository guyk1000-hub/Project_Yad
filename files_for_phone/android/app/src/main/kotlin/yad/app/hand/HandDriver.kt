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
}

sealed interface HandState {
    data object Disconnected : HandState
    data class Connecting(val step: String) : HandState
    data object Ready : HandState
    data class Failed(val reason: String) : HandState
}
