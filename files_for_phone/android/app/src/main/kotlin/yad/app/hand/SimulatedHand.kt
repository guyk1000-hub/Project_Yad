package yad.app.hand

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import yad.core.HandCommand
import yad.core.HandSink

/** Stands in for the Haifa3D hand: accepts every command and remembers the hand's last state. */
class SimulatedHand : HandSink {
    private val _position = MutableStateFlow<HandCommand?>(null)
    val position: StateFlow<HandCommand?> = _position

    override suspend fun send(command: HandCommand) {
        _position.value = command
    }
}
