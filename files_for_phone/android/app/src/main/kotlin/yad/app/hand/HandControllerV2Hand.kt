package yad.app.hand

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothGattServer
import android.bluetooth.BluetoothGattServerCallback
import android.bluetooth.BluetoothGattService
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.content.Context
import android.os.ParcelUuid
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import yad.core.HandCommand
import yad.core.HandControllerV2Protocol
import java.util.UUID

/**
 * The new hand (Haifa3D Hand-Controller-System v2), used as it is. Here the hand is the BLE
 * central, so the phone plays the hand's "manager" app: it advertises the manager service and
 * waits. The hand finds it while it starts up (power the hand on after "waiting for the hand"),
 * connects and subscribes; then each command is one notification (HandControllerV2Protocol).
 *
 * The caller must hold BLUETOOTH_CONNECT and BLUETOOTH_ADVERTISE.
 */
@SuppressLint("MissingPermission")
class HandControllerV2Hand(private val context: Context) : HandDriver {
    override val label = "New hand (controller v2)"

    private val _state = MutableStateFlow<HandState>(HandState.Disconnected)
    override val state: StateFlow<HandState> = _state

    private val manager = context.getSystemService(BluetoothManager::class.java)
    private val serviceUuid = UUID.fromString(HandControllerV2Protocol.SERVICE_UUID)
    private val sendLock = Mutex()
    private var server: BluetoothGattServer? = null
    private var characteristic: BluetoothGattCharacteristic? = null
    private var serviceAdded: CompletableDeferred<Boolean>? = null
    private var pendingNotify: CompletableDeferred<Int>? = null
    private var advertising = false

    /** The hand, once it has subscribed to our commands. */
    @Volatile
    private var hand: BluetoothDevice? = null

    private val advertiseCallback = object : AdvertiseCallback() {
        override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
            if (hand == null) _state.value = HandState.Connecting(WAITING)
        }

        override fun onStartFailure(errorCode: Int) {
            advertising = false
            _state.value = HandState.Failed("couldn't advertise to the hand (error $errorCode)")
        }
    }

    private val callback = object : BluetoothGattServerCallback() {
        override fun onServiceAdded(status: Int, service: BluetoothGattService) {
            serviceAdded?.complete(status == BluetoothGatt.GATT_SUCCESS)
        }

        override fun onConnectionStateChange(device: BluetoothDevice, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                if (hand == null) _state.value = HandState.Connecting("hand connected, waiting for it to subscribe")
            } else if (device == hand) {
                hand = null
                pendingNotify?.complete(-1)
                _state.value = HandState.Connecting("the hand disconnected; $WAITING")
                startAdvertising()
            }
        }

        override fun onDescriptorWriteRequest(
            device: BluetoothDevice, requestId: Int, descriptor: BluetoothGattDescriptor,
            preparedWrite: Boolean, responseNeeded: Boolean, offset: Int, value: ByteArray,
        ) {
            if (descriptor.uuid == CCCD) {
                val on = value.contentEquals(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE) ||
                    value.contentEquals(BluetoothGattDescriptor.ENABLE_INDICATION_VALUE)
                if (on) {
                    hand = device
                    stopAdvertising()
                    _state.value = HandState.Ready
                } else if (device == hand) {
                    hand = null
                    _state.value = HandState.Connecting(WAITING)
                    startAdvertising()
                }
            }
            if (responseNeeded) server?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, offset, value)
        }

        override fun onDescriptorReadRequest(device: BluetoothDevice, requestId: Int, offset: Int, descriptor: BluetoothGattDescriptor) {
            val v = if (device == hand) BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE else BluetoothGattDescriptor.DISABLE_NOTIFICATION_VALUE
            server?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, 0, v)
        }

        override fun onCharacteristicReadRequest(device: BluetoothDevice, requestId: Int, offset: Int, c: BluetoothGattCharacteristic) {
            server?.sendResponse(device, requestId, BluetoothGatt.GATT_SUCCESS, 0, ByteArray(3))
        }

        override fun onNotificationSent(device: BluetoothDevice, status: Int) {
            pendingNotify?.complete(status)
        }
    }

    override suspend fun connect() {
        val bt = manager?.adapter
        if (bt == null || !bt.isEnabled) {
            _state.value = HandState.Failed("Bluetooth is off")
            return
        }
        disconnect()
        _state.value = HandState.Connecting("starting")
        val s = manager.openGattServer(context, callback) ?: run {
            _state.value = HandState.Failed("couldn't open a Bluetooth server on this phone")
            return
        }
        server = s
        val c = BluetoothGattCharacteristic(
            UUID.fromString(HandControllerV2Protocol.CHARACTERISTIC_UUID),
            BluetoothGattCharacteristic.PROPERTY_NOTIFY or BluetoothGattCharacteristic.PROPERTY_READ,
            BluetoothGattCharacteristic.PERMISSION_READ,
        )
        c.addDescriptor(BluetoothGattDescriptor(CCCD, BluetoothGattDescriptor.PERMISSION_READ or BluetoothGattDescriptor.PERMISSION_WRITE))
        characteristic = c
        val service = BluetoothGattService(serviceUuid, BluetoothGattService.SERVICE_TYPE_PRIMARY).apply { addCharacteristic(c) }
        val added = CompletableDeferred<Boolean>()
        serviceAdded = added
        if (!s.addService(service) || withTimeoutOrNull(5_000) { added.await() } != true) {
            _state.value = HandState.Failed("couldn't set up the manager service")
            return
        }
        startAdvertising()
    }

    private fun startAdvertising() {
        val advertiser = manager?.adapter?.bluetoothLeAdvertiser ?: return
        if (advertising || server == null) return
        advertising = true
        advertiser.startAdvertising(
            AdvertiseSettings.Builder()
                .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
                .setConnectable(true)
                .setTimeout(0)
                .build(),
            AdvertiseData.Builder().addServiceUuid(ParcelUuid(serviceUuid)).setIncludeDeviceName(false).build(),
            advertiseCallback,
        )
    }

    private fun stopAdvertising() {
        if (!advertising) return
        advertising = false
        manager?.adapter?.bluetoothLeAdvertiser?.stopAdvertising(advertiseCallback)
    }

    override suspend fun send(command: HandCommand) =
        notify(checkNotNull(HandControllerV2Protocol.encode(command)) { "the new hand can't do $command" })

    override suspend fun setSpeed(value: Int) = notify(HandControllerV2Protocol.speed(value))

    override suspend fun setFingerPosition(finger: Int, percent: Int) = notify(HandControllerV2Protocol.position(finger, percent))

    /** One notification to the hand; throws if it isn't connected or the send fails. */
    private suspend fun notify(bytes: ByteArray) = sendLock.withLock {
        val s = server
        val c = characteristic
        val d = hand
        check(_state.value == HandState.Ready && s != null && c != null && d != null) { "hand not connected" }
        val done = CompletableDeferred<Int>()
        pendingNotify = done
        val rc = s.notifyCharacteristicChanged(d, c, false, bytes)
        check(rc == 0) { "send rejected (code $rc)" }   // BluetoothStatusCodes.SUCCESS
        val status = withTimeoutOrNull(SEND_TIMEOUT_MS) { done.await() }
        check(status == BluetoothGatt.GATT_SUCCESS) { "send failed (status $status)" }
    }

    override fun disconnect() {
        stopAdvertising()
        hand?.let { server?.cancelConnection(it) }
        hand = null
        server?.close()
        server = null
        characteristic = null
        _state.value = HandState.Disconnected
    }

    private companion object {
        val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
        const val WAITING = "waiting for the hand: turn it on (or restart it) now"
        const val SEND_TIMEOUT_MS = 2_000L
    }
}
