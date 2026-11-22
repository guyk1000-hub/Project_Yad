package yad.app.hand

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import yad.core.Haifa3dProtocol
import yad.core.HandCommand
import java.util.UUID

/**
 * The real Haifa3D hand over BLE, as files_for_pi/basic_commands_pi.py drives it: connect to the
 * hand's address (no pairing), write each command to the "direct execute" characteristic.
 *
 * The caller must hold BLUETOOTH_CONNECT (and BLUETOOTH_SCAN for the scan fallback).
 */
@SuppressLint("MissingPermission")
class Haifa3dBleHand(private val context: Context, val address: String = Haifa3dProtocol.DEFAULT_ADDRESS) : HandDriver {
    override val label = "Haifa3D (v1)"

    private val _state = MutableStateFlow<HandState>(HandState.Disconnected)
    override val state: StateFlow<HandState> = _state

    private val adapter = context.getSystemService(BluetoothManager::class.java)?.adapter
    private val writeLock = Mutex()
    private var gatt: BluetoothGatt? = null
    private var characteristic: BluetoothGattCharacteristic? = null
    private var pendingConnect: CompletableDeferred<Boolean>? = null
    private var pendingWrite: CompletableDeferred<Int>? = null
    private var pendingRead: CompletableDeferred<ByteArray?>? = null

    /** Battery level in %, read when connected and on request (null = unknown). */
    private val _battery = MutableStateFlow<Int?>(null)
    val battery: StateFlow<Int?> = _battery

    private val callback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                _state.value = HandState.Connecting("discovering services")
                g.discoverServices()
            } else {
                characteristic = null
                pendingConnect?.complete(false)
                pendingWrite?.complete(-1)
                pendingRead?.complete(null)
                _battery.value = null
                if (_state.value == HandState.Ready) _state.value = HandState.Failed("connection lost (status $status)")
            }
        }

        override fun onServicesDiscovered(g: BluetoothGatt, status: Int) {
            val uuid = UUID.fromString(Haifa3dProtocol.DIRECT_EXECUTE_UUID)
            characteristic = g.services.firstNotNullOfOrNull { s -> s.getCharacteristic(uuid) }
            pendingConnect?.complete(characteristic != null)
        }

        override fun onCharacteristicWrite(g: BluetoothGatt, c: BluetoothGattCharacteristic, status: Int) {
            pendingWrite?.complete(status)
        }

        override fun onCharacteristicRead(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray, status: Int) {
            pendingRead?.complete(if (status == BluetoothGatt.GATT_SUCCESS) value else null)
        }
    }

    override suspend fun connect() {
        val bt = adapter
        if (bt == null || !bt.isEnabled) {
            _state.value = HandState.Failed("Bluetooth is off")
            return
        }
        disconnect()
        _state.value = HandState.Connecting("connecting to $address")
        if (tryConnect(bt.getRemoteDevice(address))) return
        // Some phones only connect to a device they've seen in a scan.
        _state.value = HandState.Connecting("scanning for $address")
        val found = scanFor(address) ?: run {
            _state.value = HandState.Failed("hand not found — is it on and nearby?")
            return
        }
        _state.value = HandState.Connecting("connecting to $address")
        if (!tryConnect(found)) _state.value = HandState.Failed("couldn't connect or find the command characteristic")
    }

    private suspend fun tryConnect(device: BluetoothDevice): Boolean {
        val done = CompletableDeferred<Boolean>()
        pendingConnect = done
        gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE)
        val ok = withTimeoutOrNull(CONNECT_TIMEOUT_MS) { done.await() } == true
        if (ok) {
            _state.value = HandState.Ready
            runCatching { readBattery() }
        } else {
            gatt?.close()
            gatt = null
        }
        return ok
    }

    private suspend fun scanFor(address: String): BluetoothDevice? {
        val scanner = adapter?.bluetoothLeScanner ?: return null
        val result = CompletableDeferred<BluetoothDevice>()
        val cb = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, r: ScanResult) {
                result.complete(r.device)
            }
        }
        scanner.startScan(
            listOf(ScanFilter.Builder().setDeviceAddress(address).build()),
            ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(),
            cb,
        )
        return try {
            withTimeoutOrNull(SCAN_TIMEOUT_MS) { result.await() }
        } finally {
            scanner.stopScan(cb)
        }
    }

    /** Sends one command; throws if the hand isn't connected or the write fails. */
    override suspend fun send(command: HandCommand) =
        write(Haifa3dProtocol.DIRECT_EXECUTE_UUID, checkNotNull(Haifa3dProtocol.encode(command)) { "the old hand can't do $command" })

    /** Runs an action now (Haifa3dProtocol.action / liveStep bytes). */
    suspend fun execute(action: ByteArray) = write(Haifa3dProtocol.DIRECT_EXECUTE_UUID, action)

    /** Runs preset [n] (0..11) stored on the hand. */
    suspend fun triggerPreset(n: Int) = write(Haifa3dProtocol.TRIGGER_UUID, byteArrayOf(n.toByte()))

    suspend fun readPreset(n: Int): ByteArray? = read(Haifa3dProtocol.presetUuid(n))

    suspend fun writePreset(n: Int, action: ByteArray) = write(Haifa3dProtocol.presetUuid(n), action)

    suspend fun readConfig(id: Int): Int? = read(Haifa3dProtocol.configUuid(id))?.firstOrNull()?.toInt()?.and(0xFF)

    suspend fun writeConfig(id: Int, value: Int) = write(Haifa3dProtocol.configUuid(id), byteArrayOf(value.toByte()))

    /** [which] 0 = reset the presets, 1 = reset the configuration, to factory defaults. */
    suspend fun reset(which: Int) = write(Haifa3dProtocol.configResetUuid(which), byteArrayOf(1))

    suspend fun readBattery(): Int? =
        read(Haifa3dProtocol.BATTERY_LEVEL_UUID)?.firstOrNull()?.toInt()?.and(0xFF).also { _battery.value = it }

    private fun find(uuid: String): BluetoothGattCharacteristic {
        val g = gatt
        check(_state.value == HandState.Ready && g != null) { "hand not connected" }
        val u = UUID.fromString(uuid)
        return checkNotNull(g.services.firstNotNullOfOrNull { it.getCharacteristic(u) }) { "this hand doesn't offer $uuid" }
    }

    private suspend fun write(uuid: String, bytes: ByteArray) = writeLock.withLock {
        val c = find(uuid)
        val g = gatt!!
        val withResponse = c.properties and BluetoothGattCharacteristic.PROPERTY_WRITE != 0
        val type = if (withResponse) BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT else BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE
        val done = CompletableDeferred<Int>()
        pendingWrite = done
        val rc = g.writeCharacteristic(c, bytes, type)
        check(rc == 0) { "write rejected (code $rc)" }   // BluetoothStatusCodes.SUCCESS
        val status = withTimeoutOrNull(WRITE_TIMEOUT_MS) { done.await() }
        check(status == null && !withResponse || status == BluetoothGatt.GATT_SUCCESS) { "write failed (status $status)" }
    }

    private suspend fun read(uuid: String): ByteArray? = writeLock.withLock {
        val c = find(uuid)
        val done = CompletableDeferred<ByteArray?>()
        pendingRead = done
        check(gatt!!.readCharacteristic(c)) { "read rejected" }
        withTimeoutOrNull(WRITE_TIMEOUT_MS) { done.await() }
    }

    override fun disconnect() {
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        characteristic = null
        _state.value = HandState.Disconnected
    }

    private companion object {
        const val CONNECT_TIMEOUT_MS = 10_000L
        const val SCAN_TIMEOUT_MS = 10_000L
        const val WRITE_TIMEOUT_MS = 2_000L
    }
}
