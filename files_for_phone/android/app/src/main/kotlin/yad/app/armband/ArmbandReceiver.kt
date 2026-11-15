package yad.app.armband

import android.content.Context
import android.net.ConnectivityManager
import android.net.LinkAddress
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import mylibrary.mindrove.SensorData
import yad.core.MindRoveWua2
import kotlin.math.roundToInt
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.Inet4Address
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.SocketException
import java.net.SocketTimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Receives EMG from the MindRove armband over its WiFi network.
 *
 * The MindRove SDK's ServerManager listens on UDP 4210 on *every* interface, mobile data
 * included. Fake packets there could move the hand, so this class owns the socket and uses
 * the SDK only to decode packets (SensorData.fromUdpPacket: signed 24-bit x 0.045 µV):
 *
 * - Default: bound to the phone's own address on the WiFi network only.
 * - acceptBroadcast (explicit user opt-in, for armbands that broadcast): bound to all addresses
 *   on port 4210, but still tied to the WiFi network.
 * - In both modes, packets are accepted only from the WiFi subnet, and only from the first
 *   device that sends (the armband). Everything else is dropped and counted.
 *
 * The armband has modes (EEG = data, impedance, test). Like the SDK's sendInstruction, this
 * class switches it to EEG mode by sending [0,0,0,0,0] back to the address/port it streams
 * from; the armband acknowledges with a 5-byte packet.
 *
 * Two packet formats become samples: 216-byte WUA2 packets (this project's armband firmware;
 * decoded by yad.core.MindRoveWua2, as the Pi's Python SDK does) and the SDK's MES packets
 * (type 2..5 in the first two bytes). Anything else is counted by size and ignored.
 *
 * Status is published at most every [PUBLISH_MS] so a fast stream can't flood the UI.
 */
class ArmbandReceiver(private val context: Context) {
    data class Status(
        val running: Boolean = false,
        val mode: String = "",
        val wifi: String? = null,
        val sender: String? = null,
        val datagrams: Long = 0,
        val emgPackets: Long = 0,
        val otherPackets: Long = 0,
        /** Non-EMG packets grouped by size: size -> (count, example of the first bytes). */
        val bySize: Map<Int, Pair<Long, String>> = emptyMap(),
        val samples: Long = 0,
        val lostSamples: Long = 0,
        val rejectedPackets: Long = 0,
        val samplesPerSecond: Double = 0.0,
        val packetsPerSecond: Double = 0.0,
        val battery: Int? = null,
        val eegRequests: Int = 0,
        val eegAcknowledged: Boolean = false,
        val sendError: String? = null,
        val capture: String? = null,
        val format: String? = null,
        val error: String? = null,
    )

    private val _status = MutableStateFlow(Status())
    val status: StateFlow<Status> = _status

    /** Raw samples, oldest first. Unlimited so nothing is silently dropped; see [backlog]. */
    val samples = Channel<DoubleArray>(Channel.UNLIMITED)
    val backlog = AtomicInteger(0)

    private var socket: DatagramSocket? = null
    private var network: Network? = null
    private var job: Job? = null
    private val eegRequested = AtomicBoolean(false)
    private val captureRequest = AtomicReference<((ByteArray, Int) -> Unit)?>(null)

    fun start(scope: CoroutineScope, acceptBroadcast: Boolean) {
        stop()
        val wifi = findWifi()
        if (wifi == null) {
            _status.value = Status(error = "Not connected to WiFi. Connect the phone to the armband's network (MindRove_ARB_…) first.")
            return
        }
        val (net, link) = wifi
        val local = link.address as Inet4Address
        val sock = try {
            DatagramSocket(null).apply {
                reuseAddress = true
                bind(InetSocketAddress(if (acceptBroadcast) InetAddress.getByName("0.0.0.0") else local, PORT))
                net.bindSocket(this)
                soTimeout = 200
                receiveBufferSize = 1 shl 20   // ~2 s of packets, so a busy moment doesn't drop data
            }
        } catch (e: Exception) {
            _status.value = Status(error = "Couldn't open UDP $PORT: ${e.message}")
            return
        }
        socket = sock
        network = net
        val mode = if (acceptBroadcast) "all addresses on WiFi, filtered to the armband" else "WiFi address ${local.hostAddress} only"
        _status.value = Status(running = true, mode = mode, wifi = "${local.hostAddress}/${link.prefixLength}")
        job = scope.launch(Dispatchers.IO) { receiveLoop(sock, net, link) }
    }

    fun stop() {
        job?.cancel()
        job = null
        socket?.close()
        socket = null
        _status.update { it.copy(running = false) }
    }

    /** Ask the armband for EEG (data) mode now, in addition to the automatic requests. */
    fun requestEegMode() = eegRequested.set(true)

    /**
     * Record the next [seconds] of raw packets (every packet that reaches the socket, before any
     * filter) and pass the file bytes to [onDone]. Format, big-endian: b"YADP", int32 version=1,
     * then per packet: int64 nanos since start, 4-byte source IPv4, uint16 source port,
     * int32 length, bytes.
     */
    fun captureRaw(seconds: Int, onDone: (ByteArray, Int) -> Unit) {
        captureSeconds = seconds
        captureRequest.set(onDone)
    }

    @Volatile private var captureSeconds = 5

    private suspend fun receiveLoop(sock: DatagramSocket, net: Network, link: LinkAddress) {
        val buf = ByteArray(2048)
        val packet = DatagramPacket(buf, buf.size)
        var sender: InetSocketAddress? = null
        var lastMeasurement = -1L
        var lastEmgAt = 0L
        var lastRequestAt = 0L
        var autoRequests = 0
        var lastPublish = 0L
        var rateStart = System.nanoTime()
        var rateSamples = 0L
        var ratePackets = 0L

        // Accumulated here, published every PUBLISH_MS.
        var datagrams = 0L; var emg = 0L; var other = 0L; var rejected = 0L
        var sampleCount = 0L; var lost = 0L; var battery: Int? = null
        var sps = 0.0; var pps = 0.0
        var format: String? = null
        val bySize = HashMap<Int, Pair<Long, String>>()

        // Raw capture in progress.
        var capOut: ByteArrayOutputStream? = null
        var capData: DataOutputStream? = null
        var capStart = 0L
        var capCount = 0
        var capCallback: ((ByteArray, Int) -> Unit)? = null

        while (currentCoroutineContext().isActive && !sock.isClosed) {
            val now = System.nanoTime()

            captureRequest.getAndSet(null)?.let { cb ->
                capOut = ByteArrayOutputStream()
                capData = DataOutputStream(capOut).apply { writeBytes("YADP"); writeInt(1) }
                capStart = now; capCount = 0; capCallback = cb
                _status.update { it.copy(capture = "recording ${captureSeconds} s of raw packets…") }
            }
            if (capData != null && (now - capStart > captureSeconds * 1_000_000_000L || capOut!!.size() > MAX_CAPTURE_BYTES)) {
                capData!!.flush()
                capCallback?.invoke(capOut!!.toByteArray(), capCount)
                capOut = null; capData = null; capCallback = null
            }

            // Switch the armband to EEG mode while no EMG is flowing (at most once a second, 5 times),
            // or when the user asks.
            val target = sender
            if (target != null && now - lastRequestAt > 1_000_000_000L &&
                (eegRequested.getAndSet(false) || (now - lastEmgAt > 1_000_000_000L && autoRequests < MAX_AUTO_EEG_REQUESTS))
            ) {
                autoRequests++
                lastRequestAt = now
                val err = sendEeg(sock, net, target)
                _status.update { it.copy(eegRequests = it.eegRequests + 1, sendError = err ?: it.sendError) }
            }

            if (now - lastPublish > PUBLISH_MS * 1_000_000L) {
                lastPublish = now
                val elapsed = (now - rateStart) / 1e9
                if (elapsed > 2) {
                    sps = rateSamples / elapsed; pps = ratePackets / elapsed
                    rateStart = now; rateSamples = 0; ratePackets = 0
                }
                val sizes = bySize.entries.sortedByDescending { it.value.first }.take(8).associate { it.key to it.value }
                _status.update {
                    it.copy(
                        datagrams = datagrams, emgPackets = emg, otherPackets = other, rejectedPackets = rejected,
                        samples = sampleCount, lostSamples = lost, battery = battery,
                        samplesPerSecond = sps, packetsPerSecond = pps, bySize = sizes,
                        sender = target?.let { s -> "${s.address.hostAddress}:${s.port}" },
                        format = format,
                    )
                }
            }

            try {
                packet.length = buf.size
                sock.receive(packet)
            } catch (_: SocketTimeoutException) {
                continue
            } catch (e: SocketException) {
                if (!sock.isClosed) _status.update { it.copy(error = "Socket error: ${e.message}") }
                break
            }
            datagrams++
            ratePackets++
            val len = packet.length
            capData?.let { d ->
                d.writeLong(System.nanoTime() - capStart)
                d.write((packet.address as? Inet4Address)?.address ?: ByteArray(4))
                d.writeShort(packet.port)
                d.writeInt(len)
                d.write(buf, 0, len)
                capCount++
            }

            if (!inSubnet(packet.address, link) || (sender != null && packet.address != sender.address)) {
                rejected++
                continue
            }
            if (sender == null) sender = InetSocketAddress(packet.address, packet.port)
            if (len == EEG_MODE.size) {           // the armband's reply to a mode instruction
                _status.update { it.copy(eegAcknowledged = true) }
                continue
            }
            // Decode: WUA2 (216-byte packets, this armband's firmware) or the SDK's MES packets.
            val decoded: List<Pair<DoubleArray, Long>>
            if (MindRoveWua2.isWua2(len)) {
                val ws = MindRoveWua2.decode(buf, len)
                decoded = ws.map { it.emgUv to it.counter }
                battery = ws.last().batteryPercent.roundToInt()
                format = "WUA2 (216-byte packets, older firmware)"
            } else {
                val type = if (len >= 2) (buf[0].toInt() and 0xFF) or ((buf[1].toInt() and 0xFF) shl 8) else -1
                val sdk: List<SensorData> = if (type in EMG_TYPES) {
                    try { SensorData.fromUdpPacket(packet) } catch (_: Exception) { emptyList() }
                } else emptyList()
                decoded = sdk.map { d ->
                    doubleArrayOf(d.channel1, d.channel2, d.channel3, d.channel4,
                        d.channel5, d.channel6, d.channel7, d.channel8) to d.numberOfMeasurement.toLong()
                }
                if (sdk.isNotEmpty()) {
                    battery = sdk.last().voltage.toInt()
                    format = "MES type $type (MindRove SDK)"
                }
            }
            if (decoded.isEmpty()) {
                other++
                val prev = bySize[len]
                bySize[len] = ((prev?.first ?: 0L) + 1) to (prev?.second ?: hex(buf, len))
                continue
            }
            emg++
            lastEmgAt = System.nanoTime()
            for ((uv, counter) in decoded) {
                val gap = counter - lastMeasurement - 1
                if (lastMeasurement >= 0 && gap in 1 until 100_000) lost += gap
                lastMeasurement = counter
                samples.trySend(uv)
                backlog.incrementAndGet()
            }
            sampleCount += decoded.size
            rateSamples += decoded.size
        }
    }

    /** Sends the EEG-mode instruction; returns a description of the failure, or null on success. */
    private fun sendEeg(sock: DatagramSocket, net: Network, target: InetSocketAddress): String? {
        val first = try {
            sock.send(DatagramPacket(EEG_MODE, EEG_MODE.size, target))
            return null
        } catch (e: Exception) {
            "from port $PORT to ${target.address.hostAddress}:${target.port}: ${e.message}"
        }
        // Retry from a fresh socket on the WiFi network (never mobile data).
        val second = try {
            DatagramSocket().use { s ->
                net.bindSocket(s)
                s.send(DatagramPacket(EEG_MODE, EEG_MODE.size, target))
            }
            return "$first; fresh socket worked"
        } catch (e: Exception) {
            "fresh socket: ${e.message}"
        }
        return "$first; $second; network: ${networkInfo(net)}"
    }

    @Suppress("DEPRECATION")
    private fun networkInfo(net: Network): String {
        val cm = context.getSystemService(ConnectivityManager::class.java)
        val present = cm.allNetworks.contains(net)
        val lp = cm.getLinkProperties(net)
        val caps = cm.getNetworkCapabilities(net)
        return "present=$present iface=${lp?.interfaceName} addrs=${lp?.linkAddresses} routes=${lp?.routes?.joinToString { it.toString() }} " +
            "validated=${caps?.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)} default=${cm.activeNetwork == net}"
    }

    /** The WiFi network and the phone's IPv4 address on it. */
    @Suppress("DEPRECATION")   // allNetworks: the simplest way to find a WiFi network that isn't the default
    private fun findWifi(): Pair<Network, LinkAddress>? {
        val cm = context.getSystemService(ConnectivityManager::class.java)
        for (n in cm.allNetworks) {
            val caps = cm.getNetworkCapabilities(n) ?: continue
            if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) continue
            val link = cm.getLinkProperties(n)?.linkAddresses?.firstOrNull { it.address is Inet4Address } ?: continue
            return n to link
        }
        return null
    }

    /** Everything a bug report needs, as plain text. */
    fun diagnosticsText(): String {
        val s = _status.value
        val net = network
        return buildString {
            appendLine("Project Yad armband diagnostics")
            appendLine("mode: ${s.mode}; wifi: ${s.wifi}; sender: ${s.sender}; format: ${s.format}")
            appendLine("packets: ${s.datagrams} (%.0f/s), EMG: ${s.emgPackets}, other: ${s.otherPackets}, rejected: ${s.rejectedPackets}".format(s.packetsPerSecond))
            appendLine("samples: ${s.samples} (%.0f/s), lost: ${s.lostSamples}, battery: ${s.battery}".format(s.samplesPerSecond))
            appendLine("EEG requests: ${s.eegRequests}, acknowledged: ${s.eegAcknowledged}")
            s.sendError?.let { appendLine("send error: $it") }
            s.error?.let { appendLine("error: $it") }
            appendLine("non-EMG packets by size:")
            s.bySize.forEach { (size, v) -> appendLine("  $size bytes × ${v.first}: ${v.second}") }
            net?.let { appendLine("network: ${networkInfo(it)}") }
        }
    }

    private fun hex(b: ByteArray, len: Int) = b.copyOfRange(0, minOf(len, 32)).joinToString(" ") { "%02x".format(it) }

    private fun inSubnet(addr: InetAddress, link: LinkAddress): Boolean {
        if (addr !is Inet4Address) return false
        val a = addr.address
        val b = link.address.address
        val bits = link.prefixLength
        for (i in 0 until 4) {
            val maskBits = (bits - i * 8).coerceIn(0, 8)
            val mask = if (maskBits == 0) 0 else (0xFF shl (8 - maskBits)) and 0xFF
            if ((a[i].toInt() and mask) != (b[i].toInt() and mask)) return false
        }
        return true
    }

    companion object {
        const val PORT = 4210                    // what the armband streams to (SDK ServerThread.serverPort)
        private val EMG_TYPES = 2..5             // packet types the SDK decoder handles as EMG
        private val EEG_MODE = ByteArray(5)      // Instruction.EEG (value 0) + 4 zero bytes, as the SDK sends it
        private const val MAX_AUTO_EEG_REQUESTS = 5
        private const val PUBLISH_MS = 300L
        private const val MAX_CAPTURE_BYTES = 8 * 1024 * 1024
    }
}
