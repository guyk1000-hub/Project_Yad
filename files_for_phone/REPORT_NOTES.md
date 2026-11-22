# Report notes — phone version of Project Yad

Running notes for the final report. They follow the structure of the earlier project report
(Fill-Harmonic): Goal → Motivation → System / Method → Experiments (tables) → Results → Ethics.
Each entry: what we did, why, and the numbers. Newest last within each section.

## Goal

Move the EMG gesture-control system from a Raspberry Pi 5 to an Android phone (Galaxy S23 FE)
and build an easy app with manual overrides. The phone is the whole processor: it reads the
MindRove armband over WiFi and drives the Haifa3D hand over BLE. The Pi version was the
beta and is no longer part of the project (decided 2026-11-15).

## Motivation

- The Pi had noticeable latency and needed a screen/keyboard setup to run.
- A phone is always with the user, has a touchscreen for overrides, and is faster.
- Android, not iOS: MindRove has an Android SDK but no iOS one; iOS also restricts background
  BLE and WiFi joining, and sideloading needs a paid developer account.

## System overview (unchanged from the Pi)

1. MindRove armband: 8 EMG channels, 500 Hz, streamed over its own WiFi network (UDP 4210).
2. Filters (per sample): 4.5 Hz high-pass, 50 Hz notch, 100 Hz low-pass (biquads).
3. Windows of 100 samples (200 ms), shape [8, 100].
4. Frozen NaviFlame CNN (pretrained by MindRove), `dense_8` layer → 1024 features.
5. + 2 hand-made features (asym, mag) → 1026.
6. Per-user classifier: StandardScaler + MLP (32 → 16, tanh) → 5 classes (rest, close, open, right, left).
7. Decision: average 6 windows, emit if top probability ≥ 0.7 (one decision per 1.2 s).
8. Hand: 5-byte BLE "direct execute" commands; rest sends nothing.

## Method — how we ported it

### M0–M2: prove the phone computes exactly what the Pi computes (no hardware)

- Approach: "golden vectors". Python scripts generate input→output pairs using the Pi's own
  code; Kotlin tests must reproduce them. Tests were themselves checked by planting bugs.
- Models kept as they are: sklearn scaler + MLP exported to JSON; CNN converted to TFLite
  (19.3 MB, builtin ops only; custom augmentation layers are identity at inference and dropped).

| Stage | Phone vs Pi |
|---|---|
| Filters | bit-identical (difference 0.0) |
| Scalar features | ≤ 1.5e-7 relative |
| CNN (TFLite vs Keras, 20,250 windows) | ≤ 1.1e-6 relative, 0 changed predictions |
| MLP head | ≤ 6.6e-11 |
| Decisions, hand bytes | identical |

### Discoveries along the way

- **Two versions of the hand-made features.** `mlp_model_tomer` was trained with RMS over time
  (axis=2, a fix on 13 Apr), the other models with RMS over channels (axis=1, the code after
  15 Apr). Feeding the wrong version: 99.7% → 86.7%. Each exported model now records its version.
- **Broken math library on the dev PC.** numpy 1.23.5's OpenBLAS returns wrong float64 matrix
  products on Intel Sapphire Rapids CPUs; worked around with OPENBLAS_CORETYPE=Haswell + a self-test.
- **Armband packet format.** The armband's firmware sends 216-byte "WUA2" packets, which the
  MindRove Android SDK can't decode (it only knows newer packet types). The layout was read from
  MindRove's Python SDK library (`WUA2_PACKET::interpret`) and reimplemented. The armband also
  streams to only one device, chosen when it powers on.
- **Security.** The SDK listens on all network interfaces (incl. mobile data); a spoofed stream
  could move the hand. The app owns the socket: WiFi only, armband's subnet, first sender only.
  The hand accepts BLE commands from anyone (hardware property); real hand is off by default.

### M3: the app (replay)

- Plays recordings through the real pipeline into a simulated hand. STOP blocks the classifier
  (manual buttons still work), manual buttons, sensitivity sliders, on-device parity check.
- On the S23 FE: on-device check passed (CNN error 8.4e-7); replay 99.6% of windows correct;
  decisions 100% wherever one gesture was held (first scoring showed 96% — groups spanning a
  gesture change were scored against only their last window).

## Experiments

### E-speed: CNN on the phone (S23 FE, ms per window, budget 200 ms)

| Engine | ms/window |
|---|---|
| CPU 1 thread (XNNPACK) | 111.5 |
| CPU 2 / 4 threads (XNNPACK) | 115.5 / 117.7 |
| CPU 4 threads, no XNNPACK | **94.9** |
| CPU fp16 allowed (1 / 4 threads) | 122 / 121 |
| GPU (fp32 / fp16) | not available (delegate fails on this GPU) |

### E-live: live accuracy, session A (14 Nov 2026), guided 35 s check (6 s per gesture)

| User / model | Accuracy | REST | CLOSE | OPEN | RIGHT | LEFT |
|---|---|---|---|---|---|---|
| Guy / guy | 46% | 25 | 62 | 88 | 38 | 17 |
| Tomer / tomer | 74% | 96 | 33 | 54 | 88 | 100 |

The same data through the Pi's own Python code gives identical predictions (|Δp| ~1e-5):
the phone is not the cause. No rotation/mirroring of the armband explains it.

### E-generalization: why — models don't carry over between sessions

Honest validation (hold out a whole take; the original random split of 90%-overlapping windows
put near-copies of training windows in the validation set and showed ~99%):

| Model | Held-out take (same session) | Other session (tomer ↔ tomer_long) |
|---|---|---|
| guy | 87–96% | — |
| tomer | 64–90% | 46–48% (OPEN 0%) |
| tomer_long | 94–99% | |

### E-adapt: a few seconds of new data fixes it

Adapt on the first 3 s of each gesture from the check, test on the last 3 s:

| Approach | Guy | Tomer |
|---|---|---|
| Existing model | 48% | 72% |
| Scaler refit only | 22% | — |
| Retrain on old data + 3 s/gesture today | **95%** | **99%** |
| Retrain on today's 3 s only | 86% | 86% |

(Optimistic: test windows are from the same hold. → Motivates M5: per-session calibration.)

### M5: per-session calibration on the phone

- Guided recording: 10 s per gesture; windows every 25 samples, inside one gesture, from 1 s in.
- Retraining on the phone: the user's original recording (exported as 1026 head inputs, every
  3rd window = 1450 windows) + the new windows, weighted to 37% of the total; same network as
  fine_tune.py (32-16 tanh, L2 0.01), own L-BFGS in Kotlin, 300 iterations. Saved as a model
  version (`user · dd/mm hh:mm`), selectable and deletable in the app.
- Recipe chosen offline on the 14 Nov checks (train on 2.5 s/gesture, test on the last 3 s):

| Recipe (sklearn) | Guy | Tomer |
|---|---|---|
| windows every 10, new = 29%, full training | 90.1% | — |
| windows every 10, new = 29%, 300 iterations | 80.3% | 97.2% |
| windows every 50, new = 20%, 300 iterations | 69.3% | 95.0% |
| windows every 50, new = 37%, 300 iterations | 77.0% | 98.7% |
| **Kotlin trainer**, every 50, new = 37%, 300 iterations | **83.1%** | **95.9%** |

  More data per gesture matters most (Guy); real calibration records 4× more per gesture than
  this test. Kotlin trainer: 33 s for 300 iterations on the dev PC.

### E-calib: session B (15 Nov 2026), calibrated on the phone, then guided check

Calibration: 10 s per gesture, retrained on the phone (old recording + new windows, 37% weight),
~5 min on the S23 FE. Then a separate 35 s guided check (a new hold, not the calibration data).

| User / model | Accuracy | REST | CLOSE | OPEN | RIGHT | LEFT |
|---|---|---|---|---|---|---|
| Tomer / tomer (session A, no calibration) | 74% | 96 | 33 | 54 | 88 | 100 |
| **Tomer / tomer@20261115_1717 (calibrated)** | **94%** (113/120) | 95 | 100 | 100 | 95 | 75 (else RIGHT ×5) |
| Tomer, 2nd check later the same session (from the screenshot, file not analysed) | ~63% | 75 | 95 | 100 | 45 | 0 |

Same check (17:18) re-run on the PC with other training data (sklearn, phone recipe). The
calibration is 50 s of new data = 885 windows (177 per gesture):

| Trained on | Check accuracy | REST | CLOSE | OPEN | RIGHT | LEFT |
|---|---|---|---|---|---|---|
| Old recording only (bundled `tomer`) | 55% | 100 | 0 | 0 | 75 | 100 |
| Old + calibration, 300 it (= the phone's recipe; phone: 93%) | 94% | 100 | 100 | 96 | 92 | 83 |
| Old + calibration, 2000 it | 95% | 100 | 88 | 96 | 96 | 96 |
| **Calibration only** (from scratch, 210 it) | **98%** | 100 | 100 | 100 | 92 | 100 |

With 10 s per gesture, today's data alone beats old + today (with 3 s it was 86% vs 99%, E-adapt).
The phone's own trainer matches sklearn (93% vs 94%).


### E-calib2: second calibration (17:49) and checks across 50 min, Tomer, 15 Nov

On the phone (model `tomer@20261115_1758` = old + 17:49 calibration): checks at 18:02 **90%**
(LEFT 50%) and 18:05 **93%**. The exported model rerun on the PC gives the same probabilities
(max |Δp| 6e-6): the phone's trainer and inference are right.

All variants (sklearn, 300 it, phone recipe), each scored on the three checks (all / LEFT):

| Trained on | check 17:18 | check 18:02 | check 18:05 |
|---|---|---|---|
| Old recording only | 55% / 100 | 49% / 17 | 52% / 21 |
| Old + calib 17:12 | 94% / 83 | 61% / 25 | 55% / 8 |
| Calib 17:12 only | 98% / 100 | 68% / 38 | 74% / 71 |
| Old + calib 17:49 | 78% / 96 | 88% / 42 | 95% / 79 |
| Calib 17:49 only | 80% / 100 | 92% / 58 | 99% / 100 |
| **Calib 17:12 + 17:49 only** | **99% / 100** | **91% / 54** | **98% / 92** |
| Old + both calibs | 97% / 88 | 89% / 54 | 93% / 71 |

Findings:
1. The old PC recording makes every variant worse: drop it, train on phone recordings.
2. Even within one session, a model from one recording degrades 30–50 min later (98% → 68–74%).
   Two recordings together hold up across all checks: train on all recordings of a person.
3. LEFT at 18:02 fails for every model (≤ 58%): that hold differed, not the model. LEFT vs RIGHT
   is the weakest pair.

### M5b: record and train from zero on the phone (0.6.0, 15 Nov)

- **Record** (3 takes × 10 s per gesture, 5 s break between takes, ~3 min) or **Quick
  calibration** (1 take, ~1 min). Each recording is kept on the phone (`files/recordings/<user>/`,
  plus a copy in Downloads), and its head inputs are cached so that retraining doesn't rerun the CNN.
- After every recording, the phone trains the user's model **from zero on all their phone
  recordings** (no PC recording; windows every 50 samples; 300 L-BFGS iterations). Before
  retraining, it reports how the previous phone model did on the new recording, which is data
  that model never saw: an honest cross-session number.
- Older calibration files can be imported from Downloads. A bad recording can be deleted and the
  model retrained.
- Names never collide. `tomer`, `guy` are the bundled PC models (read-only), `tomer@…` the old
  calibrated versions, and `tomer-phone@yyyyMMdd_HHmmss` the phone-trained ones. Saving refuses
  to overwrite.
- Verified offline before the hardware session. The Kotlin windowing on the real 15 Nov files
  equals the Python windowing (445 windows each, same labels). Kotlin training from zero on both
  calibrations scores **97 / 93 / 97%** on the three checks (sklearn: 99 / 92 / 98%), in 19 s on the PC.

### E-phone: first model trained from zero on the phone (Tomer, 15 Nov, 18:36)

One full phone recording (3 takes × 10 s per gesture, 1335 windows), trained from zero on the
phone (`tomer-phone@20261115_184045`). Guided check right after (18:41): **100%**, 120/120
windows, every gesture 100%. The PC rerun of the exported model gives the same probabilities
(max |Δp| 1.7e-7). The user: "working perfectly".

The same model on the session's earlier checks (data from before the recording):

| Check | Minutes before the recording | Accuracy | CLOSE | LEFT |
|---|---|---|---|---|
| 17:18 | 78 | 76% | 0 | 100 |
| 18:02 | 34 | 85% | 79 | 54 |
| 18:05 | 31 | 89% | 67 | 92 |
| 18:41 (after) | — | **100%** | 100 | 100 |

The signal drifts over an hour, so a single recording is best near the time it was made. Hence
the short calibrations, which add recordings over time.

### E-latency: decision rules and features on the bench (Tomer, 8 files, 6.7 min of gestures)

`tools/bench.py` replays the labelled phone files at a 50 ms hop. Each file is scored with a model
trained on the other 7 (leave one file out; the 14 Nov check is a different day). Latency =
from the on-screen cue to the first correct decision, so it includes the person's own reaction
time (~0.3–0.5 s). A wrong move = the hand switching to a gesture that is neither the one being
held nor the one just before it. Phone compute time is not included.

| Features | Rule | Median latency | p90 | Reached | Steady accuracy | Wrong moves/min |
|---|---|---|---|---|---|---|
| CNN (today) | today: groups of 6 × 200 ms, ≥ 0.7 | 1.8 s | 3.0 s | 98% | 98% | 0 |
| CNN | 4 agree, every 200 ms, ≥ 0.7 | 1.4 s | 2.0 s | 98% | 99% | 0 |
| CNN | 3 agree, every 100 ms, ≥ 0.9 | 1.0 s | 1.5 s | 98% | 99% | 2.1 |
| CNN | sliding avg of 2, every 50 ms, ≥ 0.7 | 0.7 s | 1.0 s | 100% | 96% | 11.1 |
| **Classic (40 TD features, no CNN)** | today's rule | 1.8 s | 2.6 s | 100% | **100%** | 0 |
| **Classic** | **3 agree, every 200 ms, ≥ 0.7** | **1.0 s** | **1.6 s** | 100% | 100% | **0** |
| Classic | 4 agree, every 100 ms, ≥ 0.9 | 0.9 s | 1.6 s | 100% | 100% | 0.75 |
| Classic | sliding avg of 2, every 50 ms, ≥ 0.7 | 0.6 s | 0.9 s | 100% | 99% | 7.4 |

Classic features = MAV, RMS, waveform length, zero crossings and slope sign changes per channel
(Hudgins), with the same MLP. Compared with the CNN (~95 ms per window on the S23 FE), they take
microseconds.

Findings:
1. Most of today's latency is the decision rule (averaging 6 × 200 ms), not the model.
2. The classic features are at least as accurate as the CNN for a per-user model (14 Nov file:
   100% vs 79% with today's rule). They make fewer wrong moves at every speed and cost almost no compute.
3. Faster rules trade latency for wrong moves. About half of those happen in the first second
   after a gesture change (most often OPEN flashing on the way from REST to CLOSE). The other
   half come during steady holds.
4. Caveat: one person, one main day, 6.7 min of data. To be confirmed with Guy and on other days.

### Two hands, chosen in the app (0.7.0, 22 Nov)

The new hand (Haifa3D Hand-Controller-System v2) isn't complete yet: one motor board with 4
fingers, no wrist, and it only boots on USB power. We use it as it is, without changing it. The app
drives either hand, picked by hand, and each hand offers only what it supports:

| | Old hand (Haifa3D v1) | New hand (controller v2) |
|---|---|---|
| Connection | the phone connects to the hand (BLE central) | the hand connects to the phone (the phone advertises the "manager" service) |
| Commands | OPEN, CLOSE, LEFT, RIGHT (wrist) | OPEN, CLOSE, STOP, CLOSE A BIT, CALIBRATE |
| Speed | — | 10–100 (the hand's default is 50) |
| STOP button | blocks the classifier | blocks the classifier and stops the motors |
| LEFT / RIGHT gestures | move the wrist | logged as "not supported", nothing sent |

0.7.1: everything the new hand's own manager app ("Android Hand Commander", read from its APK
with `dexdump`: no source in the repo) offers, as far as the hand's code runs it:

| Manager app function | Bytes | Hand's code | In our app |
|---|---|---|---|
| Forward / Backward / Step / Stop / Calibrate | `[1]` `[2]` `[5]` `[0]` `[7]` | runs | buttons (calibrate asks first) |
| Velocity | `[8, 8, v]` | runs (reads byte 2) | speed slider |
| Position M1–M4 | `[6, motor, %]` | runs | a slider per finger |
| Direct Commands (8 editable presets, JSON load / save / share, defaults) | special codes | ignored | presets made of position / open / close, which the hand runs; editable, import / export JSON, defaults |
| Subscribe, specials, "Packet" (`0xF0`), "Screen packet" (`0xF4`) | — | ignored | shown as "not supported yet" |

0.7.2: the old hand gets everything the Haifa3D app offers (its public source and Protocol.md):

| Haifa3D app | BLE | In our app |
|---|---|---|
| Live control (hold a button: a 50 ms low-torque step every 40 ms) | direct execute | hold-to-move per motor (wrist, fingers 1–4) |
| Actions = movements (motors, direction, torque low/high, time × 50 ms) | direct execute `[1+4n, …]` | action builder (add / edit / delete movements, run) |
| 12 presets stored on the hand: read, edit, write, trigger | preset + trigger services | read all, edit in the builder, save to a slot, run |
| Configuration values (torque thresholds, slopes, filter, factors, unit, debug) + factory resets | configuration service | read / set each value, reset presets / configuration (with confirmation) |
| Battery level | standard battery service | read when connected and on tap |
| Voice commands ("speak to action") | — | placeholder |

The old encoder is now general (`Haifa3dProtocol.action`), and the Pi's four commands still come
out byte for byte the same (unit test, plus the golden test against `protocol.py`).

### M6: Fast mode, lower latency with the same model (0.8.0, 22 Nov)

The user's constraints: keep the CNN (the project's concept), leave the hands as they are, and
verify on the phone only. So the latency work is in the decision rule:
- **Groups** (as before): average N windows, one decision per group (N × 200 ms).
- **Fast:** decide after every window, as soon as the last N windows agree (same top class, each
  ≥ the threshold). It doesn't wait for a group boundary, so a change is acted on up to
  (group − 1) × 200 ms sooner. Optional 100 ms overlapping windows (the CNN takes ~95 ms).
  A switch guard asks for one window more when going straight between two moving gestures
  (the "OPEN flashing on the way to CLOSE" case). Commands are sent only on change.
- CNN engine default changed to 4 threads without XNNPACK (94.9 vs 111.5 ms per window on the S23
  FE, same fp32 results, checked by the on-device self-test).
- The guided check now measures the **reaction time** (cue → first right decision) per gesture,
  so Groups vs Fast is compared on the phone.

Results: to be measured on the phone (table to fill: mode, N, hop, accuracy, median reaction).

| Mode | Windows | Hop | Accuracy | Median reaction |
|---|---|---|---|---|
| Groups | 4 | 200 ms | | |
| Fast | 3 | 200 ms | | |
| Fast | 3 | 100 ms | | |

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Android + native Kotlin | SDK availability, BLE, TFLite |
| 2026-11-01 | Keep existing models, export weights | No retraining, same predictions |
| 2026-11-14 | All phone code in `files_for_phone/` | Pi keeps working during the move |
| 2026-11-14 | STOP = block classifier, manual still works | Safety without moving the hand |
| 2026-11-15 | Pi dropped from the project | Phone does everything the Pi did |
| 2026-11-15 | Next: per-session calibration on the phone (M5) | E-adapt results |
| 2026-11-15 | fine_tune.py reports held-out-take accuracy | Honest number |
| 2026-11-15 | Phone exports each calibrated model (JSON) | So checks with it can be re-run on the PC |
| 2026-11-15 | Record and train from zero on the phone, on all of a person's phone recordings; old PC data dropped | E-calib2 (user's idea) |
| 2026-11-15 | User-chosen model names: `<user>-<label>@<time>` | The user wanted names; the time keeps them unique |
| 2026-11-15 | Bench every latency idea offline first (`tools/bench.py`) | Latency vs wrong moves is a trade-off: measure it |
| 2026-11-22 | The new hand and its code are used as they are (not ours to change) | The user: not up to them |
| 2026-11-22 | The app supports both hands, switched by hand; each offers only what it supports | The new hand isn't complete; the project must still finish |
| 2026-11-22 | Today's screens are the engineering view; an easy end-user view comes later | The user's plan |
| 2026-11-22 | Keep the CNN; latency via the decision rule (Fast mode); verify on the phone only | The user: the CNN is the concept; no more PC checks |
| 2026-11-15 | Train on the phone (own L-BFGS), not on the PC | No PC round trip each session; matches sklearn offline |

## Ethics notes (to expand)

- Medical-adjacent device: a misclassification moves a hand. Mitigations: STOP, manual override,
  real hand off by default, decisions only above a confidence threshold.
- Data: EMG recordings are personal biometric data; kept locally, not uploaded anywhere.
- Security: open BLE on the hand and the armband's default WiFi password are known risks.
