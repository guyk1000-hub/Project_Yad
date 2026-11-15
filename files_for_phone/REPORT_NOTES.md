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
| 2026-11-15 | Train on the phone (own L-BFGS), not on the PC | No PC round trip each session; matches sklearn offline |

## Ethics notes (to expand)

- Medical-adjacent device: a misclassification moves a hand. Mitigations: STOP, manual override,
  real hand off by default, decisions only above a confidence threshold.
- Data: EMG recordings are personal biometric data; kept locally, not uploaded anywhere.
- Security: open BLE on the hand and the armband's default WiFi password are known risks.
