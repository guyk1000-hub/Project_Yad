# Project Yad — Phone App Design

Status: **draft for review** · Branch: `phone-app` (local only, nothing pushed)

## 1. Goals

1. Run the full system on a Samsung Galaxy phone instead of the Raspberry Pi. The phone receives EMG from the MindRove armband over WiFi, classifies it, and drives the Haifa3D hand over BLE.
2. An easy, intuitive Android app with manual overrides for inference and training.
3. A modular design: replacing the hand, a model, or the processor should be a contained change.

## 2. Hard constraints

| Constraint | Consequence |
|---|---|
| The Pi system works and must not break | `files_for_pi/`, `training_data/` and their behaviour stay untouched. All new work goes in new folders. |
| Re-recording costs 30+ min, and setup can take over an hour | **No re-recording to build or validate the phone app.** Everything that can be is verified against the existing `.pkl` recordings. Hardware sessions are short, batched, and planned in advance (see §9). |
| Keep the existing models | The phone runs today's `mlp_model_<user>.pkl` and `scaler_<user>.pkl` unchanged, exported to a neutral format. Training stays on the PC. |
| Never push without approval | All work stays on the local `phone-app` branch until the user approves a push. |

## 3. Architecture

```
 Armband source ──► Filter ──► Windowing ──► Features ──► Classifier ──► Decision ──► Hand sink
 (MindRove/replay)  (3 biquads) (8×100)     (CNN+2 scalars) (scaler+MLP) (smoothing,   (Haifa3D BLE /
                                                                          threshold)    simulated)
                                                                              ▲
                                    App: overrides · e-stop · sliders · profiles · recording · corrections
```

Every stage is an interface with at least two implementations: a real one, and a fake one for testing without hardware.

| Stage | Contract | Implementations |
|---|---|---|
| `EmgSource` | Stream of 8-channel samples in µV at 500 Hz | `MindRoveSource` (Android SDK), `ReplaySource` (recorded windows) |
| `Filter` | Stateful per-channel biquad cascade | Exact port of `BiquadMultiChan` |
| `Windower` | 100-sample windows shaped `[8, 100]` | Pi-compatible (non-overlapping), sliding (later) |
| `FeatureExtractor` | Window → 1026 floats | TFLite CNN (`dense_8`, 1024) + `asym`, `mag` |
| `Classifier` | 1026 floats → class probabilities | Exported scaler + MLP |
| `DecisionPolicy` | Probability stream → gesture events | Pi-compatible (average 6, threshold 0.7), sliding (later) |
| `HandSink` | Gesture/command → hand | `Haifa3dHand` (BLE, protocol from `protocol.py`), `SimulatedHand` |

Overrides (manual commands, e-stop) go **directly to `HandSink`** and never pass through the classifier.

## 4. Parity: the phone must compute exactly what the Pi computes

This is the main technical risk, and it's tested entirely offline.

- **Reference = current Python code, quirks included.** The scalar features (`asym`, `mag`) exist in **two versions**, and models were trained with both. Each exported head records which one it needs, and the phone implements both:
  - `channels`: RMS over axis=1 of `[8, L]`. This is what the Pi computes today; used by `tomer_long`, `guy` and `tomer_250`.
  - `time`: RMS over axis=2, from commit `23c0296`; used by `mlp_model_tomer`.

  Mismatching costs accuracy: `mlp_model_tomer` scores 99.7% on its recording with the right version and 86.7% with the wrong one. This is also why the Pi must keep `tomer_long` for now.
- **Golden vectors:** a Python script generates input→output pairs for each stage. Kotlin unit tests must reproduce them within a tolerance.
  - Filters: synthetic raw signals (the recordings are stored after filtering, so they can't test the filter).
  - Features, scaler, MLP, end-to-end: real windows from the existing recordings.
- **Units:** both SDKs decode EMG as a signed 24-bit integer × 0.045 µV. I found the Android constant by disassembling the library, and the same constant is present in the Python SDK's native wrapper. A 2-minute live check (§9, session A) compares at-rest RMS on the phone with the rest class in the user's recordings.

## 5. Model artifacts

| Artifact | Source | Phone format |
|---|---|---|
| CNN feature extractor | `og_feature_extraction.h5` | `feature_extractor.tflite` (input `[1,8,100,1]`, output `dense_8`, 1024). `MyMagnWarping`, `MyScaling` and `GaussianNoise` do nothing at inference, so they're removed before conversion. |
| Scaler + MLP | `scaler_<user>.pkl`, `mlp_model_<user>.pkl` | `head_<user>.json`: scaler `mean_`/`scale_`, layer weights and biases, activations (`tanh` hidden, `softmax` output), class list, a source-file hash, and an export date |

Parity checks before an artifact is accepted:
- TFLite vs Keras features on every recorded window.
- JSON head vs sklearn `predict_proba` on every recorded window.
- Prediction agreement must be 100%; probability error goes in the report.

The `.h5` was saved with Keras 2.10, so the export tooling runs in a pinned Python 3.10 + TF 2.12 environment. That matches `requirements.txt`.

## 6. Repository layout (new folders only)

```
files_for_phone/            everything for the phone, alongside files_for_pi/ (the Pi keeps running as-is)
  DESIGN.md                this document
  models/                  exported artifacts the app ships: head_<user>.json (committed), feature_extractor.tflite (regenerated)
  tools/                   PC-side Python, isolated venv, never imported by the Pi code
    export_head.py         pkl scaler+MLP → head_<user>.json (+ parity check)
    export_cnn.py          h5 → tflite (+ parity check)
    make_golden.py         golden vectors for Kotlin tests (→ tools/golden/)
    import_recording.py    phone recording → (recorded_data, recorded_labels) .pkl for fine_tune.py
    requirements.txt
  android/                 Kotlin app (Gradle)
    core/                  pure Kotlin, JVM-testable: filters, windowing, features, head, decision policies,
                           command gate (Stop), replay file reader
    app/                   the Android app, one module with packages (split into modules only if it starts to hurt):
      ml/                  TFLite feature extractor
      hand/                simulated hand (M3), Haifa3D BLE (M4)
      armband/             MindRove SDK source (M4)
      ui/                  Jetpack Compose screens
```

## 7. App features

**Inference screen:** live gesture, a confidence bar and connection status for the armband and the hand.
- **E-stop:** a large button that is always visible.
- **Manual buttons:** open, close, left, right.
- **Pause/resume classification.**
- **"That was wrong" button:** logs the window and the predicted class, then asks for the correct class (one tap).

**Sensitivity sliders:** confidence threshold and smoothing length, saved per user. A "Pi-compatible" preset restores the defaults (6 / 0.7).

**Profiles:** several users per phone. Each has their own model versions, settings, recordings and correction log, and can switch models with one tap.

**Recording:** a guided session using the same protocol as `record.py` (gesture image, countdown, accept/redo). It exports to the PC through `tools/import_recording.py`.

**Learning from corrections**, staged:
1. Log corrections only.
2. Export to the PC and retrain there with `fine_tune.py` (original data + corrections).
3. Later, optionally, train the head on the phone.

A new model is accepted only if it does no worse on held-out data. Every version is kept, with one-tap rollback.

## 8. Latency

The phone first runs **Pi-compatible mode**: the same decisions every ~1.2 s, to prove parity. Then it adds a **sliding-window mode**: decisions every ~100 ms from a rolling average of recent windows. This is checked offline by replaying the recordings, so the decisions are compared before anyone wears the armband. Stage timings are logged in the app.

**The CNN cost (measured in M1):**
- About 3.7 G multiply-adds per window. That's ~78 ms per window on one core of a Xeon PC, against 200 ms of signal per window.
- Pi-compatible mode needs 1 window per 200 ms. That's comfortable on the phone (S23 FE, Exynos 2200) with 2–4 XNNPACK threads, which M3 will measure.
- Sliding-window mode is limited by this cost: a new window every 50 ms needs under 50 ms per window. If needed: more threads, the GPU delegate, or fp16 (only if parity holds).
- *Hypothesis about the Pi's latency (unmeasured):* the Pi 5 runs this CNN in full Keras on batches of 6. If it can't keep up with 1 window per 200 ms, its queue grows and the delay keeps increasing over a session. A timing log on the Pi would confirm or rule this out.

## 9. Milestones

| # | Milestone | Needs hardware? |
|---|---|---|
| M0 ✅ | `files_for_phone/tools/` environment, `export_head.py`, `make_golden.py`; parity of the JSON head vs sklearn. Done 2026-11-14: 4 heads exported with 100% prediction agreement; `tomer_with_finger` was refused because its model and recording don't match | No |
| M1 ✅ | `export_cnn.py`; TFLite vs Keras parity; end-to-end accuracy on recordings matches the Python pipeline. Done 2026-11-14: a 19.3 MB float32 model using built-in ops only (no Flex). Over 20,250 windows: feature error ≤1.1e-6, 0 changed predictions | No |
| M2 ✅ | `android/core` + unit tests passing on the golden vectors (JVM, on the PC). Done 2026-11-14: 12 tests pass; filters bit-identical, head within 6.6e-11, decisions identical; the tests were checked with planted bugs | No |
| M3 ✅ | App skeleton: replay source + simulated hand, inference screen, overrides, sliders. Built 2026-11-14: replay of 3 users, STOP, manual buttons, sliders + Pi preset, stats, CNN benchmark, and an on-device parity check at startup. Installed on the S23 FE: the on-device check passed (CNN rel. error 8.4e-7, probabilities ±4.3e-10, 10/10 windows). CNN ~113 ms/window on one core (1–4 threads the same); replay 99.6% of windows correct | Phone only (install APK) |
| M4 ✅ | **Hardware session A (~15 min, no recording):** armband streaming + units check at rest; hand BLE: manual buttons and e-stop . Code built 2026-11-14 (APK 0.4.0-m4): own UDP receiver, Haifa3D BLE hand, units check, engine benchmark. Session A done: streaming works (WUA2 decoder); live accuracy poor, see FINDINGS.md | Phone + armband + hand |
| M5 🔨 | **Per-session calibration on the phone** (added after session A): guided 10 s/gesture recording → retrain the head on the phone (old recording + new data) → saved as a model version. Built 2026-11-15 (APK 0.5.0-m5); Kotlin trainer matches sklearn offline (guy 83% vs 77%, tomer 96% vs 99%); awaiting session B | Phone + armband |
| M5b | Live inference in Pi-compatible mode. **Session B (~15 min):** perform each gesture for a few seconds, and compare with the Pi if available | Phone + armband + hand |
| M6 | Sliding-window latency mode (offline-tuned first) | Brief check in a later session |
| M7 | Profiles, model registry, recording + export, correction logging | Phone (+ armband for recording) |
| M8 | Retraining from corrections (PC, then optionally the phone) | Uses data from real use |

M0–M2 need nothing from you. M3 needs you to install an APK on the Galaxy phone; I'll provide step-by-step instructions.

## 10. Open questions

Resolved:
- **Pi runtime (from the Pi, 2026-11-14):**
  - Its config points to `mlp_model_tomer_long.pkl` + `scaler_tomer_long.pkl`. All model files, including the `.h5`, are byte-identical to the repo.
  - The intended model is **`mlp_model_tomer`**: `_long` was a trial that performed worse. The phone defaults to `mlp_model_tomer`; `_long` is exported too, for comparison.
  - It doesn't set `batch_size` or `prediction_threshold`, so the defaults of 6 and 0.7 apply.
  - Software: Python 3.10.11, TF/Keras 2.12.0, numpy 1.23.5, scikit-learn 1.7.2, mindrove 5.2.3. `files_for_phone/tools/` pins these versions.
- **E-stop:** a single "Stop" button immediately blocks every classifier command from reaching the hand. Manual buttons keep working. It doesn't send "open" and doesn't disconnect BLE.
- **Phone:** Samsung Galaxy S23 FE. Target minSdk 33 (Android 13, its launch version). It needs the BLE permissions `BLUETOOTH_SCAN` and `BLUETOOTH_CONNECT`.
- **Hand pairing:** none. Connect directly to the address, as `basic_commands_pi.py` does.

## 11. Risks

| Risk | Mitigation |
|---|---|
| A TFLite op isn't supported for the CNN's custom blocks | Remove the custom layers; use TF Select ops as a fallback; the parity check catches any drift |
| CNN too slow on the phone | Measure in M3 with replay. Options: XNNPACK/GPU delegate, fp16 (only if parity holds) |
| Phone on armband WiFi (no internet): Android may route traffic to mobile data | Bind the app to the WiFi network; the UDP listener on port 4210 isn't affected; documented in the setup screen |
| Wrong commands to the hand during development | `SimulatedHand` by default; the real hand needs an explicit opt-in; e-stop always available |

## 12. Security notes (M4)

- **Armband socket.** The MindRove SDK's `ServerManager` listens on UDP 4210 on *all* interfaces, mobile data included. Fake EMG there could move the hand. So the app owns the socket and uses the SDK only to decode packets:
  - It binds to the phone's WiFi address by default.
  - Binding to all addresses (to receive broadcasts) is an explicit switch in the app.
  - Either way, the socket is tied to the WiFi network. It accepts only packets from the WiFi subnet, and locks onto the first device that streams. Rejected packets are counted on screen.
- **Armband WiFi password.** The armband's network uses MindRove's default password (`#mindrove`, also hard-coded in `files_for_pi/wifi_connect_pi.py`). Anyone nearby who knows it can join that network. If MindRove allows changing it, do so. Then keep the new password out of git, e.g. in a file the Pi reads.
- **Hand BLE.** The Haifa3D hand accepts commands from any BLE device without pairing. This is a property of the hand, so the app can't fix it. In the app, the real hand is off by default, and STOP always blocks the classifier.
