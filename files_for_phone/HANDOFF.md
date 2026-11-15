# Handoff — Project Yad phone app (state on 2026-11-15, evening)

## What this is

EMG gesture control of a Haifa3D prosthetic hand. A MindRove armband (8 channels, 500 Hz, WiFi)
→ Android app on a Galaxy S23 FE → hand over BLE. It replaced a Raspberry Pi 5 version, which
is no longer part of the project (`files_for_pi/` and `training_data/` are kept only as reference).
Owners: Guy Katabi and Tomer Ohayon. Repo: github.com/guyk1000-hub/Project_Yad, branch **`phone-app`**.

## Ground rules from the user

- **Never push without explicit approval**, every time. The tool permission for `git push` may be
  denied even after approval in chat; then the user pushes themselves.
- Commit as **Guy Katabi <guyk1000@gmail.com>** (set in the repo's local git config). Never an amazon.com identity.
- Hardware time is scarce; re-recording costs 30+ min. Verify offline first (golden vectors,
  replays, check files) and batch hardware sessions with step-by-step instructions.
- Keep **`files_for_phone/REPORT_NOTES.md`** updated after each milestone/experiment (for the
  final report, styled like the user's Fill-Harmonic report: goal, motivation, method, experiments with tables, ethics).

## Where to read

| File | What |
|---|---|
| `files_for_phone/DESIGN.md` | Architecture, milestones M0–M5, security notes |
| `files_for_phone/FINDINGS.md` | Why live accuracy was poor (session A) and the evidence |
| `files_for_phone/REPORT_NOTES.md` | Running report notes, all result tables, decisions log |
| `files_for_phone/tools/README.md` | PC tools (pinned Python 3.10 venv) |
| `files_for_phone/android/README.md` | Build and test the app |

## State

- **Branch `phone-app`:** M0–M5b and model names, pushed at the end of 15 Nov (if the push failed, the user pushes).
- **Tomer works:** the first model trained from zero on the phone (one 3-min recording) scored
  **100%** on the guided check right after (REPORT_NOTES.md, E-phone). The user is happy with it.
  On data from 30–80 min earlier it scored 76–89%: the signal drifts, so a later calibration adds
  recordings. Guy was skipped for now (the user's call).
- **APK `~/ProjectYad-0.6.1.apk`:** like 0.6.0 (Record / Quick calibration → train from zero on all of
  the user's phone recordings; import/delete recordings; every model exported to Downloads as JSON),
  plus an optional **name for the new model**: saved as `<user>-<label>@yyyyMMdd_HHmmss`, shown as
  "tomer · label · 15/11 18:40" (`core/ModelNames.kt`). The time keeps names unique; saving never
  overwrites. 0.6.1 is not yet on the phone.
- Session files: `tools/out/checks/` (git-ignored: personal data).

## Next steps

Hardware, whenever there's time: install 0.6.1. Tomer does a **Quick calibration** (1 min,
optionally named) and a guided check. Guy does a full **Record** (3 min), then a guided check. Ask
for the `rec_*.bin`, the check files and the `yad_model_*.json`, plus the training time.

The user's goals for the next sessions (given 15 Nov): **lower latency to an accurate reading, a
more robust system, better UI/UX.** The plan below was proposed at the end of 15 Nov and still
needs the user's OK.

### 0. Measuring stick: DONE (`tools/bench.py`, 15 Nov), first results in REPORT_NOTES E-latency

- **Best so far:** classic TD features (no CNN) + "3 agree, every 200 ms, ≥ 0.7". Median latency
  1.0 s vs 1.8 s today, 0 wrong moves, 100% accuracy, on Tomer's 8 files.
- **Next on the bench:**
  - rules with a dwell or hysteresis, or a stricter threshold for leaving REST (half the wrong
    moves are transitions);
  - training on onset windows too;
  - a 100 ms window for TD;
  - Guy's data.
- **Then on the phone:** TD features in Kotlin with golden tests (bench.td_features is the
  reference), the rule as a new DecisionPolicy, and a sliding hop.

### 0 (original plan text)

`tools/bench.py`: replays every labelled phone recording (`rec_*`, `yad_calib_*`, `yad_check_*`
in `tools/out/checks/`) through a pipeline configuration and reports, per gesture:
- **accuracy**;
- **latency**: from gesture onset to the first correct command;
- **wrong commands per minute**: false moves are the safety cost of speed.

Validation is always across recordings: train on some, test on another. Every idea below is scored
this way before it reaches the phone.

### 1. Latency (biggest lever: the decision rule, not the CNN)

Today: 200 ms windows, non-overlapping, 6 averaged, threshold 0.7. That gives one decision per
1.2 s, and a gesture change often costs 1.2–2.4 s.
- **Sliding windows:** a new 200 ms window every 50–100 ms, instead of every 200 ms.
- **A faster decision rule:** e.g. "k consecutive windows agree with p > θ", with hysteresis.
  Tune k and θ on the bench: the target is ~300–500 ms with no more wrong commands than today.
- **CNN speed** (94.9 ms/window; a 50 ms hop needs < 50 ms):
  - make no-XNNPACK × 4 threads the default engine;
  - run 2 interpreters in parallel;
  - try int8 / dynamic-range quantization, checking accuracy on the bench;
  - try features from an earlier CNN layer.
- **Alternative: no CNN.** Classic EMG features (MAV, RMS, WL, ZC, SSC per channel) feeding the
  same MLP take microseconds, and work with 100 ms windows. Now that training is per user on phone
  data, they may match the CNN. Compare on the bench.
- **Measure the real chain on the phone:** packet arrival → window → decision → BLE write done.
  Add timestamps and show them in Stats.

### 2. Robustness

- **Drift:** keep adding short calibrations (already possible). Try weighting the newest
  recordings higher, or keeping only the last N.
- **Augmentation when training:** rotate channels ±1 (armband shift), scale gain ±20%, add noise.
  This makes models tolerant of placement; check it on the bench.
- **LEFT ↔ RIGHT confusion:** more data for those two, or pick more distinct gestures.
- **Signal-quality guard:**
  - flag a flat, saturated or noisy channel, packet loss, or low battery (battery is in WUA2 byte 72);
  - pause commands while the signal is bad.
- **Keep the app alive:**
  - run as an Android foreground service, so the screen can be off;
  - reconnect to the armband and the hand automatically;
  - train in the background with a notification.
- **Learn from mistakes:** a "that was wrong" button logs the window; it becomes training data.

### 3. UI/UX (the user said "improve as we go")

- Sketch, one screen in order: **Person → Armband (✓ 500/s) → Train (Record / Quick
  calibration) → Check → Use** (big gesture display, big STOP, hand switch, manual buttons).
  Everything else goes under a closed "Advanced" section.
- Profiles for new people without a bundled PC model. A model manager: rename, delete, see what
  each model was trained on.
- Recording prompts that don't need eyes on the screen: vibration or sound at each phase change,
  and a big countdown.
- Show training progress with a time estimate.
- The bundled `train_*.bin` (18 MB in the APK) are no longer used by the app: drop them from the APK.

### Suggested order

bench → sliding window + decision rule (on the bench, then a short hardware check) → CNN engine
default and quantization → robustness items → UI pass.

## Environment (this dev machine)

- Repo: `/home/guyka/.workspace/Project_Yad`. Python tools venv: `files_for_phone/tools/.venv`
  (Python 3.10.11, TF 2.12, sklearn 1.7.2, numpy 1.23.5). The Xeon here breaks numpy 1.23.5's
  OpenBLAS; `tools/_env.py` forces a safe kernel — import it before numpy.
- Android: SDK `~/Android/Sdk`, JDK `~/.local/share/mise/installs/java/corretto-21.0.13.0.11`,
  `cd files_for_phone/android && ./gradlew :core:test :app:assembleDebug`. No emulator (no KVM):
  on-device checks only on the user's phone.
- Not in git, regenerate with the tools: `models/feature_extractor.tflite` (`export_cnn.py`, ~25 min),
  `models/train_*.bin` (`export_train_features.py`). Check files from 14 Nov: `tools/out/checks/`.
- Known risks: the armband WiFi uses MindRove's default password; the hand accepts BLE commands
  from any device; the Pi's own `.git` is corrupted (irrelevant now).
