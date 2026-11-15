# Handoff — Project Yad phone app (state on 2026-11-15)

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

- **Done (on GitHub up to `e775873`):** M0–M4. The phone reproduces the Pi pipeline exactly
  (golden tests), runs live from the armband (own WUA2 packet decoder), drives the real hand
  over BLE (off by default), STOP + manual overrides, sliders, benchmark, guided 35 s check.
  `fine_tune.py` now reports held-out-take accuracy (user-approved).
- **Committed locally, NOT pushed:** `d17884e` M5, per-session calibration on the phone
  (record 10 s/gesture → retrain head on the phone with the original recording → saved as a
  model version). Built as `~/ProjectYad-m5.apk` (0.5.0-m5). **Not yet tested on hardware.**
- **Key finding:** models don't transfer across sessions (~47% cross-session, live 46%/74%);
  old data + a few seconds of the current session restores 95–99% offline. The Kotlin trainer
  matches sklearn offline (guy 83% vs 77%, tomer 96% vs 99%).

## Next steps

1. **Hardware session B** (user, ~10 min/person): install `ProjectYad-m5.apk`; armband on (only
   the phone on its WiFi; power-cycle if it streams elsewhere); pick user model → Armband (live)
   → Start armband (500 samples/s) → **Calibrate** (10 s/gesture, then 3–5 min training) →
   Start armband again → **guided check** → try live with Real hand off, then on.
   Ask for: the check screenshot, `Downloads/yad_calib_*.bin`, `yad_check_*.bin`, and how long training took.
2. Analyse with `tools/analyze_check.py` (runs the data through the Pi code, compares with the
   phone). If Guy is still weak: longer calibration, more iterations, smaller hop.
3. Then, in rough priority: multi-session training (keep calibration data per user and train on
   all sessions), make the app usable (profiles, simple main screen, settings out of the way),
   lower latency (sliding window; the 94.9 ms no-XNNPACK engine as default), learning from
   "that was wrong" corrections.

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
