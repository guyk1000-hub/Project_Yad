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

- **On GitHub:** M0–M5 (up to `6e12775`).
- **Session B (15 Nov) done for Tomer** (see REPORT_NOTES.md E-calib, E-calib2): calibrated
  models reached 90–94% on separate guided checks (from 74%). On the PC, the old PC recording made
  every variant worse, and training only on the two phone recordings was best (99 / 91 / 98%).
  LEFT ↔ RIGHT is the weakest pair. Guy was skipped for now (the user's call).
- **Committed locally, not pushed:** M5b (APK `~/ProjectYad-0.6.0.apk`, 0.6.0-m5b). Record (3 × 10 s
  per gesture) or quick calibration (1 take), then train from zero on all of the user's phone
  recordings; import/delete recordings; phone models `<user>-phone@yyyyMMdd_HHmmss`, never
  overwritten; every phone model also exported to Downloads as `yad_model_<name>.json`.
  `analyze_check.py` reads those (`--head`) and can redo a calibration on the PC (`--calib`).
  **Not yet tested on hardware.**
- Session files: `tools/out/checks/` (git-ignored: personal data).

## Next steps

1. **Hardware session C (Tomer):** install 0.6.0. Import the two `yad_calib_tomer_20261115_*.bin`
   (→ `tomer-phone@…`), Start armband, guided check. Then **Record** (3 takes), guided check, wait
   ~5 min, guided check again. Ask for: screenshots, the training-time message, `rec_tomer_*.bin`,
   `yad_check_tomer-phone@*.bin`, `yad_model_tomer-phone@*.json`.
2. Analyse (`analyze_check.py --head …`). If LEFT stays weak: more LEFT data, or check the LEFT/RIGHT gestures themselves.
3. Then, in rough priority: new users without a bundled PC model (profiles), a simpler main
   screen with settings out of the way, lower latency (sliding window; the 94.9 ms no-XNNPACK
   engine as default), learning from "that was wrong" corrections. The bundled `train_*.bin`
   (18 MB in the APK) are no longer used by the app.

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
