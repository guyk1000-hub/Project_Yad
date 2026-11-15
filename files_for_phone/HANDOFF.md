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

- **On GitHub:** M0–M5 (up to `6e12775`). **Local, not pushed:** M5b + model names (see git log).
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

1. Next hardware time: install 0.6.1, Tomer does a **Quick calibration** (1 min, optionally named)
   and a guided check. Send `rec_tomer_*.bin`, the check file and the `yad_model_*.json`.
2. Guy: a full **Record** (3 min) on his own profile, then a guided check.
3. Then, in rough priority: a simpler UI (sketched in the 15 Nov chat; the user said "improve as we
   go"); new users without a bundled PC model (profiles); lower latency (sliding window; the
   94.9 ms no-XNNPACK engine as default); learning from "that was wrong" corrections. The bundled
   `train_*.bin` (18 MB in the APK) are no longer used by the app.

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
