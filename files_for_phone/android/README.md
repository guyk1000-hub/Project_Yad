# files_for_phone/android — the phone app (Kotlin)

| Module | What it is | Status |
|---|---|---|
| `core/` | Pure Kotlin, no Android dependency: filters, windowing, `asym`/`mag` (both versions), the MLP head, the Pi's decision policy, and the Haifa3D command bytes | M2 ✅ |
| `app/` | The Android app. M3 replays recordings through the real pipeline (TFLite CNN + head + decision policy) into a simulated hand. It has STOP, manual buttons, sensitivity sliders, stats, a CNN benchmark, and an on-device parity check at startup | M3 🔨 |

## Build the app

Needs the Android SDK (`local.properties`: `sdk.dir=...`) and the TFLite model, which isn't committed. Generate it first:

```bash
files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_cnn.py      # once, ~25 min
files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_replay.py tomer guy tomer_long
cd files_for_phone/android && ./gradlew :app:assembleDebug
# -> app/build/outputs/apk/debug/app-debug.apk
```

## Run the tests

Needs JDK 17 or 21 (`JAVA_HOME`). The Gradle wrapper downloads everything else.

```bash
cd files_for_phone/android
./gradlew :core:test
```

The tests compare `core` with the reference vectors in `../tools/golden/` and the exported models in `../models/`. Both are produced from the Pi's own Python code (see `../tools/README.md`).

| Stage | Result |
|---|---|
| Filters | bit-identical to `training_data/utils.py` (difference 0.0) |
| Windows | equal to the filtered stream cast to float32, like the Pi |
| `asym` / `mag` | within 1.5e-7 relative, both feature versions |
| MLP head | within 6.6e-11 of sklearn `predict_proba` |
| Decisions | identical, including threshold boundary cases |
| Hand command bytes | identical to `files_for_pi/protocol.py` |

The tests themselves are checked too: deliberately planted bugs (a filter Q off by 0.1, a shifted threshold, `>` instead of `>=`) make them fail.
