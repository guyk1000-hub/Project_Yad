# files_for_phone/tools — PC-side tooling for the phone app

Nothing here is imported by the Pi code, and nothing here modifies `training_data/` or `files_for_pi/`.

## Setup (once)

Python 3.10 with the same package versions as the Pi (`venvPI`):

```bash
python3.10 -m venv files_for_phone/tools/.venv
files_for_phone/tools/.venv/bin/pip install -r files_for_phone/tools/requirements.txt
```

## Scripts (run from the repo root)

| Script | What it does |
|---|---|
| `export_head.py tomer guy` / `--all` | Exports `scaler_<name>.pkl` + `mlp_model_<name>.pkl` to `files_for_phone/tools/out/models/head_<name>.json` for the phone. Before writing, it checks that the JSON gives the same predictions as sklearn on the recorded windows. |
| `export_cnn.py` | Converts the CNN (`og_feature_extraction.h5`, up to `dense_8`) to `out/models/feature_extractor.tflite`. It keeps the file only if the TFLite features and every exported head's predictions match Keras on all recordings. Run it after `export_head.py`. It takes ~25 min, because it runs single-threaded to mimic one phone core. |
| `make_golden.py --model tomer` | Writes test vectors to `files_for_phone/tools/golden/` (committed). The Kotlin tests must reproduce them. |

Example:
```bash
files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_head.py --all
```

The first run computes CNN features for every recording (~2–3 min per recording on 4 cores) and caches them in `files_for_phone/tools/out/cache/`.

## Two things worth knowing

- **Two feature versions.** The 2 extra features (`asym`, `mag`) were computed two different ways at different times, and models were trained with both:
  - `mlp_model_tomer` uses RMS over **time**.
  - `tomer_long`, `guy` and `tomer_250` use RMS over **channels**, which is what the Pi computes today.

  `export_head.py` detects which version each model was trained with from its scaler, and records it in the JSON as `input.scalar_rms_over`. Feeding a model the wrong version costs accuracy: `mlp_model_tomer` drops from 99.7% to 86.7% on its own recording.
- **Broken BLAS on some Intel CPUs.** numpy 1.23.5's bundled OpenBLAS returns wrong float64 matrix products on Sapphire Rapids (e.g. Xeon 8488C). `_env.py` forces a safe kernel and runs a self-test, and every script imports it before numpy.
