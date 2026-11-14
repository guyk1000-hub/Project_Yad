"""
Convert the NaviFlame CNN feature extractor (og_feature_extraction.h5, up to dense_8)
to TensorFlow Lite for the phone, and check it against Keras before keeping it.

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_cnn.py

Writes files_for_phone/models/feature_extractor.tflite (+ .json with metadata
and the parity report). The .h5 is not modified.

The checks run on every recorded window of every dataset that has an exported head:
  1. dense_8 features: TFLite vs Keras
  2. what matters downstream: each head's class probabilities with TFLite vs Keras features
"""
import argparse
import datetime
import json
import os
import time

import _env  # noqa: F401  must precede numpy
import numpy as np
import tensorflow as tf

import export_head
import yad_ref

INPUT_SHAPE = (1, 8, yad_ref.MODEL_INPUT_LEN, 1)  # one window at a time; features don't depend on batching


def convert() -> bytes:
    extractor = yad_ref.load_feature_extractor()

    # training=False: MyMagnWarping/MyScaling are identity layers and GaussianNoise is
    # inactive at inference, so they trace to nothing.
    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name="window")])
    def serve(window):
        return {yad_ref.FEATURE_LAYER: extractor(window, training=False)}

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], extractor)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # no Flex ops: smaller runtime on the phone
    return converter.convert()


def tflite_features(model_bytes: bytes, X: np.ndarray):
    interp = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    feats = np.empty((len(X), out["shape"][-1]), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i:i + 1])
        interp.invoke()
        feats[i] = interp.get_tensor(out["index"])[0]
    ms_per_window = 1000 * (time.perf_counter() - t0) / len(X)
    return feats, ms_per_window


def check(model_bytes: bytes, heads_dir: str) -> dict:
    report = {"datasets": {}}
    for path in sorted(os.listdir(heads_dir)):
        if not (path.startswith("head_") and path.endswith(".json")):
            continue
        name = path[len("head_"):-len(".json")]
        with open(os.path.join(heads_dir, path)) as f:
            head = json.load(f)
        X, _ = yad_ref.load_dataset(name)
        keras_f = yad_ref.cnn_features(X)
        lite_f, ms = tflite_features(model_bytes, X)
        rms_over = head["input"]["scalar_rms_over"]
        p_keras = export_head.forward(head, yad_ref.head_inputs(X, keras_f, rms_over))
        p_lite = export_head.forward(head, yad_ref.head_inputs(X, lite_f, rms_over))
        flips = np.flatnonzero(p_keras.argmax(1) != p_lite.argmax(1))
        report["datasets"][name] = {
            "windows": int(len(X)),
            "feature_max_rel_err": float(np.abs(lite_f - keras_f).max() / np.abs(keras_f).max()),
            "prob_max_abs_diff": float(np.abs(p_lite - p_keras).max()),
            "argmax_agreement": float(1 - len(flips) / len(X)),
            "flipped_windows": flips.tolist()[:20],
            "x86_ms_per_window": round(ms, 2),
        }
        r = report["datasets"][name]
        print(f"[{name}] windows={r['windows']} feature rel err={r['feature_max_rel_err']:.2e} "
              f"max|Δp|={r['prob_max_abs_diff']:.2e} argmax agree={r['argmax_agreement']:.5f} "
              f"({len(flips)} flips) {ms:.1f} ms/window on this PC")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=yad_ref.PHONE_MODELS_DIR)
    args = ap.parse_args()

    model_bytes = convert()
    print(f"converted: {len(model_bytes) / 1e6:.1f} MB")
    report = check(model_bytes, args.out)
    if not report["datasets"]:
        raise SystemExit("no exported heads found — run export_head.py first")

    ok = all(r["feature_max_rel_err"] < 1e-4 and r["argmax_agreement"] == 1.0 for r in report["datasets"].values())
    if not ok:
        print("PARITY FAILED — not written")
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "feature_extractor.tflite")
    with open(path, "wb") as f:
        f.write(model_bytes)
    meta = {
        "source_file": os.path.basename(yad_ref.feature_extractor_path()),
        "source_sha256": yad_ref.sha256(yad_ref.feature_extractor_path()),
        "tflite_sha256": yad_ref.sha256(path),
        "tensorflow_version": tf.__version__,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "input": {"shape": list(INPUT_SHAPE), "dtype": "float32",
                  "layout": "[batch=1, channel=8, sample=100, 1], filtered µV as recorded by record.py"},
        "output": {"name": yad_ref.FEATURE_LAYER, "shape": [1, 1024], "dtype": "float32"},
        "parity_check": report,
    }
    with open(path[:-len(".tflite")] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {os.path.relpath(path, yad_ref.REPO_ROOT)} ({os.path.getsize(path) / 1e6:.1f} MB) + metadata")


if __name__ == "__main__":
    main()
