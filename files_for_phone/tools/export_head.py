"""
Export a user's scaler + MLP (sklearn .pkl) to a plain JSON file the phone can run.

The JSON holds only numbers: scaler mean/scale, layer weights/biases, activations
and class ids. The .pkl files are not modified.

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_head.py tomer guy tomer_long
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_head.py --all

Each export is checked before it is written: the JSON forward pass, computed with
plain numpy, must match sklearn's predict_proba on the user's recorded windows.
"""
import argparse
import datetime
import glob
import json
import os

import _env  # noqa: F401  must precede numpy
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import yad_ref

FORMAT = "yad-head"
FORMAT_VERSION = 1
SUPPORTED_HIDDEN = {"tanh", "relu", "logistic", "identity"}


def head_to_dict(model_name: str, rms_over: str) -> dict:
    scaler, mlp = yad_ref.load_head(model_name)
    if not isinstance(scaler, StandardScaler) or not isinstance(mlp, MLPClassifier):
        raise TypeError(f"{model_name}: expected StandardScaler + MLPClassifier, got "
                        f"{type(scaler).__name__} + {type(mlp).__name__}")
    if mlp.activation not in SUPPORTED_HIDDEN or mlp.out_activation_ != "softmax":
        raise ValueError(f"{model_name}: unsupported activations {mlp.activation}/{mlp.out_activation_}")
    if scaler.n_features_in_ != mlp.n_features_in_:
        raise ValueError(f"{model_name}: scaler expects {scaler.n_features_in_} features, MLP {mlp.n_features_in_}")

    scaler_path, mlp_path = yad_ref.head_paths(model_name)
    names = yad_ref.gesture_map()
    classes = [int(c) for c in mlp.classes_]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "scaler_file": os.path.basename(scaler_path),
            "scaler_sha256": yad_ref.sha256(scaler_path),
            "mlp_file": os.path.basename(mlp_path),
            "mlp_sha256": yad_ref.sha256(mlp_path),
            "feature_extractor_file": os.path.basename(yad_ref.feature_extractor_path()),
            "feature_extractor_sha256": yad_ref.sha256(yad_ref.feature_extractor_path()),
            "sklearn_version": sklearn.__version__,
        },
        "input": {
            "n_features": int(scaler.n_features_in_),
            # Order matters: [dense_8 (1024), asym, mag].
            "layout": [{"name": yad_ref.FEATURE_LAYER, "size": int(scaler.n_features_in_) - 2},
                       {"name": "asym", "size": 1},
                       {"name": "mag", "size": 1}],
            # How asym/mag compute RMS over an [8, L] window — models were trained with both
            # (see yad_ref.RMS_OVER). "channels" = axis=1, what the Pi computes today.
            "scalar_rms_over": rms_over,
        },
        "classes": classes,
        "class_names": {str(c): names.get(c, str(c)) for c in classes},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "hidden_activation": mlp.activation,
        "output_activation": mlp.out_activation_,
        # weights[i] is [n_in][n_out]: out = in @ weights + bias
        "layers": [{"weights": W.tolist(), "bias": b.tolist()}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }


_ACT = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0),
    "logistic": lambda x: 1 / (1 + np.exp(-x)),
    "identity": lambda x: x,
}


def forward(head: dict, x: np.ndarray) -> np.ndarray:
    """Forward pass from the JSON alone (no sklearn) — this is what the phone implements."""
    a = (x.astype(np.float64) - np.asarray(head["scaler"]["mean"])) / np.asarray(head["scaler"]["scale"])
    layers = head["layers"]
    for i, layer in enumerate(layers):
        a = a @ np.asarray(layer["weights"]) + np.asarray(layer["bias"])
        if i < len(layers) - 1:
            a = _ACT[head["hidden_activation"]](a)
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def check(model_name: str, head: dict) -> dict:
    """Compare the JSON forward pass with sklearn on the user's recorded windows."""
    scaler, mlp = yad_ref.load_head(model_name)
    X, y = yad_ref.load_dataset(model_name)
    cnn = yad_ref.cnn_features(X)
    feats = yad_ref.head_inputs(X, cnn, head["input"]["scalar_rms_over"])
    p_ref = mlp.predict_proba(scaler.transform(feats))
    p_json = forward(head, feats)
    # What the Pi's inference code would score with this head (it always uses "channels").
    p_pi = mlp.predict_proba(scaler.transform(yad_ref.head_inputs(X, cnn, "channels")))
    return {
        "windows": int(len(X)),
        "max_abs_prob_diff": float(np.abs(p_ref - p_json).max()),
        "argmax_agreement": float((p_ref.argmax(1) == p_json.argmax(1)).mean()),
        # Inflated: these are the windows the model was trained on (random split of overlapping windows).
        "train_set_accuracy": float((mlp.classes_[p_ref.argmax(1)] == y).mean()),
        "train_set_accuracy_pi_features": float((mlp.classes_[p_pi.argmax(1)] == y).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model suffixes, e.g. tomer guy")
    ap.add_argument("--all", action="store_true", help="export every mlp_model_*.pkl in training_data/models")
    ap.add_argument("--rms-over", choices=yad_ref.RMS_OVER,
                    help="feature version; detected from the recording when one exists")
    ap.add_argument("--out", default=os.path.join(yad_ref.OUT_DIR, "models"))
    args = ap.parse_args()

    names = args.models
    if args.all:
        names = sorted(os.path.basename(p)[len("mlp_model_"):-len(".pkl")]
                       for p in glob.glob(os.path.join(yad_ref.MODELS_DIR, "mlp_model_*.pkl")))
    if not names:
        ap.error("give model names or --all")

    os.makedirs(args.out, exist_ok=True)
    failed = False
    for name in names:
        has_data = os.path.exists(yad_ref.dataset_path(name))
        try:
            rms_over = args.rms_over or (yad_ref.detect_rms_over(name) if has_data else None)
        except ValueError as e:
            print(f"[{name}] {e} — not written")
            failed = True
            continue
        if rms_over is None:
            print(f"[{name}] no recording to detect the feature version from; pass --rms-over — not written")
            failed = True
            continue
        head = head_to_dict(name, rms_over)
        report = check(name, json.loads(json.dumps(head))) if has_data else None
        # sklearn computes in float32 here (float32 inputs), the JSON forward in float64: ~1e-6 apart.
        ok = report is None or (report["argmax_agreement"] == 1.0 and report["max_abs_prob_diff"] < 1e-5)
        if report:
            print(f"[{name}] rms_over={rms_over} windows={report['windows']} max|Δp|={report['max_abs_prob_diff']:.2e} "
                  f"argmax agree={report['argmax_agreement']:.4f} train-set acc={report['train_set_accuracy']:.4f} "
                  f"(with Pi features: {report['train_set_accuracy_pi_features']:.4f})")
        else:
            print(f"[{name}] rms_over={rms_over} (given); no recording, exported without data check")
        if not ok:
            failed = True
            print(f"[{name}] PARITY FAILED — not written")
            continue
        head["parity_check"] = report
        path = os.path.join(args.out, f"head_{name}.json")
        with open(path, "w") as f:
            json.dump(head, f)
        print(f"[{name}] wrote {os.path.relpath(path, yad_ref.REPO_ROOT)} ({os.path.getsize(path) // 1024} KB)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
