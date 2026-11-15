"""
Generate golden test vectors: input -> expected output for each pipeline stage,
produced by the Pi's own code (see yad_ref.py). The Kotlin tests must reproduce them.

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/make_golden.py --model tomer

Writes files_for_phone/tools/golden/*.json (small, meant to be committed):
    filters.json          synthetic raw µV stream -> Pi filter chain, window by window
    scalar_features.json  recorded windows -> asym, mag
    head_<model>.json     head inputs (1026) -> class probabilities
    cnn_features.json     recorded windows -> dense_8 (1024), for the on-device TFLite test
    decision.json         per-window probabilities -> Pi decisions (batch 6, threshold 0.7)
    hand_commands.json    gesture -> BLE characteristic + payload bytes
"""
import argparse
import json
import os

import _env  # noqa: F401  must precede numpy
import numpy as np

import yad_ref

WINDOWS_PER_CLASS = 2
DECISION_WINDOWS = 120


def _write(name: str, obj: dict):
    os.makedirs(yad_ref.GOLDEN_DIR, exist_ok=True)
    path = os.path.join(yad_ref.GOLDEN_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"wrote {os.path.relpath(path, yad_ref.REPO_ROOT)} ({os.path.getsize(path) // 1024} KB)")


def _f32(a: np.ndarray):
    """float32 values serialised exactly (float32 -> float64 is lossless)."""
    return np.asarray(a, dtype=np.float32).astype(np.float64).tolist()


def golden_filters(seed: int = 0):
    """
    Synthetic raw signal: per-channel DC offset, drift, 50 Hz mains, EMG-band tones, noise
    and a step. The recordings are stored *after* filtering, so they can't test this stage.
    """
    rng = np.random.default_rng(seed)
    fs, n_windows, L = yad_ref.SAMPLING_RATE, 10, yad_ref.MODEL_INPUT_LEN
    t = np.arange(n_windows * L) / fs
    raw = np.empty((len(t), 8))
    for ch in range(8):
        raw[:, ch] = (rng.uniform(-3000, 3000)
                      + 400 * np.sin(2 * np.pi * 0.3 * t)
                      + 200 * np.sin(2 * np.pi * 50 * t + ch)
                      + 60 * np.sin(2 * np.pi * (20 + 15 * ch) * t)
                      + 40 * np.sin(2 * np.pi * 150 * t)
                      + rng.normal(0, 20, len(t))
                      + np.where(t > 1.0, 500.0, 0.0))

    filters = yad_ref.make_pi_filters()
    out = np.empty_like(raw)
    # Same as the Pi: consecutive non-overlapping windows, filter state carried across windows.
    for w in range(n_windows):
        out[w * L:(w + 1) * L] = yad_ref.pi_preprocess_window(raw[w * L:(w + 1) * L].copy(), filters)

    fresh = yad_ref.make_pi_filters()
    _write("filters.json", {
        "sampling_rate": fs,
        "window_len": L,
        "filters": [{"type": f.type.name, "Fc": f.Fc, "Q": f.Q, "peak_gain_db": f.peakGain,
                     "a0": f.a0, "a1": f.a1, "a2": f.a2, "b1": f.b1, "b2": f.b2} for f in fresh],
        "raw": raw.tolist(),          # [T][8] µV, float64
        "filtered": out.tolist(),     # [T][8], float64 (the Pi casts to float32 only after filtering)
    })


def golden_model_stages(model_name: str):
    X, y = yad_ref.load_dataset(model_name)
    scaler, mlp = yad_ref.load_head(model_name)

    # A few windows of each class, taken from the middle of each class's block.
    idx = []
    for c in np.unique(y):
        where = np.flatnonzero(y == c)
        idx += where[np.linspace(len(where) // 4, 3 * len(where) // 4, WINDOWS_PER_CLASS, dtype=int)].tolist()
    Xs = X[idx]
    rms_over = yad_ref.detect_rms_over(model_name)

    versions = {}
    for mode in yad_ref.RMS_OVER:
        asym, mag = yad_ref.scalar_features(Xs, mode)
        versions[mode] = {"asym": _f32(asym[:, 0]), "mag": _f32(mag[:, 0])}
    _write("scalar_features.json", {
        "note": "Two versions exist (yad_ref.RMS_OVER). 'channels' = axis=1 of [8, L], what the Pi computes; "
                "'time' = axis=2. Each exported head names the one it needs in input.scalar_rms_over.",
        "windows": _f32(Xs[..., 0]),   # [K][8][100]
        "versions": versions,
    })

    cnn_all = yad_ref.cnn_features(X)
    _write("cnn_features.json", {
        "model": "og_feature_extraction.h5",
        "model_sha256": yad_ref.sha256(yad_ref.feature_extractor_path()),
        "layer": yad_ref.FEATURE_LAYER,
        "windows": _f32(Xs[..., 0]),
        "features": _f32(cnn_all[idx]),  # [K][1024]
    })

    feats = yad_ref.head_inputs(Xs, cnn_all[idx], rms_over)
    _write(f"head_{model_name}.json", {
        "model_name": model_name,
        "scalar_rms_over": rms_over,
        "labels": y[idx].tolist(),
        "inputs": _f32(feats),                                       # [K][1026]
        "probs": mlp.predict_proba(scaler.transform(feats)).tolist(),  # [K][n_classes]
    })

    # Decision policy over a realistic probability stream: consecutive windows that cross
    # class boundaries, so some groups fall below the threshold.
    starts = np.flatnonzero(np.diff(y) != 0) + 1
    s = int(starts[0]) - DECISION_WINDOWS // 2 if len(starts) else 0
    seq = np.arange(max(s, 0), max(s, 0) + DECISION_WINDOWS)
    probs = mlp.predict_proba(scaler.transform(yad_ref.head_inputs(X[seq], cnn_all[seq], rms_over)))
    # Recorded windows are almost always confident, so add a synthetic stream that mixes
    # confident and uncertain groups to exercise the threshold (and a trailing partial group).
    rng = np.random.default_rng(1)
    synth = []
    for g in range(12):
        alpha = np.full(len(mlp.classes_), 0.3)
        if g % 3 != 2:
            alpha[g % len(alpha)] = 8.0
        synth += list(rng.dirichlet(alpha, size=yad_ref.BATCH_SIZE))
    synth += list(rng.dirichlet(np.ones(len(mlp.classes_)), size=yad_ref.BATCH_SIZE - 1))

    def case(name, seq_probs, threshold=yad_ref.PREDICTION_THRESHOLD):
        emitted = yad_ref.pi_decisions(list(seq_probs), threshold=threshold)
        return {"name": name,
                "threshold": threshold,
                "per_window_probs": np.asarray(seq_probs).tolist(),
                "decisions": [{"after_window": i, "gesture": top, "avg": avg.tolist()} for i, top, avg in emitted]}

    # Groups whose top probability sits on and around the threshold, so a shifted threshold
    # changes the outcome. With the Pi's 0.7, no average of 6 doubles equals 0.7 exactly, so
    # ">" vs ">=" is tested with 0.5 (six 0.5s average to exactly 0.5); that matters once
    # the app's sensitivity slider allows other thresholds.
    n_cls = len(mlp.classes_)

    def groups(tops):
        rows = []
        for g, top in enumerate(tops):
            row = np.full(n_cls, (1 - top) / (n_cls - 1))
            row[g % n_cls] = top
            rows += [row.copy() for _ in range(yad_ref.BATCH_SIZE)]
        return rows

    # The phone's replay (every 10th recorded window, see export_replay.py) with its true labels.
    rep = np.arange(0, len(X), 10)
    replay_probs = mlp.predict_proba(scaler.transform(yad_ref.head_inputs(X[rep], cnn_all[rep], rms_over)))
    replay_case = case(f"replay_{model_name}_labelled", replay_probs)
    replay_case["labels"] = y[rep].tolist()

    _write("decision.json", {
        "batch_size": yad_ref.BATCH_SIZE,
        "threshold": yad_ref.PREDICTION_THRESHOLD,
        "cases": [case(f"recorded_{model_name}", probs),
                  case("synthetic_mixed_confidence", synth),
                  case("threshold_near_0.7", groups([0.69, 0.70, 0.701, 0.72, 0.75, 0.80])),
                  case("threshold_exactly_0.5", groups([0.49, 0.5, 0.51]), threshold=0.5),
                  replay_case],
    })


def golden_hand_commands():
    import protocol as prt
    gm = yad_ref.gesture_map()
    # From files_for_pi/basic_commands_pi.py: open/close/left/right are written to DIRECT_UUID;
    # anything else (rest) sends nothing.
    sent = {"open": "open", "close": "close", "left": "left", "right": "right"}
    _write("hand_commands.json", {
        "characteristic_uuid": prt.DIRECT_UUID,
        "hand_address": "B8:D6:1A:40:EF:D6",
        "gestures": {str(k): {"name": v,
                              "payload": list(prt.commands[sent[v]]) if v in sent else None}
                     for k, v in gm.items()},
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tomer")
    args = ap.parse_args()
    golden_filters()
    golden_model_stages(args.model)
    golden_hand_commands()


if __name__ == "__main__":
    main()
