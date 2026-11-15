"""
Analyse a guided check recorded on the phone (Downloads/yad_check_<model>_<time>.bin).

Runs the same raw samples through the Pi's own pipeline (filters, non-overlapping windows,
Keras CNN, sklearn head) and answers:
  1. Does the phone compute what the Pi code computes on this data?  (phone vs Python probs)
  2. Per-gesture accuracy with the Pi code: is the problem the phone or the signal?
  3. Would another model, or the armband rotated / mirrored by some channels, do better?

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/analyze_check.py ~/yad_check_tomer_*.bin
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/analyze_check.py --selftest FILE   # format only

A check made with a model calibrated on the phone (name "<user>@<time>") needs that model's
Downloads/yad_model_<user>@<time>.json next to the check file (or --head). With --calib
Downloads/yad_calib_<user>_<time>.bin it also retrains the same way on the PC with sklearn and
scores that on the check: is a weak result the phone's trainer or the data?
"""
import argparse
import json
import os
import struct

import _env  # noqa: F401  must precede numpy
import numpy as np

import yad_ref

GESTURES = {0: "REST", 1: "CLOSE", 2: "OPEN", 3: "RIGHT", 4: "LEFT"}


def read_check(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"YADC":
        raise ValueError(f"{path} is not a guided-check file")
    version, header_len = struct.unpack_from("<ii", data, 4)
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    header = json.loads(data[12:12 + header_len].decode("utf-8"))
    pos = 12 + header_len
    (n,) = struct.unpack_from("<i", data, pos); pos += 4
    labels = np.frombuffer(data, dtype=np.int8, count=n, offset=pos).astype(int); pos += n
    raw = np.frombuffer(data, dtype="<f8", count=n * 8, offset=pos).reshape(n, 8).copy(); pos += n * 64
    (nw,) = struct.unpack_from("<i", data, pos); pos += 4
    k = len(header["classes"])
    rec = np.dtype([("end", "<i4"), ("label", "i1"), ("pred", "i1"), ("probs", "<f8", (k,))])
    windows = np.frombuffer(data, dtype=rec, count=nw, offset=pos)
    pos += nw * rec.itemsize
    if pos != len(data):
        raise ValueError(f"{len(data) - pos} trailing bytes — format mismatch")
    # The phone cuts windows from the start of the stream, not the check: drop any window that
    # began before the check's first sample.
    windows = windows[windows["end"] >= header["window_length"] - 1]
    return {"header": header, "labels": labels, "raw": raw, "windows": windows}


class JsonHead:
    """A head exported as JSON (export_head.py, or calibrated on the phone): scaler + MLP, in numpy."""

    def __init__(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.mean, self.scale = np.array(d["scaler"]["mean"]), np.array(d["scaler"]["scale"])
        self.layers = [(np.array(l["weights"]), np.array(l["bias"])) for l in d["layers"]]
        self.act = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0), "identity": lambda x: x,
                    "logistic": lambda x: 1 / (1 + np.exp(-x))}[d["hidden_activation"]]
        self.classes_ = np.array(d["classes"])

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        a = (F - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            a = a @ W + b
            if i < len(self.layers) - 1:
                a = self.act(a)
        e = np.exp(a - a.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


class SklearnHead:
    """A head from the repo's pickles, with the same interface as JsonHead."""

    def __init__(self, scaler, mlp):
        self.scaler, self.mlp, self.classes_ = scaler, mlp, mlp.classes_

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        return self.mlp.predict_proba(self.scaler.transform(F))


def read_train(path: str):
    """Head inputs + labels from a file written by export_train_features.py."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"YADT", path
    _, n, d, _ = struct.unpack_from("<4i", data, 4)
    y = np.frombuffer(data, "<i4", n, 20).astype(int)
    F = np.frombuffer(data, "<f4", n * d, 20 + 8 * n).reshape(n, d).astype(np.float64)
    return F, y


# The phone's calibration recipe (core/Calibration.kt).
CALIB_HOP, CALIB_NEW_SHARE, CALIB_ITERATIONS = 25, 0.37, 300


def calibration_windows(c: dict) -> tuple:
    """Same windows as Calibration.windows: every HOP samples, inside one gesture, >= 1 s into it."""
    labels, L, sr = c["labels"], c["header"]["window_length"], c["header"]["sampling_rate"]
    filtered = pi_filter(c["raw"])
    phase_start = np.zeros(len(labels), dtype=int)
    for i in range(1, len(labels)):
        phase_start[i] = i if labels[i] != labels[i - 1] else phase_start[i - 1]
    starts = [s for s in range(0, len(labels) - L + 1, CALIB_HOP)
              if labels[s] >= 0 and labels[s + L - 1] == labels[s] and s >= phase_start[s] + sr]
    X = np.stack([filtered[s:s + L].T for s in starts])[..., None].astype(np.float32)
    return X, labels[starts]


def retrain_on_pc(calib_path: str, rms_over: str, F_check: np.ndarray, mask: np.ndarray, truth: np.ndarray):
    """The phone's calibration, redone with sklearn (300 iterations as on the phone, and 2000)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    c = read_check(calib_path)
    user = c["header"]["model"].split("@")[0]
    Xc, yc = calibration_windows(c)
    Fc = yad_ref.head_inputs(Xc, yad_ref.cnn_features(Xc), rms_over)
    Fo, yo = read_train(os.path.join(yad_ref.PHONE_MODELS_DIR, f"train_{user}.bin"))
    F, y = np.vstack([Fo, Fc]), np.concatenate([yo, yc])
    w = np.concatenate([np.ones(len(yo)), np.full(len(yc), CALIB_NEW_SHARE / (1 - CALIB_NEW_SHARE) * len(yo) / len(yc))])
    print(f"\nretrained on the PC (sklearn): {len(yo)} old + {len(yc)} calibration windows "
          f"({os.path.basename(calib_path)}, per gesture {np.bincount(yc, minlength=5).tolist()})")
    sc = StandardScaler().fit(F, sample_weight=w)
    for iters in (CALIB_ITERATIONS, 2000):
        m = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=iters, random_state=42, activation="tanh",
                          solver="lbfgs", alpha=0.01).fit(sc.transform(F), y, sample_weight=w)
        pred = m.classes_[m.predict_proba(sc.transform(F_check)).argmax(1)][mask]
        print(f"  {m.n_iter_:5d} iterations:        {per_gesture(pred, truth)}", flush=True)


def pi_filter(raw: np.ndarray) -> np.ndarray:
    """The Pi's filter chain over the whole stream. The filters are per-sample IIR with state, so
    one pass over everything equals the Pi's window-by-window processing."""
    return yad_ref.pi_preprocess_window(raw.copy(), yad_ref.make_pi_filters(yad_ref.SAMPLING_RATE))


def phone_aligned_windows(filtered: np.ndarray, ends: np.ndarray, L: int) -> np.ndarray:
    """Windows [N, 8, L, 1] ending at the phone's window ends, as float32 (like the Pi)."""
    return np.stack([filtered[e - L + 1:e + 1].T for e in ends])[..., None].astype(np.float32)


def scored_mask(check: dict) -> np.ndarray:
    """Same rule as the app: whole window inside one gesture, starting >= 1 s after the gesture began."""
    labels, w, h = check["labels"], check["windows"], check["header"]
    L, settle = h["window_length"], h["settle_samples"]
    phase_start = np.zeros(len(labels), dtype=int)
    for i in range(1, len(labels)):
        phase_start[i] = i if labels[i] != labels[i - 1] else phase_start[i - 1]
    ok = []
    for e, lab in zip(w["end"], w["label"]):
        s = e - L + 1
        ok.append(lab >= 0 and s >= 0 and labels[s] == lab and s >= phase_start[e] + settle)
    return np.array(ok)


def per_gesture(pred: np.ndarray, truth: np.ndarray) -> str:
    parts = []
    for g, name in GESTURES.items():
        m = truth == g
        if m.any():
            parts.append(f"{name} {100 * np.mean(pred[m] == g):3.0f}%")
    return "  ".join(parts) + f"   | all {100 * np.mean(pred == truth):.0f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--selftest", action="store_true", help="only check that the file parses")
    ap.add_argument("--head", help="JSON of a model calibrated on the phone (default: yad_model_<model>.json beside the check)")
    ap.add_argument("--calib", help="the calibration file (yad_calib_*.bin): also retrain on the PC and compare")
    args = ap.parse_args()

    for path in args.files:
        c = read_check(path)
        h, w = c["header"], c["windows"]
        print(f"\n=== {os.path.basename(path)}: model {h['model']} (RMS over {h['scalar_rms_over']}), "
              f"{len(c['labels'])} samples = {len(c['labels']) / h['sampling_rate']:.1f} s, {len(w)} windows")
        mask = scored_mask(c)
        truth = w["label"][mask].astype(int)
        print(f"phone (as recorded):      {per_gesture(w['pred'][mask].astype(int), truth)}")
        if args.selftest:
            continue

        L = h["window_length"]
        filtered = pi_filter(c["raw"])
        X = phone_aligned_windows(filtered, w["end"], L)
        cnn = yad_ref.cnn_features(X)

        # 1 + 2: the phone's model through the Pi code.
        F = yad_ref.head_inputs(X, cnn, h["scalar_rms_over"])
        if "@" in h["model"]:   # calibrated on the phone: its weights exist only in the exported JSON
            head_path = args.head or os.path.join(os.path.dirname(path), f"yad_model_{h['model']}.json")
            head = JsonHead(head_path) if os.path.exists(head_path) else None
            if head is None:
                print(f"(no {os.path.basename(head_path)}: can't rerun the phone's model; pass --head)")
        else:
            head = SklearnHead(*yad_ref.load_head(h["model"]))
        if head is not None:
            p = head.predict_proba(F)
            settled = np.arange(len(w)) >= 20   # skip the first 4 s: the Pi code's filters start cold here
            diff = np.abs(p - w["probs"])[settled].max()
            print(f"Pi code, same model:      {per_gesture(head.classes_[p.argmax(1)][mask], truth)}")
            print(f"phone vs Pi code: max |Δp| = {diff:.2e} after the first 4 s "
                  f"({'same computation' if diff < 1e-3 else 'DIFFERENT — phone bug, look closer'})")
        if args.calib:
            retrain_on_pc(args.calib, h["scalar_rms_over"], F, mask, truth)

        # 3: other models, and the armband rotated or mirrored. CNN features once per channel order.
        print("\nother models / channel orders (Pi code):")
        fx = yad_ref.load_feature_extractor()
        orders = {}
        for mirror in (False, True):
            for k in range(8):
                order = np.roll(np.arange(8)[::-1] if mirror else np.arange(8), k)
                Xo = X[:, order]
                orders[f"{'mirrored, ' if mirror else ''}rotated {k}"] = (Xo, fx.predict(Xo, verbose=0, batch_size=128))
        names = sorted(f[len("mlp_model_"):-4] for f in os.listdir(yad_ref.MODELS_DIR)
                       if f.startswith("mlp_model_") and f.endswith(".pkl"))
        for name in names:
            if name.endswith("_250"):
                continue   # trained on 250 Hz data; this check is 500 Hz
            try:
                rms_over = yad_ref.detect_rms_over(name)
            except Exception:
                continue   # model and recording don't match (e.g. tomer_with_finger)
            sc, m = yad_ref.load_head(name)
            results = []
            for tag, (Xo, co) in orders.items():
                pr = m.classes_[m.predict_proba(sc.transform(yad_ref.head_inputs(Xo, co, rms_over))).argmax(1)][mask]
                results.append((np.mean(pr == truth), tag, pr))
            worn = results[0]
            best = max(results, key=lambda r: r[0])
            print(f"  {name:12s} as worn:            {per_gesture(worn[2], truth)}")
            if best[1] != worn[1] and best[0] > worn[0] + 0.1:
                print(f"  {name:12s} best ({best[1]:>19s}): {per_gesture(best[2], truth)}  <- much better")

if __name__ == "__main__":
    main()
