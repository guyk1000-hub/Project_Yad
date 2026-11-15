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
        scaler, mlp = yad_ref.load_head(h["model"])
        p = mlp.predict_proba(scaler.transform(yad_ref.head_inputs(X, cnn, h["scalar_rms_over"])))
        settled = np.arange(len(w)) >= 20   # skip the first 4 s: the Pi code's filters start cold here
        diff = np.abs(p - w["probs"])[settled].max()
        print(f"Pi code, same model:      {per_gesture(mlp.classes_[p.argmax(1)][mask], truth)}")
        print(f"phone vs Pi code: max |Δp| = {diff:.2e} after the first 4 s "
              f"({'same computation' if diff < 1e-3 else 'DIFFERENT — phone bug, look closer'})")

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
