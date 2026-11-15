"""
Export a user's recorded training data as head inputs (1026 features per window) for training on
the phone, so the phone doesn't have to run the CNN over thousands of old windows.

Recorded windows overlap by 90%, so every 3rd one is kept (--stride): that loses little and keeps
the file and the phone's training time small.

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_train_features.py tomer guy tomer_long

Format (little-endian): b"YADT", int32 version=1, int32 n, int32 n_features, int32 rms_over
(0 = channels, 1 = time), int32 labels[n], int32 takes[n], float32 features[n][n_features].
"""
import argparse
import os
import struct

import _env  # noqa: F401  must precede numpy
import numpy as np

import yad_ref

MAGIC, VERSION = b"YADT", 1


def takes(labels: np.ndarray) -> np.ndarray:
    """Take number per window: each contiguous run of one label is a take (as in fine_tune.py)."""
    out = np.zeros(len(labels), dtype=int)
    seen = {}
    for i, lab in enumerate(labels):
        if i == 0 or labels[i - 1] != lab:
            seen[lab] = seen.get(lab, -1) + 1
        out[i] = seen[lab]
    return out


def export(name: str, stride: int, out_dir: str) -> str:
    X, y = yad_ref.load_dataset(name)
    rms_over = yad_ref.detect_rms_over(name)
    F = yad_ref.head_inputs(X, yad_ref.cnn_features(X), rms_over).astype("<f4")
    t = takes(y)
    F, y, t = F[::stride], y[::stride], t[::stride]
    path = os.path.join(out_dir, f"train_{name}.bin")
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<4i", VERSION, len(y), F.shape[1], yad_ref.RMS_OVER.index(rms_over)))
        f.write(y.astype("<i4").tobytes())
        f.write(t.astype("<i4").tobytes())
        f.write(np.ascontiguousarray(F).tobytes())
    print(f"[{name}] {len(y)} windows, rms over {rms_over}, per class {np.bincount(y).tolist()} "
          f"-> {os.path.relpath(path, yad_ref.REPO_ROOT)} ({os.path.getsize(path) // 1024} KB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default=yad_ref.PHONE_MODELS_DIR)
    args = ap.parse_args()
    for name in args.models:
        export(name, args.stride, args.out)


if __name__ == "__main__":
    main()
