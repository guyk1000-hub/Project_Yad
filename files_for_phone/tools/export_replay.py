"""
Export a recording as a compact replay file the phone app can play back without the armband.

record.py stores windows of 100 samples taken every 10 samples, so consecutive windows overlap
by 90%. Taking every 10th window (--stride 10) gives back-to-back 200 ms windows, i.e. what a
live stream would produce.

Usage (from the repo root):
    files_for_phone/tools/.venv/bin/python files_for_phone/tools/export_replay.py tomer guy

Format (little-endian): b"YADR", int32 version=1, int32 n, int32 channels, int32 length,
int32 labels[n], float32 windows[n][channels][length] (filtered µV, as recorded).
"""
import argparse
import os
import struct

import _env  # noqa: F401  must precede numpy
import numpy as np

import yad_ref

MAGIC, VERSION = b"YADR", 1


def export(model_name: str, stride: int, out_dir: str) -> str:
    X, y = yad_ref.load_dataset(model_name)
    X, y = X[::stride, ..., 0], y[::stride]           # [n, 8, 100]
    path = os.path.join(out_dir, f"replay_{model_name}.bin")
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<4i", VERSION, len(X), X.shape[1], X.shape[2]))
        f.write(y.astype("<i4").tobytes())
        f.write(np.ascontiguousarray(X, dtype="<f4").tobytes())
    counts = np.bincount(y)
    print(f"[{model_name}] {len(X)} windows ({len(X) * X.shape[2] / yad_ref.SAMPLING_RATE:.0f} s), "
          f"per class {counts.tolist()} -> {os.path.relpath(path, yad_ref.REPO_ROOT)} ({os.path.getsize(path) // 1024} KB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=yad_ref.PHONE_MODELS_DIR)
    args = ap.parse_args()
    for name in args.models:
        export(name, args.stride, args.out)


if __name__ == "__main__":
    main()
