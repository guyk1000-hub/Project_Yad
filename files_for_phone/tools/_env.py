"""
Import this before numpy in every tools/ script.

numpy 1.23.5 (pinned to match the Pi) bundles an old OpenBLAS that returns wrong
float64 matrix products on Intel Sapphire Rapids CPUs (seen on a Xeon 8488C:
relative error ~1.4, i.e. garbage). Forcing the Haswell (AVX2) kernel fixes it and
runs on any x86-64 CPU from 2013 onwards. The Pi (ARM) is not affected.
check_blas() verifies the result instead of trusting the workaround.
"""
import os
import platform

if platform.machine() in ("x86_64", "AMD64"):
    os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")


def check_blas():
    import numpy as np
    rng = np.random.default_rng(0)
    for dtype in (np.float64, np.float32):
        a = rng.standard_normal((200, 1026)).astype(dtype)
        b = rng.standard_normal((1026, 32)).astype(dtype)
        ref = np.einsum("ij,jk->ik", a.astype(np.float64), b.astype(np.float64))
        err = float(np.abs(a @ b - ref).max() / np.abs(ref).max())
        if err > 1e-4:
            raise RuntimeError(
                f"numpy matrix multiplication is broken on this machine ({dtype.__name__} rel. error {err:.2g}). "
                f"OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')}. Try OPENBLAS_CORETYPE=Haswell or SkylakeX.")
