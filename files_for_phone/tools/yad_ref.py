"""
Reference pipeline for the phone app.

The reference is the code that runs on the Raspberry Pi today. Wherever
possible it is *imported* from training_data/ and files_for_pi/ rather than
copied, so parity tests compare against what actually runs. The few pieces
that live inside functions (and so can't be imported) are copied verbatim,
with a pointer to their source.

Nothing in this module modifies the Pi code or its files.
"""
import hashlib
import os
import pickle
import sys

import _env  # must precede numpy

import numpy as np

_env.check_blas()

TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", ".."))  # files_for_phone/tools -> repo root
MODELS_DIR = os.path.join(REPO_ROOT, "training_data", "models")
DATA_DIR = os.path.join(REPO_ROOT, "training_data", "data")
OUT_DIR = os.path.join(TOOLS_DIR, "out")
GOLDEN_DIR = os.path.join(TOOLS_DIR, "golden")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "files_for_pi"))  # protocol.py is imported as a top-level module there
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from training_data.utils import BiquadMultiChan, FilterTypes, MyMagnWarping, MyScaling  # noqa: E402
from training_data.inference_mix import _compute_scalar_features  # noqa: E402

# Runtime parameters the Pi uses. Its config.json sets none of
# batch_size / prediction_threshold, so the defaults in
# files_for_pi/realtime_pi_classify.py apply.
SAMPLING_RATE = 500
MODEL_INPUT_LEN = 100
BATCH_SIZE = 6
PREDICTION_THRESHOLD = 0.7
FEATURE_LAYER = "dense_8"


def gesture_map():
    from realtime_pi_classify import GESTURE_MAP
    return dict(GESTURE_MAP)


def make_pi_filters(sampling_rate: int = SAMPLING_RATE):
    """Copied from files_for_pi/realtime_pi_classify.py main() (defined inline there)."""
    return [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0),
    ]


def pi_preprocess_window(window_T: np.ndarray, filters) -> np.ndarray:
    """
    Copied from training_data/inference_mix.py real_time_inference._preprocess_data
    (downsample_factor=1, which is what the Pi passes). window_T: [L, 8], filtered in place.
    """
    n_samples, n_channels = window_T.shape
    for f in filters:
        for ch in range(n_channels):
            for i in range(n_samples):
                window_T[i, ch] = f.process(window_T[i, ch], ch)
    return window_T


def pi_decisions(per_window_probs, batch_size: int = BATCH_SIZE, threshold: float = PREDICTION_THRESHOLD):
    """
    Copied from training_data/inference_mix.py real_time_inference._inference_worker.
    Returns a list of (index_of_last_window_in_group, top_class, avg_probs) for each emitted decision.
    """
    acc_probs, emitted = [], []
    for i, p in enumerate(per_window_probs):
        acc_probs.append(p)
        while len(acc_probs) >= batch_size:
            avg = np.mean(acc_probs[:batch_size], axis=0)
            del acc_probs[:batch_size]
            top = int(np.argmax(avg))
            if float(avg[top]) >= float(threshold):
                emitted.append((i, top, avg))
    return emitted


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_paths(model_name: str):
    """model_name is the suffix used in the repo, e.g. 'tomer' -> mlp_model_tomer.pkl + scaler_tomer.pkl."""
    return (os.path.join(MODELS_DIR, f"scaler_{model_name}.pkl"),
            os.path.join(MODELS_DIR, f"mlp_model_{model_name}.pkl"))


def load_head(model_name: str):
    scaler_path, mlp_path = head_paths(model_name)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(mlp_path, "rb") as f:
        mlp = pickle.load(f)
    return scaler, mlp


def dataset_path(model_name: str) -> str:
    return os.path.join(DATA_DIR, f"recorded_gestures_{model_name}.pkl")


def load_dataset(model_name: str):
    """Returns X [N, 8, 100, 1] float32 (already filtered, as recorded by record.py) and y [N]."""
    with open(dataset_path(model_name), "rb") as f:
        data, labels = pickle.load(f)
    return np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def feature_extractor_path() -> str:
    return os.path.join(MODELS_DIR, "og_feature_extraction.h5")


def load_feature_extractor():
    """Same construction as training_data/inference_mix.py."""
    from keras.models import Model, load_model
    fx = load_model(feature_extractor_path(),
                    custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling})
    return Model(inputs=fx.input, outputs=fx.get_layer(FEATURE_LAYER).output)


def cnn_features(X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """dense_8 embeddings [N, 1024], cached on disk (keyed by model + data hashes) since this is the slow step."""
    os.makedirs(os.path.join(OUT_DIR, "cache"), exist_ok=True)
    key = hashlib.sha256(sha256(feature_extractor_path()).encode() + X.tobytes()).hexdigest()[:16]
    cache = os.path.join(OUT_DIR, "cache", f"dense8_{key}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    feats = load_feature_extractor().predict(X, batch_size=batch_size, verbose=0)
    np.save(cache, feats)
    return feats


# Two versions of the asym/mag features exist, and models were trained with both:
#   "channels": RMS over axis=1 of [N, 8, L] (i.e. over the 8 channels). This is what
#               training_data/inference_mix.py and fine_tune.py compute today (since commit 0cff945).
#               Used by e.g. mlp_model_tomer_long, mlp_model_guy.
#   "time":     RMS over axis=2 (over time, per channel) — fine_tune.py at commit 23c0296.
#               Used by mlp_model_tomer.
RMS_OVER = ("channels", "time")


def scalar_features(X: np.ndarray, rms_over: str):
    if rms_over == "channels":
        return _compute_scalar_features(X)  # the Pi's own code
    if rms_over == "time":
        W = X[..., 0].astype(np.float32)
        rms = np.sqrt(np.mean(W * W, axis=2, dtype=np.float32))       # [N, 8]
        e_left = rms[:, [0, 1, 2, 3]].mean(axis=1, keepdims=True)
        e_right = rms[:, [4, 5, 6, 7]].mean(axis=1, keepdims=True)
        return e_right - e_left, rms.mean(axis=1, keepdims=True)
    raise ValueError(f"rms_over must be one of {RMS_OVER}")


def head_inputs(X: np.ndarray, cnn_feats: np.ndarray, rms_over: str = "channels") -> np.ndarray:
    """Same layout as inference_mix._inference_worker for a D+2 scaler: [dense_8, asym, mag]."""
    asym, mag = scalar_features(X, rms_over)
    return np.hstack((cnn_feats, asym, mag))


def detect_rms_over(model_name: str) -> str:
    """
    Which feature version was this head trained with? Recompute the scaler's fit on the same
    training split fine_tune.py uses; the right version reproduces the stored mean (|z| ~ 0).
    """
    from sklearn.model_selection import train_test_split
    X, y = load_dataset(model_name)
    scaler, _ = load_head(model_name)
    train_idx, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    z = {}
    for mode in RMS_OVER:
        asym, mag = scalar_features(X[train_idx], mode)
        means = np.array([asym.mean(), mag.mean()])
        z[mode] = float(np.abs((means - scaler.mean_[-2:]) / scaler.scale_[-2:]).max())
    best = min(z, key=z.get)
    if z[best] > 0.01:
        raise ValueError(f"{model_name}: no feature version matches the scaler (|z| per version: {z})")
    return best
