
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from keras.models import Model, load_model
import pickle
import logging
import numpy as np
import sys

from training_data.utils import MyMagnWarping, MyScaling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- CLASS IDS (for reference) -----
# 0 = rest, 1 = close, 2 = open, 3 = right, 4 = left

# ----- Electrode layout (physical sides) -----
# Adjust to your armband channel order if needed.
EMG_LEFT_IDXS  = (0, 1, 2, 3)   # channels on the "left-side" of the forearm
EMG_RIGHT_IDXS = (4, 5, 6, 7)   # channels on the "right-side" of the forearm

def _side_asymmetry_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    Returns [N,1] = mean(RMS(right_channels)) - mean(RMS(left_channels))
    batch_windows: [N, 8, L, 1]
    """
    W = batch_windows[..., 0]                       # [N, 8, L]
    rms = np.sqrt(np.mean(W**2, axis=1))           # [N, 8]
    e_left  = np.mean(rms[:, list(EMG_LEFT_IDXS)],  axis=1, keepdims=True)
    e_right = np.mean(rms[:, list(EMG_RIGHT_IDXS)], axis=1, keepdims=True)
    return (e_right - e_left).astype(np.float32)   # + sign tends toward "right" activation


def _magnitude_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    Returns [N,1] = mean RMS over all 8 channels
    batch_windows: [N, 8, L, 1]
    """
    W = batch_windows[..., 0]                       # [N, 8, L]
    rms = np.sqrt(np.mean(W**2, axis=1))           # [N, 8]
    mag = np.mean(rms, axis=1, keepdims=True)      # [N, 1]
    return mag.astype(np.float32)


def _takes(labels: np.ndarray) -> np.ndarray:
    """
    Take number of each window: record.py records each gesture several times ("takes"), so a take
    is a contiguous run of one label. The first run of each gesture is take 0, the next take 1, ...
    """
    takes = np.zeros(len(labels), dtype=int)
    seen = {}
    for i, lab in enumerate(labels):
        if i == 0 or labels[i - 1] != lab:
            seen[lab] = seen.get(lab, -1) + 1
        takes[i] = seen[lab]
    return takes


def _held_out_take_accuracy(features: np.ndarray, labels: np.ndarray):
    """
    Honest accuracy estimate: for each take k, train a fresh scaler + MLP (same settings as
    fine_tune_model) on the other takes and test on take k. Returns the mean, or None when there
    is only one take per gesture. Doesn't change the saved model.
    """
    takes = _takes(labels)
    common = sorted(set.intersection(*(set(takes[labels == c]) for c in np.unique(labels))))
    if len(common) < 2:
        return None
    scores = []
    for k in common:
        test = takes == k
        sc = StandardScaler().fit(features[~test])
        m = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=20000, random_state=42,
                          activation='tanh', solver='lbfgs', alpha=0.01, learning_rate='constant')
        m.fit(sc.transform(features[~test]), labels[~test])
        scores.append(m.score(sc.transform(features[test]), labels[test]))
        logging.info(f"  held-out take {k}: {scores[-1]:.4f}")
    return float(np.mean(scores))


def fine_tune_model(
    feature_extractor_path,
    recorded_data,
    recorded_labels,
    mlp_model_path,
    scaler_path,
):
    """
    Fine-tunes the MLP on CNN embeddings + 2 scalar EMG features (asymmetry, magnitude).
    recorded_data: list/array of windows [L,8,1]
    recorded_labels: list/array of ints in {0..4} (0=rest,1=close,2=open,3=right,4=left)
    """
    model = load_model(
        feature_extractor_path,
        custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling}
    )

    X = np.asarray(recorded_data, dtype=np.float32)  # [N, L, 8, 1]
    y = np.asarray(recorded_labels, dtype=np.int32)

    # Same split as before, done on indices so the honest evaluation below can reuse the features.
    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = X[idx_train], X[idx_val], y[idx_train], y[idx_val]

    # CNN feature extractor (keep layer name aligned with your model)
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense_8").output)
    feats_all = feature_extractor.predict(X, verbose=0)  # [N, D]

    # Add 2 scalar features
    feats_all_plus = np.hstack([feats_all, _side_asymmetry_feature(X), _magnitude_feature(X)])  # [N, D+2]
    feats_train_plus = feats_all_plus[idx_train]
    feats_val_plus   = feats_all_plus[idx_val]

    # Scale & train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(feats_train_plus)
    X_val_scaled   = scaler.transform(feats_val_plus)
    logging.info("Data preprocessed and scaled.")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        logging.info("Scaler saved.")

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=20000,
        random_state=42,
        activation='tanh',
        solver='lbfgs',
        alpha=0.01,
        learning_rate='constant',
    )
    mlp.fit(X_train_scaled, y_train)
    split_accuracy = mlp.score(X_val_scaled, y_val)
    # Optimistic: neighbouring windows overlap by 90%, so a random split puts near-copies of the
    # training windows in the validation set.
    logging.info(f"MLP validation accuracy (random split, optimistic): {split_accuracy:.4f}")

    honest = _held_out_take_accuracy(feats_all_plus, y)
    if honest is None:
        logging.warning("Only one take per gesture: can't hold one out, reporting the random split.")
        mlp_accuracy = split_accuracy
    else:
        mlp_accuracy = honest
        logging.info(f"MLP held-out-take accuracy (honest): {mlp_accuracy:.4f}")

    with open(mlp_model_path, "wb") as f:
        pickle.dump(mlp, f)
        logging.info("MLP model saved.")

    if mlp_accuracy < 0.60:
        response = input(
            "MLP validation accuracy is below 60%. Would you like to stop and check the device (Yes/No)? "
        ).strip().lower()
        if response in ("yes", "y"):
            logging.info(
                "Suggested actions:\n"
                "- Check device connection.\n"
                "- Ensure proper device positioning.\n"
                "- Clean sensors.\n"
                "- Record new data."
            )
            logging.info("Exiting the code.")
            sys.exit(0)
        else:
            logging.info("Continuing despite low validation accuracy.")

    return scaler, mlp_accuracy

