
from sklearn.model_selection import train_test_split, cross_val_score
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
    batch_windows: [N, L, 8, 1]
    """
    W = batch_windows[..., 0]                       # [N, L, 8]
    rms = np.sqrt(np.mean(W**2, axis=1))            # [N, 8]
    e_left  = np.mean(rms[:, list(EMG_LEFT_IDXS)],  axis=1, keepdims=True)
    e_right = np.mean(rms[:, list(EMG_RIGHT_IDXS)], axis=1, keepdims=True)
    return (e_right - e_left).astype(np.float32)    # + sign tends toward "right" activation

def _magnitude_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    Returns [N,1] = mean RMS over all 8 channels
    batch_windows: [N, L, 8, 1]
    """
    W = batch_windows[..., 0]                       # [N, L, 8]
    rms = np.sqrt(np.mean(W**2, axis=1))            # [N, 8]
    mag = np.mean(rms, axis=1, keepdims=True)       # [N, 1]
    return mag.astype(np.float32)

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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CNN feature extractor (keep layer name aligned with your model)
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense_8").output)
    feats_train = feature_extractor.predict(X_train, verbose=0)  # [N, D]
    feats_val   = feature_extractor.predict(X_val,   verbose=0)  # [N, D]

    # Add 2 scalar features
    a_train = _side_asymmetry_feature(X_train)  # [N,1]
    a_val   = _side_asymmetry_feature(X_val)    # [N,1]
    m_train = _magnitude_feature(X_train)       # [N,1]
    m_val   = _magnitude_feature(X_val)         # [N,1]

    feats_train_plus = np.hstack([feats_train, a_train, m_train])  # [N, D+2]
    feats_val_plus   = np.hstack([feats_val,   a_val,   m_val])    # [N, D+2]

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
    mlp_accuracy = mlp.score(X_val_scaled, y_val)
    logging.info(f"MLP validation accuracy: {mlp_accuracy}")
    logging.info(f"MLP cross-validation accuracy: {cross_val_score(mlp, X_train_scaled, y_train, cv=5)}")

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
