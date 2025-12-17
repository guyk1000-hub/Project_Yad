import threading
import time
import os
from queue import Queue, Empty
import cv2
import numpy as np
import pickle
from keras.models import load_model, Model
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from training_data.utils import MyMagnWarping, MyScaling

# ----- CLASS IDS -----
# 0 = rest, 1 = close, 2 = open, 3 = right, 4 = left
#for 250HZ change timeout = 0.4 , batch = 4 and config file downsample factor = 2 
# for 500HZ change timeout = 0.2 , batch = 8 , downsample factor = 1
# ----- Electrode layout (physical sides) -----
EMG_LEFT_IDXS  = np.array([0, 1, 2, 3])
EMG_RIGHT_IDXS = np.array([4, 5, 6, 7])

_last_displayed_gesture = None


def show_image_for_prediction(prediction: int, gesture_image_path: str, skip_gestures):
    """Display g{prediction}.png without blocking; respects skip_gestures mapping."""
    global _last_displayed_gesture

    # Fast skip-gesture adjustment
    for sg in skip_gestures:
        if prediction >= sg:
            prediction += 1

    if _last_displayed_gesture == prediction:
        return
    _last_displayed_gesture = prediction

    image_file = os.path.join(gesture_image_path, f"g{prediction}.png")
    if not os.path.exists(image_file):
        return

    img = cv2.imread(image_file)
    if img is None:
        return

    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    name = "Current Gesture"

    cv2.destroyAllWindows()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, img.shape[1], img.shape[0])
    cv2.waitKey(1)


def _compute_scalar_features(batch_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    batch_windows: [B, L, 8, 1]
    returns:
      asym: [B,1] = mean(RMS(right)) - mean(RMS(left))
      mag:  [B,1] = mean RMS over all channels
    """
    # [B, L, 8]
    W = batch_windows[..., 0].astype(np.float32)

    # RMS per channel: [B, 8]
    rms = np.sqrt(np.mean(W * W, axis=1, dtype=np.float32))

    # Side energies
    e_left = rms[:, EMG_LEFT_IDXS].mean(axis=1, keepdims=True)
    e_right = rms[:, EMG_RIGHT_IDXS].mean(axis=1, keepdims=True)

    asym = e_right - e_left                 # [B, 1]
    mag = rms.mean(axis=1, keepdims=True)   # [B, 1]

    return asym, mag


def real_time_inference(
    feature_extractor_path: str,
    mlp_model_path: str,
    scaler_path: str,
    filters,
    model_input_len: int = 100,
    gyro_threshold: int = 500,    # unused; kept for API compatibility
    prediction_threshold: float = 0.6,
    batch_size: int = 8,
    downsample_factor: int = 1,   # <<< NEW
):
    """
    Real-time inference:
      - Extract CNN embedding from window(s)
      - Optionally add 1–2 scalar features (asymmetry, magnitude)
      - Auto-match scaler.n_features_in_ so it works with D, D+1 or D+2 models
      - Optionally downsample in time by `downsample_factor` after filtering.
    Yields: (prediction:int, avg_probabilities:np.ndarray)
    """
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    # ===== Load models & scaler =====
    fx = load_model(
        feature_extractor_path,
        custom_objects={"MyMagnWarping": MyMagnWarping, "MyScaling": MyScaling},
    )
    feature_extractor = Model(
        inputs=fx.input,
        outputs=fx.get_layer("dense_8").output
    )

    with open(scaler_path, "rb") as fsc:
        scaler = pickle.load(fsc)
    with open(mlp_model_path, "rb") as fmlp:
        mlp_model = pickle.load(fmlp)

    n_expected = getattr(scaler, "n_features_in_", None)

    # ===== MindRove board =====
    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board = BoardShim(board_id, params)

    # ===== Queues / threading =====
    data_queue: "Queue[np.ndarray]" = Queue()      # [B, L, 8, 1]
    output_queue: "Queue[tuple[int, np.ndarray]]" = Queue()
    stop_event = threading.Event()

    def _preprocess_data(window_T: np.ndarray) -> np.ndarray:
        """
        Apply filter stack (in-place), then optional downsample.
        window_T: [L_raw, 8] -> filtered [L_eff, 8]
        """
        n_samples, n_channels = window_T.shape
        # Bring lookups to local vars (faster in tight loops)
        f_list = filters

        for f in f_list:
            for ch in range(n_channels):
                for i in range(n_samples):
                    window_T[i, ch] = f.process(window_T[i, ch], ch)

        # Optional decimation after filtering
        if downsample_factor > 1:
            window_T = window_T[::downsample_factor, :]

        return window_T

    def _inference_worker():
        acc_probs = []

        while not stop_event.is_set():
            try:
                batch_windows = data_queue.get(timeout=0.2)  # [B, L, 8, 1]
            except Empty:
                continue

            # CNN features
            feats = feature_extractor.predict(batch_windows, verbose=0)  # [B, D]
            asym, mag = _compute_scalar_features(batch_windows)          # [B, 1], [B, 1]

            D = feats.shape[1]

            # Decide how many extra features to add based on the scaler
            if n_expected is None or n_expected == D + 2:
                feats_plus = np.hstack((feats, asym, mag))               # D+2
            elif n_expected == D + 1:
                feats_plus = np.hstack((feats, asym))                    # D+1
            elif n_expected == D:
                feats_plus = feats                                       # D
            else:
                # Generic truncate/pad to n_expected
                fp = np.hstack((feats, asym, mag))
                if fp.shape[1] >= n_expected:
                    feats_plus = fp[:, :n_expected]
                else:
                    pad = np.zeros((fp.shape[0], n_expected - fp.shape[1]), dtype=fp.dtype)
                    feats_plus = np.hstack((fp, pad))

            feats_scaled = scaler.transform(feats_plus)
            probs = mlp_model.predict_proba(feats_scaled)                # [B, n_cls]

            acc_probs.extend(probs)

            # Aggregate probabilities over batches of size batch_size
            while len(acc_probs) >= batch_size:
                avg = np.mean(acc_probs[:batch_size], axis=0)
                del acc_probs[:batch_size]

                top = int(np.argmax(avg))
                if float(avg[top]) >= float(prediction_threshold):
                    output_queue.put((top, avg))

    try:
        board.prepare_session()
        board.start_stream(450000)

        worker = threading.Thread(target=_inference_worker, daemon=True)
        worker.start()

        batch = []

        # how many raw samples we pull before optional downsampling
        raw_len = model_input_len * downsample_factor

        while not stop_event.is_set():
            # Avoid 100% CPU busy-waiting
            if board.get_board_data_count() < raw_len:
                time.sleep(0.002)
                continue

            # [channels, T_raw]
            raw = board.get_board_data(raw_len)
            emg = raw[:8]  # [8, T_raw]

            # [L_eff, 8]
            emg_proc_T = _preprocess_data(emg.T)

            # after downsampling we expect exactly model_input_len
            if emg_proc_T.shape[0] != model_input_len:
                # config mismatch (e.g., wrong downsample_factor) – skip this window
                continue

            # [L, 8, 1] -> use float32 (better for Keras)
            window = emg_proc_T.T[..., np.newaxis].astype(np.float32)
            batch.append(window)

            if len(batch) == batch_size:
                data_queue.put(np.array(batch, dtype=np.float32))
                batch = []

            # Drain output queue
            while True:
                try:
                    yield output_queue.get_nowait()
                except Empty:
                    break

    except Exception as e:
        raise RuntimeError(f"Error during real-time inference: {e}") from e

    finally:
        stop_event.set()
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass