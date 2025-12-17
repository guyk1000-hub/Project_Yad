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
# Your mapping:
# 0 = rest
# 1 = close
# 2 = open
# 3 = right
# 4 = left
# 5 = point
POINT_ID = 5
REST_ID = 0
OPEN_ID = 2

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


def _compute_scalar_features(batch_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scalar features:
      - asym: [B,1] mean(RMS(right)) - mean(RMS(left))
      - mag : [B,1] mean RMS across all channels
      - spread_norm: [B,1] std(RMS across channels) / (mean RMS + eps)
        This tends to be higher when activation is more "localized" on fewer channels,
        which is often useful for 'point' as a tie-breaker.
    """
    W = batch_windows[..., 0]
    rms = np.sqrt(np.mean(W**2, axis=1))  # NOTE: matches your current working layout/implementation

    e_left  = np.mean(rms[:, EMG_LEFT_IDXS],  axis=1, keepdims=True)
    e_right = np.mean(rms[:, EMG_RIGHT_IDXS], axis=1, keepdims=True)

    asym = (e_right - e_left).astype(np.float32)
    mag  = np.mean(rms, axis=1, keepdims=True).astype(np.float32)

    spread = np.std(rms, axis=1, keepdims=True).astype(np.float32)
    spread_norm = (spread / (mag + 1e-6)).astype(np.float32)

    return asym, mag, spread_norm


def real_time_inference(
    feature_extractor_path: str,
    mlp_model_path: str,
    scaler_path: str,
    filters,
    model_input_len: int = 100,
    gyro_threshold: int = 500,    # unused; kept for API compatibility
    prediction_threshold: float = 0.6,
    batch_size: int = 8,
    downsample_factor: int = 1,
):
    """
    Real-time inference:
      - Extract CNN embedding from window(s)
      - Add 2 scalar EMG features (asymmetry, magnitude) EXACTLY like training
      - Enforce D+2 feature vector to stay aligned with saved scaler+MLP
      - Compute an extra 'spread_norm' feature used ONLY as a post-rule tie-breaker for POINT
      - Optional downsampling after filtering

    Yields: (prediction:int, avg_probabilities:np.ndarray)
    """
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    # ----- Point booster tuning knobs (no retrain needed) -----
    # Only applies when POINT is close to top prediction (uncertain case)
    POINT_MARGIN = 0.08           # if (top_prob - point_prob) <= this, consider boosting point
    POINT_MIN_PROB = 0.18         # don't boost if point prob is tiny
    SPREAD_NORM_THRESH = 0.35     # higher => stricter boosting (start around 0.30-0.45)
    BOOST_ONLY_AGAINST = {REST_ID, OPEN_ID}  # typical confusions for point

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
    data_queue: "Queue[np.ndarray]" = Queue()
    output_queue: "Queue[tuple[int, np.ndarray]]" = Queue()
    stop_event = threading.Event()

    def _preprocess_data(window_T: np.ndarray) -> np.ndarray:
        """
        Apply filter stack (in-place), then optional downsample.
        window_T: [L_raw, 8] -> filtered [L_eff, 8]
        """
        n_samples, n_channels = window_T.shape
        f_list = filters

        for f in f_list:
            for ch in range(n_channels):
                for i in range(n_samples):
                    window_T[i, ch] = f.process(window_T[i, ch], ch)

        if downsample_factor > 1:
            window_T = window_T[::downsample_factor, :]

        return window_T

    def _inference_worker():
        acc_probs = []
        acc_spread = []

        while not stop_event.is_set():
            try:
                batch_windows = data_queue.get(timeout=0.2)
            except Empty:
                continue

            feats = feature_extractor.predict(batch_windows, verbose=0)  # [B, D]
            asym, mag, spread_norm = _compute_scalar_features(batch_windows)

            D = feats.shape[1]
            expected = n_expected if n_expected is not None else (D + 2)
            if expected != D + 2:
                raise RuntimeError(
                    f"Feature mismatch: scaler expects {expected} features, "
                    f"but extractor produced D={D}. This model was trained with D+2."
                )

            # IMPORTANT: keep training alignment (D+2 only)
            feats_plus = np.hstack((feats, asym, mag))
            feats_scaled = scaler.transform(feats_plus)

            probs = mlp_model.predict_proba(feats_scaled)  # [B, n_cls]

            acc_probs.extend(probs)
            acc_spread.extend(spread_norm.reshape(-1).tolist())

            # Aggregate probabilities over chunks of size batch_size
            while len(acc_probs) >= batch_size and len(acc_spread) >= batch_size:
                avg = np.mean(acc_probs[:batch_size], axis=0)
                avg_spread = float(np.mean(acc_spread[:batch_size]))

                del acc_probs[:batch_size]
                del acc_spread[:batch_size]

                top = int(np.argmax(avg))
                top_prob = float(avg[top])

                # ----- Point tie-breaker rule (no retrain) -----
                # If the model is already confident, don't touch it.
                # Only intervene when point is close AND spread_norm indicates "localized" activity.
                if POINT_ID < len(avg):
                    point_prob = float(avg[POINT_ID])
                    if (
                        top in BOOST_ONLY_AGAINST and
                        point_prob >= POINT_MIN_PROB and
                        (top_prob - point_prob) <= POINT_MARGIN and
                        avg_spread >= SPREAD_NORM_THRESH
                    ):
                        top = POINT_ID
                        top_prob = point_prob

                if top_prob >= float(prediction_threshold):
                    output_queue.put((top, avg))

    try:
        board.prepare_session()
        board.start_stream(450000)

        worker = threading.Thread(target=_inference_worker, daemon=True)
        worker.start()

        batch = []
        raw_len = model_input_len * downsample_factor

        while not stop_event.is_set():
            if board.get_board_data_count() < raw_len:
                time.sleep(0.002)
                continue

            raw = board.get_board_data(raw_len)
            emg = raw[:8]  # [8, T_raw]

            emg_proc_T = _preprocess_data(emg.T)  # [L_eff, 8]
            if emg_proc_T.shape[0] != model_input_len:
                continue

            # Window format: [8, L, 1]
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
