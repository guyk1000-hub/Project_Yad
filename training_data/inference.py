import os
import threading
from queue import Queue, Empty

import cv2
import numpy as np
import pickle
from keras.models import Model, load_model
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

from training_data.utils import MyMagnWarping, MyScaling

# ----- CLASS IDS -----
# 0 = rest, 1 = close, 2 = open, 3 = right, 4 = left, 5 = point

# ----- Electrode layout (physical sides) -----
# Adjust to match your armband’s channel order if needed.
EMG_LEFT_IDXS = (0, 1, 2, 3)
EMG_RIGHT_IDXS = (4, 5, 6, 7)

_last_displayed_gesture = None


def show_image_for_prediction(prediction: int,
                              gesture_image_path: str,
                              skip_gestures):
    """
    Display g{prediction}.png without blocking; respects skip_gestures mapping.
    """
    global _last_displayed_gesture

    # Map prediction index if some gestures are skipped/removed
    for sg in skip_gestures:
        if prediction >= sg:
            prediction += 1

    i = prediction
    if _last_displayed_gesture == i:
        return
    _last_displayed_gesture = i

    image_file = os.path.join(gesture_image_path, f"g{i}.png")
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


def _side_asymmetry_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    batch_windows: [B, L, 8, 1]
    returns [B,1] = mean(RMS(right)) - mean(RMS(left))
    """
    W = batch_windows[..., 0]          # [B, L, 8]
    rms = np.sqrt(np.mean(W ** 2, axis=1))  # [B, 8]
    e_left = np.mean(rms[:, list(EMG_LEFT_IDXS)], axis=1, keepdims=True)
    e_right = np.mean(rms[:, list(EMG_RIGHT_IDXS)], axis=1, keepdims=True)
    return (e_right - e_left).astype(np.float32)


def _magnitude_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    batch_windows: [B, L, 8, 1]
    returns [B,1] = mean RMS over all channels
    """
    W = batch_windows[..., 0]          # [B, L, 8]
    rms = np.sqrt(np.mean(W ** 2, axis=1))  # [B, 8]
    mag = np.mean(rms, axis=1, keepdims=True)  # [B, 1]
    return mag.astype(np.float32)


def _channel_variance_feature(batch_windows: np.ndarray) -> np.ndarray:
    """
    batch_windows: [B, L, 8, 1]
    returns [B,1] = variance of per-channel RMS across 8 channels.
    """
    W = batch_windows[..., 0]                # [B, L, 8]
    rms = np.sqrt(np.mean(W ** 2, axis=1))   # [B, 8]
    var = np.var(rms, axis=1, keepdims=True) # [B, 1]
    return var.astype(np.float32)


def _channel_sparsity_feature(batch_windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    batch_windows: [B, L, 8, 1]
    returns [B,1] = max_rms / mean_rms across channels.
    High when one/few channels dominate, low when spread evenly.
    """
    W = batch_windows[..., 0]                # [B, L, 8]
    rms = np.sqrt(np.mean(W ** 2, axis=1))   # [B, 8]
    max_r = np.max(rms, axis=1, keepdims=True)
    mean_r = np.mean(rms, axis=1, keepdims=True)
    ratio = max_r / (mean_r + eps)
    return ratio.astype(np.float32)


def real_time_inference(
    feature_extractor_path: str,
    mlp_model_path: str,
    scaler_path: str,
    filters,
    model_input_len: int = 100,
    gyro_threshold: float = 500.0,  # unused; kept for API compatibility
    prediction_threshold: float = 0.6,
    batch_size: int = 8,
    downsample_factor: int = 1,
):
    """
    Real-time inference:

    - MindRove always streams at 500 Hz (via BoardShim).
    - We run the biquad filter stack at the raw rate.
    - Optionally downsample in time by `downsample_factor` AFTER filtering, i.e.:

        raw_len = model_input_len * downsample_factor           # samples from board
        filtered_raw -> decimate -> L = model_input_len samples # to the model

    - Extract CNN embedding from window(s)
    - Add up to 4 scalar features (asymmetry, magnitude, channel variance, sparsity)
    - Auto-match scaler.n_features_in_ so it works with D..D+4 models

    Yields:
        (prediction:int, avg_probabilities:np.ndarray)
    """

    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    # -------------------------------------------------------
    # Load feature extractor, scaler, and MLP classifier
    # -------------------------------------------------------
    fx = load_model(
        feature_extractor_path,
        custom_objects={
            "MyMagnWarping": MyMagnWarping,
            "MyScaling": MyScaling,
        },
    )
    feature_extractor = Model(
        inputs=fx.input,
        outputs=fx.get_layer("dense_8").output,
    )

    with open(scaler_path, "rb") as fsc:
        scaler = pickle.load(fsc)

    with open(mlp_model_path, "rb") as fmlp:
        mlp_model = pickle.load(fmlp)

    n_expected = getattr(scaler, "n_features_in_", None)

    # -------------------------------------------------------
    # MindRove board setup
    # -------------------------------------------------------
    BoardShim.enable_dev_board_logger()
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board = BoardShim(board_id, params)

    # Queues / threading
    data_queue: "Queue[np.ndarray]" = Queue()          # [B, L, 8, 1]
    output_queue: "Queue[tuple[int, np.ndarray]]" = Queue()
    stop_event = threading.Event()

    # -------------------------------------------------------
    # Local preprocessing: filter stack + optional decimation
    # -------------------------------------------------------
    def _preprocess_data(window_T: np.ndarray) -> np.ndarray:
        """
        window_T: [L_raw, 8] at raw sampling rate (~500 Hz)
        Returns: [L_eff, 8] after filters (+ optional downsample).
        """
        # Apply filter stack sample-by-sample (per channel)
        # filters: list of BiquadMultiChan configured at raw Fs.
        L_raw, n_ch = window_T.shape

        for i in range(L_raw):
            for ch in range(n_ch):
                x = window_T[i, ch]
                for f in filters:
                    x = f.process(x, ch)
                window_T[i, ch] = x

        # Optional decimation in time
        if downsample_factor > 1:
            window_T = window_T[::downsample_factor, :]

        return window_T

    # -------------------------------------------------------
    # Worker: consumes windows from data_queue and pushes predictions
    # -------------------------------------------------------
    def _inference_worker():
        acc_probs = []

        while not stop_event.is_set():
            try:
                batch_windows = data_queue.get(timeout=1)  # [B, L, 8, 1]
            except Empty:
                continue

            # CNN feature extraction
            feats = feature_extractor.predict(batch_windows, verbose=0)  # [B, D]

            # Extra scalar features (must match fine_tune.py order)
            asym   = _side_asymmetry_feature(batch_windows)       # [B,1]
            mag    = _magnitude_feature(batch_windows)            # [B,1]
            var_ch = _channel_variance_feature(batch_windows)     # [B,1]
            sparse = _channel_sparsity_feature(batch_windows)     # [B,1]

            D = feats.shape[1]

            # Try to match scaler's expected dimensionality
            if n_expected is None or n_expected == D + 4:
                feats_plus = np.hstack([feats, asym, mag, var_ch, sparse])   # D+4
            elif n_expected == D + 3:
                feats_plus = np.hstack([feats, asym, mag, var_ch])          # D+3
            elif n_expected == D + 2:
                feats_plus = np.hstack([feats, asym, mag])                  # D+2
            elif n_expected == D + 1:
                feats_plus = np.hstack([feats, asym])                       # D+1
            elif n_expected == D:
                feats_plus = feats                                          # D
            else:
                # Generic truncate/pad to n_expected
                fp = np.hstack([feats, asym, mag, var_ch, sparse])         # [B, D+4]
                if fp.shape[1] >= n_expected:
                    feats_plus = fp[:, :n_expected]
                else:
                    pad = np.zeros(
                        (fp.shape[0], n_expected - fp.shape[1]),
                        dtype=fp.dtype,
                    )
                    feats_plus = np.hstack([fp, pad])

            feats_scaled = scaler.transform(feats_plus)
            probs = mlp_model.predict_proba(feats_scaled)  # [B, n_cls]

            for p in probs:
                acc_probs.append(p)

            # Agg over `batch_size` windows
            while len(acc_probs) >= batch_size:
                avg = np.mean(acc_probs[:batch_size], axis=0)
                del acc_probs[:batch_size]

                top = int(np.argmax(avg))
                if float(avg[top]) >= float(prediction_threshold):
                    output_queue.put((top, avg))

    # -------------------------------------------------------
    # Main streaming loop
    # -------------------------------------------------------
    raw_len = model_input_len * max(1, downsample_factor)

    try:
        board.prepare_session()
        board.start_stream(450000)

        worker = threading.Thread(target=_inference_worker, daemon=True)
        worker.start()

        batch = []

        while not stop_event.is_set():
            if board.get_board_data_count() < raw_len:
                continue

            # raw: [channels, T_raw]
            raw = board.get_board_data(raw_len)
            emg = raw[:8]  # [8, T_raw]

            # [T_raw, 8] -> filter + (optional) downsample -> [L_eff, 8]
            emg_proc_T = _preprocess_data(emg.T)

            # Sanity: after decimation, we expect L_eff == model_input_len
            if emg_proc_T.shape[0] != model_input_len:
                # If not, either adjust model_input_len or downsample_factor
                # in config to keep them consistent.
                # For now, we skip this window rather than crash.
                continue

            # [L_eff, 8] -> [L_eff, 8, 1]
            window = np.expand_dims(emg_proc_T.T, axis=2).astype(np.float32)
            batch.append(window)

            if len(batch) == batch_size:
                data_queue.put(np.array(batch))
                batch = []

            # Emit any ready predictions
            while not output_queue.empty():
                yield output_queue.get()

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







# " # training_data/inference.py

# import os
# import threading
# from queue import Queue, Empty

# import cv2
# import numpy as np
# import pickle
# from keras.models import Model, load_model
# from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# from training_data.utils import MyMagnWarping, MyScaling

# # ----- CLASS IDS -----
# # 0 = rest, 1 = close, 2 = open, 3 = right, 4 = left

# # ----- Electrode layout (physical sides) -----
# # Adjust to match your armband’s channel order if needed.
# EMG_LEFT_IDXS = (0, 1, 2, 3)
# EMG_RIGHT_IDXS = (4, 5, 6, 7)

# _last_displayed_gesture = None


# def show_image_for_prediction(prediction: int,
#                               gesture_image_path: str,
#                               skip_gestures):
#     """
#     Display g{prediction}.png without blocking; respects skip_gestures mapping.
#     """
#     global _last_displayed_gesture

#     # Map prediction index if some gestures are skipped/removed
#     for sg in skip_gestures:
#         if prediction >= sg:
#             prediction += 1

#     i = prediction
#     if _last_displayed_gesture == i:
#         return
#     _last_displayed_gesture = i

#     image_file = os.path.join(gesture_image_path, f"g{i}.png")
#     if not os.path.exists(image_file):
#         return

#     img = cv2.imread(image_file)
#     if img is None:
#         return

#     img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
#     name = "Current Gesture"
#     cv2.destroyAllWindows()
#     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
#     cv2.imshow(name, img)
#     cv2.resizeWindow(name, img.shape[1], img.shape[0])
#     cv2.waitKey(1)


# def _side_asymmetry_feature(batch_windows: np.ndarray) -> np.ndarray:
#     """
#     batch_windows: [B, L, 8, 1]
#     returns [B,1] = mean(RMS(right)) - mean(RMS(left))
#     """
#     W = batch_windows[..., 0]          # [B, L, 8]
#     rms = np.sqrt(np.mean(W ** 2, axis=1))  # [B, 8]
#     e_left = np.mean(rms[:, list(EMG_LEFT_IDXS)], axis=1, keepdims=True)
#     e_right = np.mean(rms[:, list(EMG_RIGHT_IDXS)], axis=1, keepdims=True)
#     return (e_right - e_left).astype(np.float32)


# def _magnitude_feature(batch_windows: np.ndarray) -> np.ndarray:
#     """
#     batch_windows: [B, L, 8, 1]
#     returns [B,1] = mean RMS over all channels
#     """
#     W = batch_windows[..., 0]          # [B, L, 8]
#     rms = np.sqrt(np.mean(W ** 2, axis=1))  # [B, 8]
#     mag = np.mean(rms, axis=1, keepdims=True)  # [B, 1]
#     return mag.astype(np.float32)


# def real_time_inference(
#     feature_extractor_path: str,
#     mlp_model_path: str,
#     scaler_path: str,
#     filters,
#     model_input_len: int = 100,
#     gyro_threshold: float = 500.0,  # unused; kept for API compatibility
#     prediction_threshold: float = 0.6,
#     batch_size: int = 8,
#     downsample_factor: int = 1,
# ):
#     """
#     Real-time inference:

#     - MindRove always streams at 500 Hz (via BoardShim).
#     - We run the biquad filter stack at the raw rate.
#     - Optionally downsample in time by `downsample_factor` AFTER filtering, i.e.:

#         raw_len = model_input_len * downsample_factor           # samples from board
#         filtered_raw -> decimate -> L = model_input_len samples # to the model

#     - Extract CNN embedding from window(s)
#     - Optionally add 1–2 scalar features (asymmetry, magnitude)
#     - Auto-match scaler.n_features_in_ so it works with D, D+1 or D+2 models

#     Yields:
#         (prediction:int, avg_probabilities:np.ndarray)
#     """

#     if downsample_factor < 1:
#         raise ValueError("downsample_factor must be >= 1")

#     # -------------------------------------------------------
#     # Load feature extractor, scaler, and MLP classifier
#     # -------------------------------------------------------
#     fx = load_model(
#         feature_extractor_path,
#         custom_objects={
#             "MyMagnWarping": MyMagnWarping,
#             "MyScaling": MyScaling,
#         },
#     )
#     feature_extractor = Model(
#         inputs=fx.input,
#         outputs=fx.get_layer("dense_8").output,
#     )

#     with open(scaler_path, "rb") as fsc:
#         scaler = pickle.load(fsc)

#     with open(mlp_model_path, "rb") as fmlp:
#         mlp_model = pickle.load(fmlp)

#     n_expected = getattr(scaler, "n_features_in_", None)

#     # -------------------------------------------------------
#     # MindRove board setup
#     # -------------------------------------------------------
#     BoardShim.enable_dev_board_logger()
#     board_id = BoardIds.MINDROVE_WIFI_BOARD
#     params = MindRoveInputParams()
#     board = BoardShim(board_id, params)

#     # Queues / threading
#     data_queue: "Queue[np.ndarray]" = Queue()          # [B, L, 8, 1]
#     output_queue: "Queue[tuple[int, np.ndarray]]" = Queue()
#     stop_event = threading.Event()

#     # -------------------------------------------------------
#     # Local preprocessing: filter stack + optional decimation
#     # -------------------------------------------------------
#     def _preprocess_data(window_T: np.ndarray) -> np.ndarray:
#         """
#         window_T: [L_raw, 8] at raw sampling rate (~500 Hz)
#         Returns: [L_eff, 8] after filters (+ optional downsample).
#         """
#         # Apply filter stack sample-by-sample (per channel)
#         # filters: list of BiquadMultiChan configured at raw Fs.
#         L_raw, n_ch = window_T.shape

#         for i in range(L_raw):
#             for ch in range(n_ch):
#                 x = window_T[i, ch]
#                 for f in filters:
#                     x = f.process(x, ch)
#                 window_T[i, ch] = x

#         # Optional decimation in time
#         if downsample_factor > 1:
#             window_T = window_T[::downsample_factor, :]

#         return window_T

#     # -------------------------------------------------------
#     # Worker: consumes windows from data_queue and pushes predictions
#     # -------------------------------------------------------
#     def _inference_worker():
#         acc_probs = []

#         while not stop_event.is_set():
#             try:
#                 batch_windows = data_queue.get(timeout=1)  # [B, L, 8, 1]
#             except Empty:
#                 continue

#             # CNN feature extraction
#             feats = feature_extractor.predict(batch_windows, verbose=0)  # [B, D]

#             # Extra scalar features
#             asym = _side_asymmetry_feature(batch_windows)  # [B, 1]
#             mag = _magnitude_feature(batch_windows)        # [B, 1]
#             D = feats.shape[1]

#             if n_expected is None or n_expected == D + 2:
#                 feats_plus = np.hstack([feats, asym, mag])   # D+2
#             elif n_expected == D + 1:
#                 feats_plus = np.hstack([feats, asym])        # D+1
#             elif n_expected == D:
#                 feats_plus = feats                           # D
#             else:
#                 # Generic truncate/pad to n_expected
#                 fp = np.hstack([feats, asym, mag])
#                 if fp.shape[1] >= n_expected:
#                     feats_plus = fp[:, :n_expected]
#                 else:
#                     pad = np.zeros(
#                         (fp.shape[0], n_expected - fp.shape[1]),
#                         dtype=fp.dtype,
#                     )
#                     feats_plus = np.hstack([fp, pad])

#             feats_scaled = scaler.transform(feats_plus)
#             probs = mlp_model.predict_proba(feats_scaled)  # [B, n_cls]

#             for p in probs:
#                 acc_probs.append(p)

#             # Agg over `batch_size` windows
#             while len(acc_probs) >= batch_size:
#                 avg = np.mean(acc_probs[:batch_size], axis=0)
#                 del acc_probs[:batch_size]

#                 top = int(np.argmax(avg))
#                 if float(avg[top]) >= float(prediction_threshold):
#                     output_queue.put((top, avg))

#     # -------------------------------------------------------
#     # Main streaming loop
#     # -------------------------------------------------------
#     raw_len = model_input_len * max(1, downsample_factor)

#     try:
#         board.prepare_session()
#         board.start_stream(450000)

#         worker = threading.Thread(target=_inference_worker, daemon=True)
#         worker.start()

#         batch = []

#         while not stop_event.is_set():
#             if board.get_board_data_count() < raw_len:
#                 continue

#             # raw: [channels, T_raw]
#             raw = board.get_board_data(raw_len)
#             emg = raw[:8]  # [8, T_raw]

#             # [T_raw, 8] -> filter + (optional) downsample -> [L_eff, 8]
#             emg_proc_T = _preprocess_data(emg.T)

#             # Sanity: after decimation, we expect L_eff == model_input_len
#             if emg_proc_T.shape[0] != model_input_len:
#                 # If not, either adjust model_input_len or downsample_factor
#                 # in config to keep them consistent.
#                 # For now, we skip this window rather than crash.
#                 continue

#             # [L_eff, 8] -> [L_eff, 8, 1]
#             window = np.expand_dims(emg_proc_T.T, axis=2).astype(np.float32)
#             batch.append(window)

#             if len(batch) == batch_size:
#                 data_queue.put(np.array(batch))
#                 batch = []

#             # Emit any ready predictions
#             while not output_queue.empty():
#                 yield output_queue.get()

#     except Exception as e:
#         raise RuntimeError(f"Error during real-time inference: {e}") from e

#     finally:
#         stop_event.set()
#         try:
#             board.stop_stream()
#         except Exception:
#             pass
#         try:
#             board.release_session()
#         except Exception:
#             pass 
#         ""
