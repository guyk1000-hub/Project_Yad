# training_data/windows_training_code/inference_example.py
# -*- coding: utf-8 -*-
"""
Real-time EMG inference launcher for the package layout:

training_data/
  ├─ inference.py
  ├─ utils.py
  └─ windows_training_code/
       └─ inference_example.py  (this file)

Run from repo root:
    python -m training_data.windows_training_code.inference_example
"""

import json
import os
import sys
from pathlib import Path
from threading import Event, Thread
from queue import Queue

import cv2
import numpy as np

# Package-relative imports (no sys.path hacks needed)
# Add TWO levels up (Project_Yad) to sys.path so 'training_data' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

from training_data.inference import real_time_inference, show_image_for_prediction
from training_data.utils import FilterTypes, BiquadMultiChan, send_output_to_socket


# (Optional) quiet TensorFlow logs
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass


def load_config():
    """
    Search for config.json in several sensible locations:

    - training_data/config.json
    - Project_Yad/config.json
    - Project_Yad/assets/config.json
    - CWD/config.json
    - or explicit CONFIG_PATH env var
    """
    here = Path(__file__).resolve()
    pkg_dir = here.parents[1]      # .../training_data
    repo_root = pkg_dir.parent     # .../Project_Yad

    candidates = [
        pkg_dir / "config.json",
        repo_root / "config.json",
        repo_root / "assets" / "config.json",   # <-- your main case
        Path.cwd() / "config.json",
    ]

    explicit = os.environ.get("CONFIG_PATH")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            candidates.insert(0, p)

    config_path = next((p for p in candidates if p.is_file()), None)

    if config_path is None:
        tried = "\n - ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Configuration file not found.\nLooked for:\n - " + tried
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve paths relative to the config location
    config_dir = config_path.parent
    for key in [
        "data_path",
        "feature_extractor_path",
        "mlp_model_path",
        "scaler_path",
        "gesture_image_path",
    ]:
        if key in config and isinstance(config[key], str):
            config[key] = str((config_dir / config[key]).resolve())

    print(f"[cfg] Using: {config_path}")
    return config


def main():
    config = load_config()

    feature_extractor_path = config["feature_extractor_path"]
    mlp_model_path = config["mlp_model_path"]
    scaler_path = config["scaler_path"]
    gesture_image_path = config["gesture_image_path"]

    show_predicted_image = bool(config.get("show_predicted_image", False))
    send_to_socket = bool(config.get("send_to_socket", False))

    # Raw sampling rate from MindRove (keep this at 500 Hz)
    sampling_rate = int(config.get("sampling_rate", 500))

    # Effective window length AFTER any downsampling
    model_input_len = int(config.get("model_input_len", 100))

    # Optional downsampling factor (1 = no decimation, 2 = 500 -> 250 Hz effective)
    downsample_factor = int(config.get("downsample_factor", 1))

    # Filters designed at RAW sampling_rate (500 Hz)
    filters = [
        # High-pass ~4.5 Hz (remove DC / slow drift)
        BiquadMultiChan(
            8,
            FilterTypes.bq_type_highpass,
            4.5 / sampling_rate,
            0.5,
            0.0,
        ),
        # Notch at 50 Hz (mains)
        BiquadMultiChan(
            8,
            FilterTypes.bq_type_notch,
            50.0 / sampling_rate,
            4.0,
            0.0,
        ),
        # Low-pass ~100 Hz (anti-aliasing for 250 Hz when downsample_factor=2)
        BiquadMultiChan(
            8,
            FilterTypes.bq_type_lowpass,
            100.0 / sampling_rate,
            0.5,
            0.0,
        ),
    ]

    print("Starting Real-Time Inference...")
    print(f"  sampling_rate (raw)  : {sampling_rate} Hz")
    print(f"  downsample_factor    : {downsample_factor}")
    print(f"  model_input_len (eff): {model_input_len} samples")

    # Optional socket thread for sending predictions
    socket_stop_event = Event()
    output_queue = Queue()

    socket_thread = None
    if send_to_socket:
        socket_thread = Thread(
            target=send_output_to_socket,
            args=(socket_stop_event, output_queue),
            daemon=True,
        )
        socket_thread.start()

    try:
        for prediction, probabilities in real_time_inference(
            feature_extractor_path=feature_extractor_path,
            mlp_model_path=mlp_model_path,
            scaler_path=scaler_path,
            filters=filters,
            model_input_len=model_input_len,
            gyro_threshold=int(config.get("gyro_threshold", 90)),
            prediction_threshold=float(config.get("prediction_threshold", 0.6)),
            batch_size=int(config.get("batch_size", 8)),
            downsample_factor=downsample_factor,
        ):
            # Print prediction + probabilities
            probs_str = (
                np.round(probabilities, 3).tolist()
                if "np" in globals()
                else "…"
            )
            print(f"Pred: {prediction}  Probs: {probs_str}")

            # Optionally send to socket (for your visualizer)
            if send_to_socket:
                output_queue.put(prediction)

            # Optionally show gesture image
            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, [])

    except KeyboardInterrupt:
        print("Inference stopped by user.")

    finally:
        if send_to_socket:
            socket_stop_event.set()
            if socket_thread is not None:
                socket_thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
