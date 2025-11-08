# -*- coding: utf-8 -*-
"""
Real-time EMG inference launcher for the package layout:
training_data/
  ├─ inference.py
  ├─ utils.py
  └─ windows_training_code/
       └─ run_realtime.py  (this file)

Run from repo root:
    python -m training_data.windows_training_code.run_realtime
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from threading import Event, Thread
from queue import Queue

# Package-relative imports (no sys.path hacks needed)
import os, sys
# add TWO levels up (Project_Yad) to sys.path so 'training_data' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from training_data.inference import real_time_inference, show_image_for_prediction
from training_data.utils import FilterTypes, BiquadMultiChan, send_output_to_socket


# (Optional) quiet TensorFlow logs
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass



def load_config():
    here = Path(__file__).resolve()
    pkg_dir = here.parents[1]           # .../training_data
    repo_root = pkg_dir.parent          # .../Project_Yad

    candidates = [
        pkg_dir / "config.json",        # training_data/config.json
        repo_root / "config.json",      # Project_Yad/config.json
        repo_root / "assets" / "config.json",  # Project_Yad/assets/config.json  <-- your case
        Path.cwd() / "config.json",     # CWD/config.json
    ]

    # Allow explicit override via env or CLI-style var
    explicit = os.environ.get("CONFIG_PATH")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            candidates.insert(0, p)

    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        tried = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError("Configuration file not found. Looked for:\n  - " + tried)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_dir = config_path.parent
    for key in ["data_path", "feature_extractor_path", "mlp_model_path", "scaler_path", "gesture_image_path"]:
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

    show_predicted_image = config["show_predicted_image"]
    send_to_socket = config["send_to_socket"]

    sampling_rate = int(config.get("sampling_rate", 500))
    model_input_len = int(config.get("model_input_len", 100))

    filters = [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0),
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0),
    ]

    print("Starting Real-Time Inference...")

    if send_to_socket:
        stop_event = Event()
        output_queue = Queue()
        socket_thread = Thread(target=send_output_to_socket, args=(stop_event, output_queue))
        socket_thread.start()

    try:
        for prediction, probabilities in real_time_inference(
            feature_extractor_path=feature_extractor_path,
            mlp_model_path=mlp_model_path,
            scaler_path=scaler_path,
            filters=filters,
            model_input_len=model_input_len,
            gyro_threshold=int(config.get("gyro_threshold", 90)),   # <-- TUNE: 60–120
            prediction_threshold=float(config.get("prediction_threshold", 0.6)),
            batch_size=int(config.get("batch_size", 8)),
        ):
            print(f"Pred: {prediction}  Probs: {np.round(probabilities, 3) if 'np' in globals() else '…'}")

            if send_to_socket:
                output_queue.put(prediction)

            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, [])

    except KeyboardInterrupt:
        print("Inference stopped.")
        if send_to_socket:
            stop_event.set()
            socket_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
