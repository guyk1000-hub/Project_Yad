# -*- coding: utf-8 -*-
import os
import sys
import pickle
from threading import Event, Thread
from queue import Queue
import cv2
import json
from pathlib import Path
import numpy as np

# --- Make 'training_data' importable by adding repo root (two levels up) ---
# .../Project_Yad/training_data/windows_training_code/example.py (this file)
# We need to add .../Project_Yad to sys.path so 'training_data' is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from training_data.record import record_gestures
from training_data.fine_tune import fine_tune_model
from training_data.inference import real_time_inference, show_image_for_prediction
from training_data.utils import FilterTypes, BiquadMultiChan, send_output_to_socket

# (Optional) quieter TF logs
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass


# ---------------- Config loading (also looks in assets/) ----------------
def load_config():
    here = Path(__file__).resolve()
    pkg_dir = here.parents[1]           # .../training_data
    repo_root = pkg_dir.parent          # .../Project_Yad

    candidates = [
        pkg_dir / "config.json",                # Project_Yad/training_data/config.json
        repo_root / "config.json",              # Project_Yad/config.json
        repo_root / "assets" / "config.json",   # Project_Yad/assets/config.json  <-- your case
        Path.cwd() / "config.json",             # current working directory
    ]

    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        tried = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError("Configuration file not found. Looked for:\n  - " + tried)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_dir = config_path.parent
    for key in ["data_path", "feature_extractor_path", "mlp_model_path", "scaler_path", "gesture_image_path"]:
        if key in config and isinstance(config[key], str):
            config[key] = os.path.abspath(os.path.join(config_dir, config[key]))

    print(f"[cfg] Using: {config_path}")
    return config


def main():
    config = load_config()
    # Paths from config
    data_path = config["data_path"]
    feature_extractor_path = config["feature_extractor_path"]
    mlp_model_path = config["mlp_model_path"]
    scaler_path = config["scaler_path"]
    gesture_image_path = config["gesture_image_path"]
    

    # Flags 
    record = config["record"]
    fine_tune = config["fine_tune"]
    show_predicted_image = config["show_predicted_image"]
    send_to_socket = config["send_to_socket"]
    
    # Skip gestures
    skip_gestures = []
    # Sampling rate and model input length
    sampling_rate = 500
    model_input_len = 100

    # Define filters
    filters = [
        BiquadMultiChan(8, FilterTypes.bq_type_highpass, 4.5 / sampling_rate, 0.5, 0.0), # Dc filter
        BiquadMultiChan(8, FilterTypes.bq_type_notch, 50.0 / sampling_rate, 4.0, 0.0), # 50 Hz noise
        BiquadMultiChan(8, FilterTypes.bq_type_lowpass, 100.0 / sampling_rate, 0.5, 0.0), 
    ]

    # Step 1: Record Gestures
    print("Step 1: Recording Gestures")
    if record:
        record_gestures(
            filters=filters,
            data_path=data_path,
            gesture_image_path=gesture_image_path,
            skip_gestures=skip_gestures,
            gestures_repeat=3,
            recording_time_sec=6,
            sampling_rate=sampling_rate,
            model_input_len=model_input_len,
            overlap_frac=model_input_len // 10,
        )
    else:
        print("Skipping gesture recording.")

    # Step 2: Fine-Tune the Model
    print("Step 2: Fine-Tuning the Model")
    
    if fine_tune:
        with open(data_path, "rb") as f:
            recorded_data, recorded_labels = pickle.load(f)

        scaler, val_accuracy = fine_tune_model(
            feature_extractor_path=feature_extractor_path,
            recorded_data=recorded_data,
            recorded_labels=recorded_labels,
            scaler_path=scaler_path,
            mlp_model_path=mlp_model_path,
        )
        print(f"Fine-tuning complete. Validation accuracy: {val_accuracy:.2f}")
    else:
        print("Skipping fine-tuning.")

    # Step 3: Real-Time Inference
    print("Step 3: Starting Real-Time Inference")
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
            gyro_threshold=500,
            prediction_threshold=0.4,
            batch_size=5,
        ):
            #print(f"Prediction: {prediction}, Probabilities: {probabilities.round(4)}")
            print(f"Predicted gesture: {prediction}")
            if send_to_socket:
                output_queue.put(prediction)
            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, skip_gestures)
            
    except KeyboardInterrupt:
        print("Real-time inference stopped.")
        if send_to_socket:
            stop_event.set()
            socket_thread.join()
        # close all windows
        cv2.destroyAllWindows()

    print("Example script completed.")

if __name__ == "__main__":
    main()

