import os
import sys
import json
import cv2
import time
from threading import Event, Thread
from queue import Queue
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training_data.inference_mix import real_time_inference, show_image_for_prediction
from training_data.utils import FilterTypes, BiquadMultiChan, send_output_to_socket
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


GESTURE_MAP = {
    0: 'rest',
    1: 'close',
    2: 'open',
    3: 'right',
    4: 'left'
}


# config loading
def load_config():
    """
    Load the configuration file (config.json) from common locations.
    Looks for:
    1. <project_root>/assets/config.json
    2. <this_script_directory>/config.json
    3. Current working directory/config.json
    """

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    possible_paths = [
        os.path.join(base_dir, "assets", "config.json"),
        os.path.join(os.path.dirname(__file__), "config.json"),
        os.path.abspath("config.json")
    ]

    config_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(f"config.json not found in: {possible_paths}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_dir = os.path.dirname(config_path)
    for key in [
        "data_path_pi",
        "feature_extractor_path",
        "mlp_model_path_pi",
        "scaler_path_pi",
        "gesture_image_path_pi"
    ]:
        if key in config and isinstance(config[key], str):
            config[key] = os.path.abspath(os.path.join(config_dir, config[key]))

    return config


def main(shared_data=None):
    if shared_data is not None:
        print("[realtime] Waiting for WiFi / armband connection...")
        while True:
            try:
                if shared_data.get('connected', 0) == 1:
                    print("[realtime] WiFi connected, starting inference.")
                    break
            except Exception:
                pass
            time.sleep(0.5)

    config = load_config()

    feature_extractor_path = config["feature_extractor_path_pi"]
    mlp_model_path = config["mlp_model_path_pi"]
    scaler_path = config["scaler_path_pi"]
    gesture_image_path = config["gesture_image_path_pi"]

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
        # socket_thread = Thread(target=send_output_to_socket, args=(stop_event, output_queue))
        # socket_thread.start()

    try:
        for prediction, probabilities in real_time_inference(
            feature_extractor_path=feature_extractor_path,
            mlp_model_path=mlp_model_path,
            scaler_path=scaler_path,
            filters=filters,
            model_input_len=model_input_len,
            gyro_threshold=int(config.get("gyro_threshold", 90)),
            prediction_threshold=float(config.get("prediction_threshold", 0.7)),
            batch_size=int(config.get("batch_size", 6)),
        ):
            print(f"Pred: {prediction}  Probs: {np.round(probabilities, 3) if 'np' in globals() else '…'}")

            if shared_data is not None:
                print(GESTURE_MAP.get(prediction, 'rest'))
                shared_data['move'] = GESTURE_MAP.get(prediction, 'rest')
                shared_data['action'] = 1

            if send_to_socket:
                output_queue.put(prediction)

            if show_predicted_image:
                show_image_for_prediction(prediction, gesture_image_path, [])

    except KeyboardInterrupt:
        print("Inference stopped.")
        if send_to_socket:
            stop_event.set()
            # socket_thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

