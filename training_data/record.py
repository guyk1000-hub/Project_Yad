import pickle
import cv2
import os
import logging
import time
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def show_image(i, gesture_image_path):
    """Display the image for the gesture (non-blocking)."""
    image_file = os.path.join(gesture_image_path, f"g{i}.png")
    if os.path.exists(image_file):
        img = cv2.imread(image_file)
        if img is None:
            logging.warning(f"Error reading image '{image_file}'.")
            return
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
        name = "Current Gesture"
        cv2.destroyAllWindows()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)  # Always on top
        cv2.imshow(name, img)
        cv2.resizeWindow(name, img.shape[1], img.shape[0])
        cv2.waitKey(1)
    else:
        logging.error(f"Image file not found: {image_file}")


def preprocess_data(data, filters, downsample_factor: int = 1):
    """
    Apply filter stack and optional downsampling.

    Input:
        data              : [L_raw, 8]  (raw samples at sampling_rate, e.g. 500 Hz)
        filters           : list of BiquadMultiChan
        downsample_factor : 1 = no decimation, 2 = keep every 2nd sample, etc.

    Output:
        processed_data    : [L_eff, 8] after filtering (+ optional decimation)
                            where L_eff ~= L_raw / downsample_factor
    """
    # 1) filtering at raw sampling rate
    for i in range(len(data)):
        for ch in range(data.shape[1]):
            for filter_ in filters:
                data[i, ch] = filter_.process(data[i, ch], ch)

    # 2) optional decimation in time
    if downsample_factor is not None and downsample_factor > 1:
        data = data[::downsample_factor, :]

    return data


def record_gestures(
    filters,
    data_path,
    gesture_image_path="gestures_tomer",
    skip_gestures=None,
    gestures_repeat=3,
    recording_time_sec=6,
    sampling_rate=500,
    model_input_len=100,
    overlap_frac=10,          # step size in *effective* samples
    num_gestures=6,
    downsample_factor: int = 1,
):
    """
    Record gestures from the MindRove board and save them to a file.

    - The board always streams at 'sampling_rate' (e.g. 500 Hz).
    - Filters are applied at this raw rate.
    - If downsample_factor > 1, the data is decimated after filtering:
        L_eff = L_raw / downsample_factor
    - model_input_len is the window length *after* downsampling
      (e.g. 50 samples at 250 Hz when downsample_factor=2).

    New behavior:
    After each recording, user can:
      [ENTER/a] accept and keep the windows
      [r]       redo (discard that recording and re-record same gesture)
      [q]       quit early
    """
    if skip_gestures is None:
        skip_gestures = []

    recorded_data = []
    recorded_labels = []
    fft_warmup = 10
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    params = MindRoveInputParams()
    board_shim = BoardShim(board_id, params)

    # guard: overlap_frac must be int >= 1
    try:
        overlap_frac = int(overlap_frac)
    except Exception:
        overlap_frac = 10
    if overlap_frac < 1:
        overlap_frac = 1

    def _prompt_after_record(num_windows: int, gesture_id: int) -> str:
        """
        Ask the user what to do after a recording.
        Returns one of: 'accept', 'redo', 'quit'
        """
        choice = input(
            f"Recorded {num_windows} windows for gesture {gesture_id}. "
            f"[ENTER/a]=accept, [r]=redo, [q]=quit: "
        ).strip().lower()
        if choice in ("", "a"):
            return "accept"
        if choice == "r":
            return "redo"
        if choice == "q":
            return "quit"
        # default if user typed something weird
        return "redo"

    try:
        # Prepare and start streaming from the MindRove board
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        logging.info("Starting streaming from MindRove board.")

        # Warm-up phase to ensure the board is ready
        start_time = time.time()
        while time.time() - start_time < fft_warmup:
            if board_shim.get_board_data_count() > 0:
                raw_data = board_shim.get_board_data(sampling_rate)
                emg_data = raw_data[:8]
                _ = preprocess_data(emg_data.T, filters, downsample_factor)

        # Start recording gestures
        for repeat in range(gestures_repeat):
            for gesture_id in range(num_gestures):
                if gesture_id in skip_gestures:
                    continue

                accepted = False

                while not accepted:
                    show_image(gesture_id, gesture_image_path)

                    start_cmd = input(
                        f"{repeat}/{gestures_repeat} - Perform gesture id {gesture_id}. "
                        f"Press ENTER to start recording for {recording_time_sec} seconds "
                        f"(or 'q' to quit): "
                    ).strip().lower()

                    if start_cmd == "q":
                        logging.info("User requested quit during pre-record prompt.")
                        raise KeyboardInterrupt

                    gesture_data = []
                    board_shim.get_board_data()  # clear buffer

                    # Wait until the desired number of raw samples are received
                    need = recording_time_sec * sampling_rate
                    while board_shim.get_board_data_count() < need:
                        pass

                    # Collect the raw data at sampling_rate (e.g. 500 Hz)
                    raw_data = board_shim.get_board_data(need)
                    if raw_data.shape[0] < 8:
                        logging.error("Board returned fewer than 8 channels; re-recording.")
                        continue

                    emg_data = raw_data[:8]  # Only EMG data [8, T_raw]

                    # Filter at raw Fs and downsample if requested
                    processed_data = preprocess_data(emg_data.T, filters, downsample_factor)  # [T_eff, 8]

                    # Safety: only slice if we have at least one full window
                    T_eff = len(processed_data)
                    if T_eff < model_input_len:
                        logging.warning(
                            f"Not enough samples for a full window (have {T_eff}, need {model_input_len}). Re-recording."
                        )
                        continue

                    for i in range(0, T_eff - model_input_len, overlap_frac):
                        sample = processed_data[i:i + model_input_len]      # [L_eff, 8]
                        # Keep orientation as before: [8, L, 1]
                        sample = np.expand_dims(sample.T, axis=2).astype(np.float32)
                        gesture_data.append(sample)

                    action = _prompt_after_record(len(gesture_data), gesture_id)

                    if action == "quit":
                        logging.info("User requested quit after recording.")
                        raise KeyboardInterrupt

                    if action == "redo":
                        logging.info(f"Redoing gesture {gesture_id} (discarding last recording).")
                        # IMPORTANT: we do not append gesture_data, so effectively "deleted"
                        continue

                    # accept
                    recorded_data.extend(gesture_data)
                    recorded_labels.extend([gesture_id] * len(gesture_data))
                    logging.info(f"Accepted {len(gesture_data)} windows for gesture {gesture_id}")
                    accepted = True

        # Save the recorded data to a file
        with open(data_path, "wb+") as f:
            pickle.dump((recorded_data, recorded_labels), f)

        logging.info(
            f"Gestures successfully recorded and saved to {data_path} "
            f"({len(recorded_data)} windows, labels {len(recorded_labels)})."
        )

    except KeyboardInterrupt:
        logging.info("Recording interrupted by user. Saving what was accepted so far...")
        # Save partial accepted data as well (optional but useful)
        try:
            with open(data_path, "wb+") as f:
                pickle.dump((recorded_data, recorded_labels), f)
            logging.info(
                f"Partial data saved to {data_path} "
                f"({len(recorded_data)} windows, labels {len(recorded_labels)})."
            )
        except Exception as e:
            logging.error(f"Failed saving partial data: {e}")

    except Exception as e:
        logging.error(f"Error recording gestures: {e}")
        raise

    finally:
        cv2.destroyAllWindows()
        try:
            board_shim.stop_stream()
        except Exception:
            pass
        try:
            board_shim.release_session()
        except Exception:
            pass
        logging.info("Disconnected from MindRove board.")
