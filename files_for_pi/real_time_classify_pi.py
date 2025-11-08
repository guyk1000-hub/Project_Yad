from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from filtering_PI import apply_filters
from sklearn.svm import SVC
import joblib
import numpy as np
from feature_extraction_PI import extract_features
import time
from multiprocessing import Manager

model_path = r"/home/raspi5/mindrove-emg-classifier/SVM_models/svm_model_guy.pkl"

svm_model = joblib.load(model_path)
check_every = 20 #number of samples
gyro_threshold = 100 # used for decide if we are in rest mode or no
move_min_size = 40  # in samples
def handeling_nans(array):
    array = np.nan_to_num(array, nan=0)
    return array
def normalize_signals(emg_data):
    min_val = np.min(emg_data, axis=0, keepdims=True)
    max_val = np.max(emg_data, axis=0, keepdims=True)
    normalized_signals = (emg_data - min_val) / (max_val - min_val)
    return normalized_signals
def extracting_features(emg_data,sampling_rate):
    # Apply filters to EMG signals
    filtered_emg_data = apply_filters(emg_data, sampling_rate)
    # Normalize the EMG signals
    #normalized_emg = normalize_signals(filtered_emg_data)
    # Extract features
    return extract_features(filtered_emg_data, sampling_rate)

def movement_from_model(emg_data,sampling_rate):
    features_emg_data = extracting_features(emg_data,sampling_rate)
    features_array = np.array(list(features_emg_data.values())).reshape(1, -1)
    features_array = handeling_nans(features_array)
    return svm_model.predict(features_array)


def real_time_classify_per_move_PI(shared_data):
    while True:
        if shared_data['connected'] == 1:
            BoardShim.enable_dev_board_logger()
            params = MindRoveInputParams()
            board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
            try:
                # Prepare session
                board_shim.prepare_session()
                board_shim.start_stream()

                emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
                gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
                sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
                print(f"[dbg] emg={emg_channels} gyro={gyro_channels} fs={sampling_rate}")


                emg_channels = BoardShim.get_emg_channels(board_shim.board_id)
                gyro_channels = BoardShim.get_gyro_channels(board_shim.board_id)
                sampling_rate = BoardShim.get_sampling_rate(board_shim.board_id)
                time.sleep(3)
                move_data = np.empty((8,0))
                flag_move = 0 # used for symbol whether we are in the middle of movement or not
                print("start classifying")
                while True:
                    if board_shim.get_board_data_count() >= check_every: ## NOT HAPPENING WITH CHECK EVERY #check the new data every period of samples that we chose
                        new_data = board_shim.get_board_data()# get only the newest N samples
                        emg_data = np.array(new_data[emg_channels])
                        gyro_data = np.array(new_data[gyro_channels])

                        if np.any(np.abs(gyro_data) > gyro_threshold):# if the gyro value is bigger than the threshold we chose , start saving the move's data
                            move_data = np.hstack((move_data, emg_data))
                            if flag_move == 0:
                                flag_move = 1
                        elif flag_move == 1:# if the gyro values is lower than the threshold we chose, if it is the end of a movement we will check the classify of this movement
                            movement = movement_from_model(move_data, sampling_rate)
                            if movement == 0:
                                shared_data['move'] = 'rest'
                                print('rest')
                            if movement == 1:
                                shared_data['move'] = 'close'
                                print('close')
                            elif movement == 2:
                                shared_data['move'] = 'open'
                                print('open')
                            elif movement == 3:
                                shared_data['move'] = 'right'
                                print('right')
                            elif movement == 4:
                                shared_data['move'] = 'left'
                                print('left')
                            if move_data.shape[1] > move_min_size:
                                print("the move is ", movement)
                                shared_data['action'] = 1
                            move_data = np.empty((8, 0))
                            flag_move = 0
            except Exception as e:
                print(f"Error: {e}")
            finally:
                if board_shim.is_prepared():
                    board_shim.release_session()


if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=real_time_classify_per_move_PI, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[commands] Stopped by user.")