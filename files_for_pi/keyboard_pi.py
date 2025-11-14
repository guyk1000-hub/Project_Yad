# keyboard_pi.py
import time
from multiprocessing import Manager
import keyboard  # For keyboard event detection

def keyboard_classify_PI(shared_data):
    while True:
            event = keyboard.read_event()  # Waits for a keyboard event
            if event.event_type == keyboard.KEY_DOWN and shared_data['start'] == 1:  # Check if a key is pressed down
                shared_data['move'] = event.name
                shared_data['action'] = 1
                print("You pressed - ", event.name)

            time.sleep(0.5)  # Update every half second

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()

        from multiprocessing import Process

        p = Process(target=keyboard_classify_PI, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[classify] Stopped by user.")