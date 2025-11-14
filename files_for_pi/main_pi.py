# main_PI.py
from multiprocessing import Process, Manager
import time

import wifi_connect_pi
import keyboard_pi
import basic_commands_pi
import realtime_pi_classify


if __name__ == "__main__":
    with Manager() as manager:
        # Shared memory between all processes
        shared_data = manager.dict()
        shared_data['move'] = 'NONE'
        shared_data['action'] = 0
        shared_data['start'] = 0
        shared_data['connected'] = 0   # WiFi/board connection state

        # ---------------------------------------------------------
        #  PROCESS 1 — WiFi watchdog (runs forever)
        # ---------------------------------------------------------
        p1 = Process(
            target=wifi_connect_pi.wifi_connect_PI,
            args=(shared_data,)
        )

        # ---------------------------------------------------------
        #  PROCESS 2 — Keyboard movement listener (optional)
        # ---------------------------------------------------------
        # p2 = Process(
        #     target=keyboard_pi.keyboard_classify_PI,
        #     args=(shared_data,)
        # )

        # ---------------------------------------------------------
        #  PROCESS 3 — Basic command handler
        # ---------------------------------------------------------
        # p3 = Process(
        #     target=basic_commands_pi.commands_PI,
        #     args=(shared_data,)
        # )

        # ---------------------------------------------------------
        #  PROCESS 4 — Real-time EMG classification
        # ---------------------------------------------------------
        p4 = Process(
            target=realtime_pi_classify.main,
            args=(shared_data,)
        )

        # Start processes
        p1.start()
        # p2.start()
        # p3.start()
        p4.start()

        print("All processes started.\nPress Ctrl-C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping all processes...")

            p1.terminate()
            # p2.terminate()
            # p3.terminate()
            p4.terminate()

            print("All processes stopped.")
