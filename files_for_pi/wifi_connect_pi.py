import subprocess
import time
from multiprocessing import Process, Manager

SSID = "MindRove_ARB_0a25b8"
PASSWORD = "#mindrove"


def run_cmd(cmd):
    """Run shell command and return stdout as string."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(result.stdout.strip())
    if result.stderr:
        print("[stderr]", result.stderr.strip())
    return result.returncode

def get_current_wifi():
    """Returns the SSID of the currently connected Wi-Fi network."""
    result = subprocess.run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if line.startswith("yes:"):
            return line.split(":")[1]
    return None

def disconnect_wifi():
    """Disconnects from the current Wi-Fi network."""
    subprocess.run(["nmcli", "connection", "down", SSID], capture_output=True, text=True)

def connect_to_wifi():
    """Connects to the desired Wi-Fi network."""
    result = subprocess.run(["nmcli", "device", "wifi", "connect", SSID, "password", PASSWORD], capture_output=True, text=True)
    return "successfully activated" in result.stdout

def wifi_connect_PI(shared_data):
    """Ensures the device stays connected to the correct Wi-Fi."""
    while True:
        current_wifi = get_current_wifi()
        if current_wifi != SSID:
            print(f"Connecting to {SSID}...")
            disconnect_wifi()
            time.sleep(3)  # Allow time to disconnect
            if connect_to_wifi():
                print("Connected successfully!")
                shared_data['connected'] = 1
            else:
                
                shared_data['connected'] = 0
                try:
                    print("difficulties-trying alt way")
                    # 1. Show existing connections matching MindRove
                    run_cmd(f'nmcli con show | grep -F "{SSID}"')

                    # 2. Delete the old profile if it exists (ignore errors)
                    run_cmd(f'sudo nmcli con delete "{SSID}" 2>/dev/null || true')

                    # 3. Connect to the MindRove network
                    ret = run_cmd(f'sudo nmcli dev wifi connect "{SSID}" password "{PASSWORD}"')
                    if ret == 0:
                        print(f"[wifi] Connected successfully to {SSID}")
                        shared_data['connected'] = 1
                    else:
                        print(f"[wifi] Failed to connect to {SSID}")
                        shared_data['connected'] = 0

                except Exception as e:
                    print("Failed to connect.")

                
        else:
            print(f"Already connected to {SSID}")
            shared_data['connected'] = 1
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        shared_data['connected'] = 0  # Initially not connected

        p = Process(target=wifi_connect_PI, args=(shared_data,))
        p.start()

        try:
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            print("[Wi-Fi] Stopped by user.")