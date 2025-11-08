# Project Yad – EMG-Based Gesture Recognition

Project Yad is a collaborative EMG-based gesture recognition system built by **Guy Katabi** and **Tomer Ohayon**.  
It uses EMG signals from an armband (e.g., MindRove) and processes them in real-time on a Raspberry Pi to classify hand gestures.

## 🔧 Features

- Real-time EMG acquisition and filtering  
- Feature extraction (e.g., MAV, RMS, WL, ZC, SSC, WAMP, etc.)  
- MLP / SVM-based gesture classification  
- Configurable paths and parameters via `assets/config.json`  
- Designed to run on Raspberry Pi 5

## 📂 Project Structure

```text
Project_Yad/
├── assets/
│   ├── config.json
│   └── models/          # (optional, typically kept local or small demo models)
├── realtime/
│   ├── feature_extractor.py
│   ├── filters.py
│   ├── mlp_inference.py
│   └── ...
├── training_data/       # raw/processed data and experiments (not all tracked in git)
├── .gitignore
├── requirements.txt
└── README.md


how to start:

git clone https://github.com/guyk1000-hub/Project_Yad.git
cd Project_Yad

python3 -m venv venvPI
source venvPI/bin/activate   # On Linux/macOS

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


make sure assets/config.json exists and is configured correctly

source venvPI/bin/activate
python main_pi.py

Authors:

Guy Katabi – @guyk1000-hub

Tomer Ohayon – @tomerohayon77

Acknowledgements:
This project was inspired by and partially adapted from:

MindRove / NaviFlame
 – EMG data streaming and control interface

tomerohayon77 / mindrove-emg-classifier
 – feature extraction and classification pipeline

We thank the authors of these open-source projects for their contributions, which helped guide parts of this work.

📜 License

This project is licensed under the MIT License (or another license you choose).
