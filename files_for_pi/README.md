# Running PI Example Scripts

 <br>

## **MAKE SURE YOU ARE RUNNING .VENV**
Make sure you have Python 3.10 installed. (Other versions are also compatible: Python >=3.7 and <3.11).<br>


Once installed, create a virtual environment:
```bash
python3.10 -m venv venv10
```

### **Activate the Virtual Environment**
**Windows (PowerShell):**
```bash
venv10\Scripts\activate
```

### **Verify the Python Version**
```bash
python --version
```
It should show:
```bash
Python 3.10.x
```

 <br>

## Install Dependencies**
Inside the virtual environment, install all required dependencies:

```bash
pip install -r requirements.txt
```
 <br>
 

## Modify config.json to Customize Functionality**
The config.json file (located in the root folder) controls the behavior of the system.
You can modify the boolean (true/false) values to enable or disable certain functions.

Example: Default config.json
```json
{
    "data_path": "/home/user/Yad/NaviFlame/naviflame/data/recorded_gestures_guy.pkl",
    "feature_extractor_path": "/home/user/Yad/NaviFlame/naviflame/models/og_fine_tune_guy.h5",
    "mlp_model_path": "/home/user/Yad/NaviFlame/naviflame/models/mlp_model_guy.pkl",
    "scaler_path": "/home/user/Yad/NaviFlame/naviflame/models/scaler_guy.pkl",
    "gesture_image_path": "/home/user/Yad/NaviFlame/naviflame/gestures",
    "record": false,
    "fine_tune": false,
    "show_predicted_image": true,
    "send_to_socket": true
}
```
### **How to Change Functionality**
- **Enable Gesture Recording:**<br>
Change `"record": false,` to `"record": true,`

- **Enable Fine-Tuning of the Model:**<br>
Change `"fine_tune": false,` to `"fine_tune": true,`
- **Disable Gesture Image Display:**<br>
Change `"show_predicted_image": true,` to `"show_predicted_image": false,`
- **Disable Sending Data to Socket:**<br>
Change `"send_to_socket": true,` to `"send_to_socket": false,`<br><br>

After modifying the `config.json` file, save the changes, then proceed to run the example scripts.

 
 <br>

## **Step 4: Run the Example Scripts**
Navigate to the examples/ folder:

```bash
cd examples
```
Now, you can run an example script:

```bash
python example.py
python inference_example.py
```

If running from outside the examples folder from the root path, ensure you specify the correct path:

```bash
python examples/example.py
python examples/inference_example.py
``` 
<br>