# AutoML Virtual Assistant

A voice-activated AutoML assistant designed to securely verify users via biometric and password authentication, and process raw audio commands into clean actionable text. 

## Structure

Modules covered in this repository:
- **Module 1**: User Identification / Verification (Password, Face Recognition, Voice Recognition).
- **Module 2**: Wake Word Detection.
- **Module 3**: Text Input Handling.
- **Module 4**: Audio Capture.
- **Module 5**: Speech-to-Text Transcription.

## Running Locally Without Docker

To run the app locally using Python and `venv`:
```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit demo
streamlit run app.py
```

## Running With Docker

You can containerize the app. Building the container handles all the system-level C++ library dependencies (like `ffmpeg`, `libportaudio2` and `libgl1-mesa-glx`) so the audio capture, Whisper transcription, and OpenCV face matching runs out of the box.

```bash
# 1. Build the Docker image
docker build -t automl-assistant .

# 2. Run the Docker container (Binding port 8501)
docker run -p 8501:8501 --name automl-instance automl-assistant
```
Then navigate to `http://localhost:8501` to use the UI.

## File Organization & Privacy 

User privacy is explicitly respected:
* Data is stored locally per profile within `data/enrollment/<profile_id>/`.
* **Face embeddings:** Extracted securely using `facenet-pytorch` and saved as mathematical 512-dimensional arrays (`face_encoding.npy`). Raw images are NOT saved.
* **Voice embeddings:** Speaker identities extracted using `resemblyzer` and saved as mathematical 256-dimensional arrays (`voice_embedding.npy`). Original enrollment recordings are discarded.
* **Passwords:** Kept as hashed string values in `meta.json`. 

> **Important**: All user authentication metadata (`data/enrollment/`) is excluded via `.gitignore` and `.dockerignore`, so sensitive local biometric embeddings will not be accidentally pushed to GitHub.
