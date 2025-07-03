**😊 Real-Time Emotion Detection App**

A real-time emotion recognition system built using MediaPipe Face Mesh, FER (Facial Expression Recognition), and Streamlit. This application uses your webcam to detect facial emotions and display the dominant emotion live in the browser.

**📖 Project Description**

This project captures live video from the user's webcam, detects facial landmarks using MediaPipe, and analyzes facial expressions using the FER (Facial Expression Recognition) library. It is deployed via Streamlit, providing a simple and interactive user interface to start and stop the camera and view detected emotions in real time.

**🔍 Key Features**

🎥 Real-time webcam feed in the browser

🧠 Emotion detection using pre-trained FER model

🗺️ Face detection and bounding box with MediaPipe Face Mesh

🧾 Displays dominant emotion (e.g., Happy, Sad, Angry, Normal)

🖱️ Start and Stop camera buttons for full control

💡 Lightweight, fast, and responsive using Streamlit

**🛠️ Technologies Used**

Python

Streamlit (for interactive web UI)

OpenCV (for webcam and image handling)

FER (for emotion detection)

MediaPipe (for facial landmark detection)


▶️ How to Run

Run the Streamlit app:

streamlit run emotion.py

Use the buttons in the interface to start/stop your webcam and see detected emotions.

**✅ Advantages**
No need to write complex code to view facial expressions

Runs entirely in the browser – no local UI needed

Useful for emotion analysis, online classes, mood tracking, and AI-human interaction

Replaces “neutral” emotion label with user-friendly "Normal"

**🧪 Sample Emotions Detected**

😐 Normal (was "neutral")

😀 Happy

😡 Angry

😢 Sad

😮 Surprise

