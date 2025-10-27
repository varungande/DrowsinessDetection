# Drowsiness Detection System

A real-time eye-blink and drowsiness detection program built using Python, OpenCV, and dlib.  
This project uses facial landmarks to monitor eye activity through your webcam and alerts when signs of fatigue or drowsiness are detected.

## Features
- Real-time face and eye tracking using your webcam  
- Detects blinking frequency and prolonged eye closure  
- Displays "Blinking" and "Drowsiness Detected" overlays on the video feed  
- Lightweight implementation using classical computer vision (no deep learning model required)  

---

## How It Works
1. The program uses dlib’s pre-trained 68-point facial landmark detector to locate key points on the face (eyes, nose, mouth, etc.).
2. It measures the eye aspect ratio (EAR) — the ratio between the width and height of the eye.
3. When your eyes are open, EAR stays low; when you blink or close your eyes, EAR increases sharply.
4. If the eyes remain closed for several consecutive frames, the system concludes the user might be drowsy and displays a red warning.

---

## Technologies Used
- Python 3  
- OpenCV – video capture and image processing  
- dlib – facial landmark detection  
- NumPy – numerical operations  
- math (hypot) – distance calculation

