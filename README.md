# Real-Time Computer Vision & AR Interaction Suite

## 📖 Overview
[cite_start]This repository contains a collection of Python-based computer vision prototypes developed for the Computer Vision & 3D Coursework (2024/2025)[cite: 1]. The project explores three distinct domains of computer vision:
1.  [cite_start]**Real-Time Object Detection & Augmented Reality (AR)**[cite: 12, 21].
2.  **Biometric Security (Face Mesh Analysis)**.
3.  [cite_start]**HCI & 3D Reconstruction** (Hand Tracking via UDP)[cite: 19].

## 📂 Project Structure

### 1. `objectdetect.py` (Object Detection & AR)
A real-time detection system built with **YOLOv8 Nano**. It goes beyond simple classification by integrating interactive AR elements.
* **Core Model:** YOLOv8n (`ultralytics`).
* **Features:**
    * Detects objects (Person, Laptop, Book, Chair) with high-confidence filtering.
    * **Activity Inference:** Uses bounding box geometry (aspect ratio) to deduce if a person is "Standing," "Sitting," or "Waving."
    * **Interactive AR:** Users can hover the mouse over detected objects to reveal "virtual" data overlays and simulated 3D icons.

### 2. `handtracking3d.py` (3D Interaction)
A remote controller interface designed to bridge physical movement with a 3D engine (Unity).
* **Core Library:** `cvzone.HandTrackingModule` (MediaPipe wrapper).
* **Features:**
    * Tracks 21 distinct hand landmarks ($x, y, z$).
    * [cite_start]**UDP Communication:** Streams coordinate data via socket to `127.0.0.1:5052`, enabling real-time control of Unity 3D assets[cite: 20].

### 3. `facialreco2.py` (Biometric Security)
A facial landmark detection prototype intended for security authentication and liveness detection.
* **Core Library:** `mediapipe.face_mesh`.
* **Features:**
    * Maps 468 3D facial landmarks in real-time.
    * Optimized for performance by managing writable flags on image buffers.

---

## 🛠️ Installation & Dependencies

Ensure you have Python 3.8+ installed. Install the required libraries using pip:

```bash
pip install opencv-python ultralytics mediapipe cvzone numpy
