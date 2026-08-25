# Face Recognition Attendance System

A real-time face recognition-based attendance management system built using Python and OpenCV. The application captures facial images, trains an LBPH face recognition model, recognizes registered individuals through a webcam, and automatically records their attendance in a CSV file.

---

## Features

- Register new individuals using a webcam
- Automatic face detection using Haar Cascade
- Face recognition using LBPH
- Real-time face recognition
- Automatic attendance marking
- Prevents duplicate attendance on the same day
- Detects unknown faces
- Stores attendance records in CSV format
- Displays recognition confidence
- Simple command-line interface

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Face detection and recognition |
| OpenCV Contrib | LBPH face recognizer |
| NumPy | Numerical processing |
| CSV | Attendance storage |
| Haar Cascade | Face detection |

---

## Project Structure

```text
face-recognition-attendance-system/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── dataset/
