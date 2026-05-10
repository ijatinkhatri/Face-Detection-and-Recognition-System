# Face Detection and Recognition System

A GUI-based Face Detection and Recognition System built using Python, OpenCV, and Tkinter.

This project allows users to:
- Register new faces
- Train the recognition model
- Recognize faces in real-time using webcam
- Delete datasets
- View registered users
- Reset all saved data

---

## Features

- Real-time face detection using Haar Cascade
- Face recognition using LBPH algorithm
- User-friendly GUI interface
- Dataset management system
- Automatic model retraining
- Webcam integration

---

## Technologies Used

- Python
- OpenCV
- Tkinter
- NumPy

---

## Project Files

| File Name | Description |
|-----------|-------------|
| `gui.py` | Main GUI application |
| `register.py` | Captures face images for registration |
| `train.py` | Trains the face recognition model |
| `recognize_face.py` | Performs real-time face recognition |
| `trainer.yml` | Saved trained model |
| `labels.npy` | Label mapping file |
| `train/` | Dataset folder containing face images |

---

## Installation

### 1. Clone Repository
git clone <your-repository-link>
cd <repository-name>
2. Install Dependencies
pip install opencv-contrib-python numpy
How to Run
Run GUI Application
python gui.py
Working Process
Register Face
Click "Register Face"
Enter name
Webcam captures 30 face images automatically
Train Model

Run:

python train.py
Start Recognition

Run:

python recognize_face.py

or use the GUI button.

Important Notes

Webcam is required
trainer.yml and labels.npy are generated after training
The train folder stores all face datasets
Press Q to exit webcam window

Future Improvements

Attendance system integration
Better GUI design
Deep learning face recognition
Database support
Email notifications

Author

Developed by Jatin Khatri
