# Face Detection and Recognition System

A Python-based face detection and recognition system that allows users to register faces, train the model, and recognize faces using a webcam. The project also includes a GUI for easy interaction.

---

## 📌 Project Overview

This project is designed to detect and recognize human faces using Python and OpenCV. Users can register face data, train the recognition model, and recognize faces in real time through the webcam.

---

## 🚀 Features

- Face registration system
- Face detection using webcam
- Face recognition in real time
- Model training using registered face data
- GUI-based interface
- Easy to use and beginner-friendly

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Tkinter
- NumPy
- Haar Cascade Classifier
- Webcam

---

## 📁 Project Files

| File Name | Description |
|----------|-------------|
| `register.py` | Captures and stores face images of a user |
| `train.py` | Trains the face recognition model using registered images |
| `recognize_face.py` | Recognizes faces using the trained model |
| `gui.py` | Provides a graphical user interface for the system |

---

## ⚙️ Working Principle

1. The user registers their face using `register.py`.
2. Captured face images are stored in the dataset folder.
3. `train.py` trains the model using the stored face images.
4. `recognize_face.py` detects and recognizes faces in real time.
5. `gui.py` provides buttons/options to access the system easily.

---

## ▶️ How to Run

### 1. Install Required Libraries
pip install opencv-python opencv-contrib-python numpy
2. Register a Face
python register.py
3. Train the Model
python train.py
4. Recognize Face
python recognize_face.py
5. Run GUI
python gui.py
📂 Suggested Folder Structure
face-detection-recognition-system/
│
├── register.py
├── train.py
├── recognize_face.py
├── gui.py
├── dataset/
├── trainer/
└── README.md
📚 Applications
Attendance system
Security system
Identity verification
College mini project
Smart surveillance system
🧠 Future Improvements
Add attendance marking feature
Add database support
Improve recognition accuracy
Add login system
Add cloud storage support
👨‍💻 Author

Developed by Jatin Khatri.

📄 License

This project is for educational purposes only.