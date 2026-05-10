# import cv2
# import numpy as np

# # Load recognizer & labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer.yml')
# label_dict = np.load('labels.npy', allow_pickle=True).item()

# # Load face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(0)

# print("Press Q to quit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5)

#     for (x, y, w, h) in faces:
#         face_img = gray[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (200, 200))

#         label, confidence = recognizer.predict(face_img)

#         # Set threshold to 70
#         if confidence < 77:
#             name = label_dict.get(label, "Unknown")
#             color = (0, 255, 0)
#         else:
#             name = "Unknown"
#             color = (0, 0, 255)

#         text = f"{name} ({round(confidence, 2)})"

#         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

#     cv2.imshow("Face Recognition", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np

def main():
    # Load the trained model and label dictionary
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')  # Load trained model

    label_dict = np.load('labels.npy', allow_pickle=True).item()  # Load labels (name -> label mapping)

    # Set up the face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (200, 200))

            # Recognize the face using the trained model
            label, confidence = recognizer.predict(face_img_resized)

            # Get the name from the label
            if confidence < 60:
                name = label_dict.get(label, "Unknown")
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces

            # Draw rectangle around face and display name
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({round(confidence, 2)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Show the frame with the detected face and name
        cv2.imshow("Face Recognition", frame)

        # Exit if 'q' or 'Esc' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Run the recognition if this file is run directly
