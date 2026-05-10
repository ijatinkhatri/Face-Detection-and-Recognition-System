# import cv2
# import os
# import numpy as np

# # Path to training images
# data_path = 'train'
# faces = []
# labels = []
# label_dict = {}  # label:int → name
# current_label = 0

# # Prepare data
# for person_name in os.listdir(data_path):
#     person_path = os.path.join(data_path, person_name)
#     if not os.path.isdir(person_path):
#         continue

#     label_dict[current_label] = person_name

#     for image_name in os.listdir(person_path):
#         image_path = os.path.join(person_path, image_name)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         if image is None:
#             continue

#         faces.append(image)
#         labels.append(current_label)

#     current_label += 1

# # Train recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.train(faces, np.array(labels))

# # Save model and label names
# recognizer.save('trainer.yml')
# np.save('labels.npy', label_dict)

# print("Training complete.")





import cv2
import os
import numpy as np

# Initialize face recognizer and face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []
label_dict = {}
current_label = 0

# Loop over each person in the 'train' folder
for person_name in os.listdir('train'):
    person_path = os.path.join('train', person_name)
    if not os.path.isdir(person_path):
        continue

    # Create a label mapping (name -> label)
    label_dict[current_label] = person_name

    # Loop over all images of the person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        faces.append(image)
        labels.append(current_label)

    current_label += 1

# Train the recognizer on the collected faces and labels
recognizer.train(faces, np.array(labels))

# Save the trained model to 'trainer.yml'
recognizer.save('trainer.yml')

# Save the label dictionary (name -> label mapping) to 'labels.npy'
np.save('labels.npy', label_dict)

print("[INFO] Training complete. Model saved as 'trainer.yml'.")
