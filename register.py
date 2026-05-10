import cv2
import os

# Set up
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
name = input("Enter the name of the person: ").strip()
save_path = os.path.join("train", name)

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# 🧠 Get last image number to avoid overwriting
def get_next_image_number(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    numbers = []
    for f in files:
        try:
            num = int(os.path.splitext(f)[0])
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers, default=0) + 1  # start from next number

start_index = get_next_image_number(save_path)

cap = cv2.VideoCapture(0)
print("[INFO] Starting image capture. Press 'q' to quit early.")

img_count = 0
max_images = 30  # Number of face images to capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        file_index = start_index + img_count
        file_path = os.path.join(save_path, f"{file_index}.jpg")
        cv2.imwrite(file_path, face_img)
        img_count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{img_count}/{max_images}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or img_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Successfully saved {img_count} new images to: {save_path}")
