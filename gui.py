import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
import os
import shutil
import cv2
import numpy as np

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")

        # Set window size, minimum size, and background color
        self.root.geometry("600x500")
        self.root.minsize(600, 500)
        self.root.configure(bg="#F0F0F0")

        self.title_label = tk.Label(root, text="Face Recognition System", font=("Arial", 22, "bold"), bg="#F0F0F0", fg="#333")
        self.title_label.pack(pady=30)

        button_frame = tk.Frame(root, bg="#F0F0F0")
        button_frame.pack(pady=40)

        self.start_button = tk.Button(button_frame, text="Start Recognition", width=25, height=2, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", relief="flat", command=self.start_recognition)
        self.start_button.grid(row=0, column=0, padx=20, pady=15)

        self.register_button = tk.Button(button_frame, text="Register Face", width=25, height=2, font=("Arial", 14, "bold"), bg="#2196F3", fg="white", relief="flat", command=self.open_register_window)
        self.register_button.grid(row=1, column=0, padx=20, pady=15)

        self.delete_button = tk.Button(button_frame, text="Delete Dataset", width=25, height=2, font=("Arial", 14, "bold"), bg="#FFA500", fg="white", relief="flat", command=self.open_delete_window)
        self.delete_button.grid(row=2, column=0, padx=20, pady=15)

        # Show Registered People Button
        self.show_button = tk.Button(button_frame, text="Show Registered People", width=25, height=2, font=("Arial", 14, "bold"), bg="#9C27B0", fg="white", relief="flat", command=self.show_registered_people)
        self.show_button.grid(row=3, column=0, padx=20, pady=15)

        # Reset All Data Button
        self.reset_button = tk.Button(button_frame, text="Reset All Data", width=25, height=2, font=("Arial", 14, "bold"), bg="#607D8B", fg="white", relief="flat", command=self.reset_all_data)
        self.reset_button.grid(row=4, column=0, padx=20, pady=15)

        self.exit_button = tk.Button(button_frame, text="Exit", width=25, height=2, font=("Arial", 14, "bold"), bg="#f44336", fg="white", relief="flat", command=self.exit_program)
        self.exit_button.grid(row=5, column=0, padx=20, pady=15)

        self.status_label = tk.Label(root, text="Status: Ready", font=("Arial", 16), bg="#F0F0F0", fg="#333")
        self.status_label.pack(pady=20)

        self.footer_label = tk.Label(root, text="Powered by Face Recognition", font=("Arial", 10), bg="#F0F0F0", fg="#888")
        self.footer_label.pack(side="bottom", pady=10)

    def start_recognition(self):
        self.status_label.config(text="Status: Recognizing...")
        threading.Thread(target=self.run_recognition).start()

    def run_recognition(self):
        subprocess.run(['python', 'recognize_face.py'])

    def open_register_window(self):
        self.register_window = tk.Toplevel(self.root)
        self.register_window.title("Register Face")

        tk.Label(self.register_window, text="Enter your name:", font=("Arial", 12)).pack(pady=10)
        self.name_entry = tk.Entry(self.register_window, font=("Arial", 12))
        self.name_entry.pack(pady=10)

        register_button = tk.Button(self.register_window, text="Start Registration", command=self.start_registration, font=("Arial", 12), bg="#4CAF50", fg="white")
        register_button.pack(pady=20)

    def start_registration(self):
        name = self.name_entry.get()
        if name.strip() == "":
            messagebox.showerror("Error", "Name cannot be empty!")
            return

        self.register_window.destroy()
        self.capture_face(name)

    def capture_face(self, name):
        save_path = os.path.join("train", name)
        os.makedirs(save_path, exist_ok=True)

        def get_next_image_number(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            numbers = []
            for f in files:
                try:
                    num = int(os.path.splitext(f)[0])
                    numbers.append(num)
                except ValueError:
                    continue
            return max(numbers, default=0) + 1

        start_index = get_next_image_number(save_path)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)

        img_count = 0
        max_images = 30

        while img_count < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img_resized = cv2.resize(face_img, (200, 200))
                file_index = start_index + img_count
                cv2.imwrite(os.path.join(save_path, f"{file_index}.jpg"), face_img_resized)
                img_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{img_count}/{max_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Registering Face", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        messagebox.showinfo("Success", f"Registration completed for {name}. {img_count} new images saved.")
        self.status_label.config(text="Status: Ready")
        self.rebuild_trainer()

    def open_delete_window(self):
        self.delete_window = tk.Toplevel(self.root)
        self.delete_window.title("Delete Dataset")

        self.refresh_names_list()

        tk.Label(self.delete_window, text="Select a person to delete their dataset:", font=("Arial", 12)).pack(pady=10)
        self.person_var = tk.StringVar(self.delete_window)
        self.person_var.set(self.names_list[0] if self.names_list else "No names available")
        person_dropdown = tk.OptionMenu(self.delete_window, self.person_var, *self.names_list)
        person_dropdown.pack(pady=10)

        delete_button = tk.Button(self.delete_window, text="Delete Dataset", command=self.delete_dataset, font=("Arial", 12), bg="#f44336", fg="white")
        delete_button.pack(pady=20)

    def refresh_names_list(self):
        self.names_list = os.listdir("train")
        self.names_list = [name for name in self.names_list if os.path.isdir(os.path.join("train", name))]

    def delete_dataset(self):
        selected_person = self.person_var.get()

        confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the dataset for {selected_person}?")
        if confirm:
            folder_path = os.path.join("train", selected_person)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

                if os.path.exists("trainer.yml"):
                    os.remove("trainer.yml")
                if os.path.exists("labels.npy"):
                    os.remove("labels.npy")

                messagebox.showinfo("Success", f"Dataset for {selected_person} deleted successfully.")
                self.delete_window.destroy()
                self.rebuild_trainer()
            else:
                messagebox.showerror("Error", "The selected dataset does not exist.")

    def rebuild_trainer(self):
        faces = []
        labels = []
        label_dict = {}
        current_label = 0

        for person_name in os.listdir("train"):
            person_folder = os.path.join("train", person_name)
            if os.path.isdir(person_folder):
                label_dict[current_label] = person_name
                for img_name in os.listdir(person_folder):
                    if img_name.endswith(".jpg"):
                        img_path = os.path.join(person_folder, img_name)
                        img = cv2.imread(img_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces.append(gray)
                        labels.append(current_label)
                current_label += 1

        if faces:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            recognizer.save("trainer.yml")
            np.save("labels.npy", label_dict)
            print("Model rebuilt and saved!")
        else:
            print("No faces to train!")
        

    def exit_program(self):
        self.root.quit()

    def show_registered_people(self):
        self.refresh_names_list()
        if not self.names_list:
            messagebox.showinfo("Registered People", "No registered users found.")
        else:
            names_str = "\n".join(self.names_list)
            messagebox.showinfo("Registered People", f"The following users are registered:\n\n{names_str}")

    def reset_all_data(self):
        confirm = messagebox.askyesno("Confirm Reset", "This will delete all registered data and models. Are you sure?")
        if confirm:
            # Delete train folder
            if os.path.exists("train"):
                shutil.rmtree("train")

            # Delete model files
            for file in ["trainer.yml", "labels.npy"]:
                if os.path.exists(file):
                    os.remove(file)

            # Recreate empty train directory
            os.makedirs("train", exist_ok=True)

            messagebox.showinfo("Reset Done", "All data has been deleted. The system is now reset.")
            self.status_label.config(text="Status: Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
