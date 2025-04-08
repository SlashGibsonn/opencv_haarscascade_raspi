import os
import cv2
import numpy as np
from PIL import Image
import time
from picamera2 import Picamera2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

known_faces = []
known_names = []

# Path ke folder gambar
images_path = "images/"
valid_extensions = ('.jpg', '.jpeg', '.png')

# Cek apakah folder ada
if not os.path.exists(images_path):
    print(f"Error: Direktori '{images_path}' tidak ada.")
else:
    print(f"Direktori '{images_path}' ada.")
    files = os.listdir(images_path)
    if not files:
        print("Error: Direktori kosong")
    else:
        print("Loading gambar untuk face recognition.")
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                img_path = os.path.join(images_path, file_name)
                try:
                    pil_image = Image.open(img_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        face = gray[y:y + h, x:x + w]
                        resized_face = cv2.resize(face, (100, 100))
                        known_faces.append(resized_face)
                        known_names.append(os.path.splitext(file_name)[0])
                        print(f"Wajah ditemukan dan dimuat: {file_name}")
                except Exception as e:
                    print(f"Error: {e}")

# Inisialisasi PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)  # Waktu inisialisasi

print("Siap untuk deteksi wajah dengan PiCamera2")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_face_roi = cv2.resize(face_roi, (100, 100))
        recognize = False

        for idx, known_face in enumerate(known_faces):
            result = cv2.matchTemplate(resized_face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, _) = cv2.minMaxLoc(result)

            if max_val > 0.6:
                recognize = True
                name = known_names[idx]
                similarity = round(max_val * 100, 2)
                cv2.putText(frame, f"{name} ({similarity}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        if not recognize:
            cv2.putText(frame, "Wajah tidak dikenal", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Face Recognition dengan PiCamera2", frame)

    if cv2.waitKey(1) & 0xFF == ord("t"):
        break

cv2.destroyAllWindows()
picam2.stop()