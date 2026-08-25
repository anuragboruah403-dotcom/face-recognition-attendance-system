import csv
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_FILE = BASE_DIR / "trained_model.yml"
LABELS_FILE = BASE_DIR / "labels.npy"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"

IMAGE_SIZE = (200, 200)
CAPTURE_COUNT = 50
CONFIDENCE_THRESHOLD = 60


def load_face_cascade():
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(
            f"Face cascade file was not found:\n{cascade_path}\n\n"
            "Reinstall opencv-contrib-python."
        )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            f"Could not load face cascade:\n{cascade_path}"
        )

    print(f"[INFO] Face cascade loaded from: {cascade_path}")
    return cascade


FACE_CASCADE = load_face_cascade()


def get_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Could not access the camera.")
        return None

    return camera


def clean_name(name):
    name = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"\s+", "_", name)


def capture_faces():
    DATASET_DIR.mkdir(exist_ok=True)

    name = input("\nEnter person's name, or type 'done': ").strip()

    if name.lower() == "done":
        return

    name = clean_name(name)

    if not name:
        print("[ERROR] Invalid name.")
        return

    person_dir = DATASET_DIR / name
    person_dir.mkdir(exist_ok=True)

    camera = get_camera()

    if camera is None:
        return

    count = len(list(person_dir.glob("*.jpg")))

    print(f"[INFO] Capturing images for {name}.")
    print("[INFO] Look at the camera.")
    print("[INFO] Press ESC or Q to stop.")

    try:
        while count < CAPTURE_COUNT:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Failed to read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) > 0:
                x, y, w, h = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, IMAGE_SIZE)

                count += 1
                image_path = person_dir / f"{count}.jpg"
                cv2.imwrite(str(image_path), face)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"{name}: {count}/{CAPTURE_COUNT}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.imshow("Face Registration", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Captured {count} images for {name}.")
    train_model()


def train_model():
    if not DATASET_DIR.exists():
        print("[ERROR] Dataset folder does not exist.")
        return

    faces = []
    labels = []
    label_map = {}

    person_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not person_folders:
        print("[ERROR] No registered people found.")
        return

    for label, person_folder in enumerate(person_folders):
        image_count = 0
        label_map[label] = person_folder.name

        for image_path in person_folder.glob("*.jpg"):
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            faces.append(image)
            labels.append(label)
            image_count += 1

        print(
            f"[INFO] {person_folder.name}: "
            f"{image_count} training images"
        )

    if not faces:
        print("[ERROR] No valid training images found.")
        return

    if not hasattr(cv2, "face"):
        print(
            "[ERROR] Install opencv-contrib-python "
            "instead of opencv-python."
        )
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))

    model.save(str(MODEL_FILE))
    np.save(str(LABELS_FILE), label_map)

    print("[INFO] Training completed successfully.")


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    file_exists = ATTENDANCE_FILE.exists()
    rows = []

    if file_exists:
        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:
            rows = list(csv.reader(file))

    for row in rows[1:]:
        if len(row) >= 2 and row[0] == name and row[1] == today:
            return False

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists or not rows:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, today, current_time])

    print(f"[ATTENDANCE] {name} marked present at {current_time}")
    return True


def recognize_faces():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        print("[ERROR] No trained model found.")
        print("[INFO] Register at least one person first.")
        return

    if not hasattr(cv2, "face"):
        print(
            "[ERROR] Install opencv-contrib-python "
            "instead of opencv-python."
        )
        return

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(str(MODEL_FILE))

    label_map = np.load(
        str(LABELS_FILE),
        allow_pickle=True
    ).item()

    camera = get_camera()

    if camera is None:
        return

    print("[INFO] Real-time attendance started.")
    print("[INFO] Press ESC or Q to stop.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("[ERROR] Failed to read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for x, y, w, h in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, IMAGE_SIZE)

                label, confidence = model.predict(face)

                if (
                    confidence <= CONFIDENCE_THRESHOLD
                    and label in label_map
                ):
                    name = str(label_map[label])
                    display_name = name.replace("_", " ").upper()
                    mark_attendance(name)
                    color = (0, 255, 0)
                else:
                    display_name = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{display_name} ({confidence:.1f})",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.imshow("Real-Time Face Attendance", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    while True:
        print("\n===== Face Attendance System =====")
        print("1. Register person and train model")
        print("2. Start real-time attendance")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            capture_faces()
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("[ERROR] Invalid choice.")


if __name__ == "__main__":
    main()
