import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

def main():
    # Dapatkan path absolut ke direktori script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "best.pt"
    model_path = os.path.join(script_dir, model_filename)

    # Cek keberadaan model
    print("Folder kerja saat ini:", script_dir)
    print("Daftar file dalam folder:", os.listdir(script_dir))
    print("Path model:", model_path)

    if not os.path.exists(model_path):
        print(f"Error: File model '{model_filename}' tidak ditemukan!")
        return

    print("Memulai program deteksi objek YOLOv8 dengan webcam...")

    print("Memuat model YOLOv8...")
    model = YOLO(model_path)

    print("Menghubungkan ke webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam")
        return

    window_name = "YOLOv8 Webcam Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Deteksi dimulai. Tekan 'q' untuk keluar.")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari webcam")
            break

        fps_frame_count += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program dihentikan")

if __name__ == "__main__":
    main()