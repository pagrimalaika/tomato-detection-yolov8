from flask import Flask, render_template, Response, jsonify, request 
import cv2
import os
import time
import atexit
from ultralytics import YOLO
import threading

app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {model_path}")

model = YOLO(model_path)

cap = None
camera_url = None
fps_start_time = time.time()
fps_frame_count = 0
fps = 0
lock = threading.Lock() 


class_colors = {
    "matang": (0, 0, 255),            # merah
    "setengah matang": (0, 165, 255), # oranye
    "mengkal": (0, 255, 0),           # hijau muda
    "muda": (34, 139, 34),            # hijau tua
    "cacat": (255, 0, 0),             # biru
    "mahkota": (128, 0, 128)          # ungu
}


def set_camera(url):
    """Set kamera baru dari IP DroidCam"""
    global cap, camera_url
    if cap:
        cap.release()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera di {url}")
    camera_url = url
    atexit.register(lambda: cap.release())

def generate_frames():
    global fps_start_time, fps_frame_count, fps, tomato_count_global
    while True:
        if not cap:
            continue
        success, frame = cap.read()
        if not success:
            break
        else:
            # Hitung FPS
            fps_frame_count += 1
            if (time.time() - fps_start_time) >= 1:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Deteksi pakai YOLO
            results = model(frame, stream=True)
            tomato_count = 0
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = result.names[cls].lower()

                    
                    color = class_colors.get(cls_name, (255, 255, 255))

                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')


def index():
    return render_template('index.html')


@app.route('/set_camera', methods=['POST'])
def set_camera_route():
    """API untuk set kamera lewat web"""
    url = request.json.get("url")
    try:
        set_camera(url)
        return jsonify({"status": "ok", "camera_url": url})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify({
        "fps": round(fps, 2),
        "camera_url": camera_url
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
