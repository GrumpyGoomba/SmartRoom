
import ast
import cv2
from pathlib import Path
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

app = Flask(__name__)

MAIN_FILE = Path(__file__).resolve().parent / "main.py"


def load_settings():
    tree = ast.parse(MAIN_FILE.read_text())

    bed_region = None
    camera_size = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "BED_REGION"
                ):
                    bed_region = ast.literal_eval(node.value)

        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "size"
                ):
                    try:
                        size = ast.literal_eval(value)
                        if (
                            isinstance(size, (tuple, list))
                            and len(size) == 2
                        ):
                            camera_size = size
                    except (ValueError, TypeError):
                        pass

    if bed_region is None or camera_size is None:
        raise ValueError("Camera settings missing from main.py")

    return bed_region, camera_size


BED_REGION, CAMERA_SIZE = load_settings()

print("Camera resolution:", CAMERA_SIZE)
print("Bed region:", BED_REGION)

picam2 = Picamera2(camera_num=0)
picam2.configure(
    picam2.create_video_configuration(
        main={
            "size": tuple(CAMERA_SIZE),
            "format": "BGR888"
        }
    )
)
picam2.start()


def generate_frames():
    while True:
        frame = picam2.capture_array()

        if frame is None:
            continue

        height, width = frame.shape[:2]
        bx1, by1, bx2, by2 = BED_REGION

        x1 = int(bx1 * width)
        y1 = int(by1 * height)
        x2 = min(width - 1, int(bx2 * width))
        y2 = min(height - 1, int(by2 * height))

        # Purple bed region
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "BED REGION",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        # Encode frame as JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Bed Camera</title>
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 20px;
            }
            img {
                width: 100%;
                max-width: 1280px;
                border: 2px solid #a020f0;
            }
        </style>
    </head>
    <body>
        <h2>Raspberry Pi Bed Camera</h2>
        <p>Live Camera Alignment Preview</p>
        <img src="/video_feed">
        <p>Bed Region: {{ region }}</p>
        <p>Resolution: {{ resolution }}</p>
    </body>
    </html>
    """, region=BED_REGION, resolution=CAMERA_SIZE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        )
    finally:
        picam2.stop()
