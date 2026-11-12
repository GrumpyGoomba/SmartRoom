import cv2
import threading
from flask import Flask, Response, render_template_string

app = Flask(__name__)

latest_frame = None
frame_condition = threading.Condition()


def update_preview(frame, bed_region):
    global latest_frame

    # Copy frame so YOLO and preview don't interfere
    preview_frame = frame.copy()

    height, width = preview_frame.shape[:2]
    bx1, by1, bx2, by2 = bed_region

    x1 = int(bx1 * width)
    y1 = int(by1 * height)
    x2 = min(width - 1, int(bx2 * width))
    y2 = min(height - 1, int(by2 * height))

    # Draw bed region
    cv2.rectangle(
        preview_frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        3
    )

    cv2.putText(
        preview_frame,
        "BED REGION",
        (x1 + 10, max(25, y1 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    success, buffer = cv2.imencode(
        ".jpg",
        preview_frame,
        [cv2.IMWRITE_JPEG_QUALITY, 70]
    )

    if not success:
        return

    with frame_condition:
        latest_frame = buffer.tobytes()
        frame_condition.notify_all()


def generate_frames():
    last_frame = None

    while True:
        with frame_condition:
            frame_condition.wait_for(
                lambda: latest_frame is not None
                and latest_frame is not last_frame,
                timeout=5
            )
            frame = latest_frame

        if frame is None or frame is last_frame:
            continue

        last_frame = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartRoom Preview</title>
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
        <h2>SmartRoom Live Preview</h2>
        <p>YOLO bed detection camera</p>
        <img src="/video_feed">
    </body>
    </html>
    """)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def start_preview(port=5000):
    thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=port,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    )
    thread.start()
    print(f"Preview server started on port {port}", flush=True)
