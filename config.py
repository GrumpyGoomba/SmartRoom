
import json
import os
import tempfile
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def validate(data):
    camera = data["camera"]
    yolo = data["yolo"]
    preview = data["preview"]
    bed = data["bed"]

    width = int(camera["width"])
    height = int(camera["height"])
    camera_fps = int(camera["fps"])
    imgsz = int(yolo["imgsz"])
    preview_fps = int(preview["fps"])
    confidence = float(yolo["confidence"])
    kp_confidence = float(yolo["keypoint_confidence"])
    transition = int(bed["transition_frames"])
    angle = float(bed["lying_angle_max"])
    region = [float(x) for x in bed["region"]]

    if not (160 <= width <= 1920 and width % 2 == 0):
        raise ValueError("Camera width must be even, 160-1920")

    if not (120 <= height <= 1080 and height % 2 == 0):
        raise ValueError("Camera height must be even, 120-1080")

    if not (1 <= camera_fps <= 60):
        raise ValueError("Camera FPS must be 1-60")

    if not (160 <= imgsz <= 1280 and imgsz % 32 == 0):
        raise ValueError("YOLO imgsz must be a multiple of 32")

    if not (1 <= preview_fps <= 30):
        raise ValueError("Preview FPS must be 1-30")

    if not (0 <= confidence <= 1 and 0 <= kp_confidence <= 1):
        raise ValueError("Confidence values must be 0-1")

    if len(region) != 4 or not (
        0 <= region[0] < region[2] <= 1
        and 0 <= region[1] < region[3] <= 1
    ):
        raise ValueError("Invalid bed region")

    if not (1 <= transition <= 100):
        raise ValueError("Invalid transition frame count")

    if not (0 <= angle <= 90):
        raise ValueError("Invalid lying angle")

    return {
        "camera": {
            "width": width,
            "height": height,
            "fps": camera_fps
        },
        "yolo": {
            "imgsz": imgsz,
            "confidence": confidence,
            "keypoint_confidence": kp_confidence
        },
        "preview": {"fps": preview_fps},
        "bed": {
            "region": region,
            "transition_frames": transition,
            "lying_angle_max": angle
        }
    }


def load_config():
    with CONFIG_PATH.open() as file:
        return validate(json.load(file))


def save_config(data):
    validated = validate(data)
    filename = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=CONFIG_PATH.parent,
            prefix=".config-",
            suffix=".json",
            delete=False
        ) as file:
            filename = file.name
            json.dump(validated, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(filename, CONFIG_PATH)
    finally:
        if filename and os.path.exists(filename):
            os.unlink(filename)

    return validated
