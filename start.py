
import threading
import time

from picamera2 import Picamera2

from config import load_config, save_config, validate
from frames import FrameBuffer
from main import BedDetector
from preview import create_app


class SmartRoom:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = load_config()
        self.frames = FrameBuffer(
            fps=self.config["preview"]["fps"]
        )

        self.pending = None
        self.detection_status = "starting"
        self.camera_status = "starting"
        self.error = None
        self.stop_event = threading.Event()

    def current_config(self):
        with self.lock:
            return self.config

    def status(self):
        with self.lock:
            return {
                "detection": self.detection_status,
                "camera": self.camera_status,
                "error": self.error
            }

    def set_status(self, detection=None, camera=None, error=None):
        with self.lock:
            if detection is not None:
                self.detection_status = detection
            if camera is not None:
                self.camera_status = camera
            self.error = error

    def request_update(self, data):
        new_config = validate(data)

        with self.lock:
            if self.pending is not None:
                raise RuntimeError("Update already pending")

            # Save before requesting the change.
            save_config(new_config)
            self.pending = new_config
            self.detection_status = "updating"
            self.error = None

    def configure_camera(self, picam2, camera_config):
        configuration = picam2.create_video_configuration(
            main={
                "size": (
                    camera_config["width"],
                    camera_config["height"]
                ),
                "format": "RGB888"
            },
            controls={
                "FrameRate": camera_config["fps"]
            }
        )

        picam2.configure(configuration)
        picam2.start()

        print(
            "Camera ready: "
            f"{camera_config['width']}x"
            f"{camera_config['height']} "
            f"@ {camera_config['fps']} FPS",
            flush=True
        )

    def apply_pending(self, picam2, detector):
        with self.lock:
            new_config = self.pending
            old_config = self.config

        if new_config is None:
            return

        camera_changed = (
            new_config["camera"] != old_config["camera"]
        )

        if camera_changed:
            self.set_status(
                detection="updating",
                camera="reconfiguring"
            )
            self.frames.clear()

            try:
                picam2.stop()
                self.configure_camera(
                    picam2, new_config["camera"]
                )
            except Exception as error:
                print("Camera update failed:", error, flush=True)

                # Attempt to restore previous settings
                try:
                    picam2.stop()
                    self.configure_camera(
                        picam2, old_config["camera"]
                    )
                    save_config(old_config)

                    with self.lock:
                        self.pending = None

                    self.set_status(
                        detection="running",
                        camera="running",
                        error=f"Update failed: {error}"
                    )
                    return

                except Exception as restore_error:
                    with self.lock:
                        self.pending = None

                    self.set_status(
                        detection="error",
                        camera="error",
                        error=str(restore_error)
                    )
                    raise

        self.frames.set_fps(new_config["preview"]["fps"])

        detector.reset()

        with self.lock:
            self.config = new_config
            self.pending = None

        self.set_status(
            detection="running",
            camera="running"
        )

        print("Configuration applied successfully.", flush=True)

    def detection_worker(self):
        picam2 = None

        try:
            picam2 = Picamera2(camera_num=0)

            self.configure_camera(
                picam2, self.current_config()["camera"]
            )

            self.set_status(
                detection="loading model",
                camera="running"
            )

            detector = BedDetector()

            self.set_status(
                detection="running",
                camera="running"
            )

            print("SmartRoom is running!", flush=True)

            while not self.stop_event.is_set():

                # Apply changes between camera captures
                self.apply_pending(picam2, detector)

                config = self.current_config()

                frame = picam2.capture_array()

                if frame is None:
                    continue

                # Cheap operation when preview disabled
                self.frames.publish(frame)

                # Run existing bed detection
                detector.process_frame(frame, config)

        except Exception as error:
            self.set_status(
                detection="error",
                camera="error",
                error=str(error)
            )
            print("SmartRoom error:", error, flush=True)

        finally:
            self.frames.clear()

            if picam2 is not None:
                picam2.close()

            print("Camera released.", flush=True)

    def start(self):
        self.worker = threading.Thread(
            target=self.detection_worker,
            name="SmartRoomDetection",
            daemon=True
        )
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, "worker"):
            self.worker.join()


if __name__ == "__main__":
    room = SmartRoom()
    room.start()

    app = create_app(room, room.frames)

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        )
    finally:
        room.stop()
