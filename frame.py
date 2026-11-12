
import threading
import time


class FrameBuffer:
    def __init__(self, fps=5):
        self.condition = threading.Condition()
        self.enabled = False
        self.viewers = 0
        self.fps = fps
        self.frame = None
        self.version = 0
        self.last_publish = 0.0

    def set_enabled(self, enabled):
        with self.condition:
            self.enabled = bool(enabled)
            self.frame = None
            self.version += 1
            self.last_publish = 0.0
            self.condition.notify_all()

    def set_fps(self, fps):
        with self.condition:
            self.fps = fps
            self.last_publish = 0.0

    def add_viewer(self):
        with self.condition:
            self.viewers += 1
            self.condition.notify_all()

    def remove_viewer(self):
        with self.condition:
            self.viewers = max(0, self.viewers - 1)
            if self.viewers == 0:
                self.frame = None

    def publish(self, frame):
        now = time.monotonic()

        with self.condition:
            if not self.enabled or self.viewers == 0:
                return

            if now - self.last_publish < 1.0 / self.fps:
                return

            self.frame = frame.copy()
            self.version += 1
            self.last_publish = now
            self.condition.notify_all()

    def clear(self):
        with self.condition:
            self.frame = None
            self.version += 1
            self.condition.notify_all()

    def get_next(self, last_version, timeout=2):
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    not self.enabled
                    or self.version != last_version
                ),
                timeout=timeout
            )

            return (
                self.frame,
                self.version,
                self.enabled
            )

    def status(self):
        with self.condition:
            return {
                "enabled": self.enabled,
                "viewers": self.viewers,
                "fps": self.fps
            }
