
import math
from ultralytics import YOLO
from request import bedroom_lights_off, bedroom_lights_on

PERSON_CLASS_ID = 0

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class BedDetector:
    def __init__(self):
        print("Loading YOLO pose model...", flush=True)
        self.model = YOLO("yolov8n-pose.pt")
        self.reset()
        print("YOLO model ready.", flush=True)

    def reset(self):
        self.previous_lying_down = None
        self.person_in_bed = None
        self.candidate_bed_state = None
        self.candidate_state_frames = 0
        self.last_bed_event = None

    def process_frame(self, frame, config):
        yolo = config["yolo"]
        bed = config["bed"]

        result = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=yolo["confidence"],
            imgsz=yolo["imgsz"],
            verbose=False
        )[0]

        if result.boxes is None or result.keypoints is None:
            return

        if result.keypoints.conf is None:
            return

        selected = None
        largest_area = 0

        for index, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                selected = index

        if selected is None:
            return

        points = result.keypoints.xy[selected]
        confidence = result.keypoints.conf[selected]

        required = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_HIP,
            RIGHT_HIP
        ]

        if not all(
            float(confidence[i]) >= yolo["keypoint_confidence"]
            for i in required
        ):
            return

        lsx, lsy = points[LEFT_SHOULDER].tolist()
        rsx, rsy = points[RIGHT_SHOULDER].tolist()
        lhx, lhy = points[LEFT_HIP].tolist()
        rhx, rhy = points[RIGHT_HIP].tolist()

        hip_px = (lhx + rhx) / 2
        hip_py = (lhy + rhy) / 2

        hip_x = hip_px / frame.shape[1]
        hip_y = hip_py / frame.shape[0]

        shoulder_x = (lsx + rsx) / 2
        shoulder_y = (lsy + rsy) / 2

        torso_angle = math.degrees(
            math.atan2(
                abs(hip_py - shoulder_y),
                abs(hip_px - shoulder_x)
            )
        )

        lying_down = (
            torso_angle <= bed["lying_angle_max"]
        )

        if lying_down != self.previous_lying_down:
            print(f"lying_down={lying_down}", flush=True)
            self.previous_lying_down = lying_down

        bx1, by1, bx2, by2 = bed["region"]

        current_bed_state = (
            bx1 <= hip_x <= bx2
            and by1 <= hip_y <= by2
        )

        if self.person_in_bed is None:
            self.person_in_bed = current_bed_state
            return

        if current_bed_state == self.person_in_bed:
            self.candidate_bed_state = None
            self.candidate_state_frames = 0
            return

        if self.candidate_bed_state == current_bed_state:
            self.candidate_state_frames += 1
        else:
            self.candidate_bed_state = current_bed_state
            self.candidate_state_frames = 1

        if (
            self.candidate_state_frames
            < bed["transition_frames"]
        ):
            return

        self.last_bed_event = (
            "GETTING_IN_BED"
            if current_bed_state
            else "GETTING_OUT_OF_BED"
        )

        print(
            f"event={self.last_bed_event} "
            f"lying_down={lying_down}",
            flush=True
        )

        if current_bed_state:
            bedroom_lights_off()
        else:
            bedroom_lights_on()

        self.person_in_bed = current_bed_state
        self.candidate_bed_state = None
        self.candidate_state_frames = 0
