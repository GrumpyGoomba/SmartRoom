import math

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
BED_REGION = (0.5, 0.73, 1.0, 1.0)
TRANSITION_FRAMES = 4
LYING_TORSO_ANGLE_MAX_DEGREES = 35

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

picam2 = Picamera2(camera_num=0)
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "BGR888"}
    )
)
picam2.start()

yolo_pose_model = YOLO("yolov8n-pose.pt")
previous_lying_down = None
person_in_bed = None
candidate_bed_state = None
candidate_state_frames = 0
last_bed_event = None

try:
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue

        pose_result = yolo_pose_model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            verbose=False,
        )[0]
        frame = pose_result.plot()

        selected_person = None
        largest_person_area = 0
        if pose_result.boxes is not None and pose_result.keypoints is not None:
            for person_index, box in enumerate(pose_result.boxes.xyxy):
                x1, y1, x2, y2 = box.tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_person_area:
                    largest_person_area = area
                    selected_person = person_index

        lying_down = None
        if selected_person is not None:
            keypoints = pose_result.keypoints.xy[selected_person]
            keypoint_confidence = pose_result.keypoints.conf[selected_person]
            required_keypoints = [
                LEFT_SHOULDER,
                RIGHT_SHOULDER,
                LEFT_HIP,
                RIGHT_HIP,
            ]
            body_visible = all(
                float(keypoint_confidence[index]) >= KEYPOINT_CONFIDENCE_THRESHOLD
                for index in required_keypoints
            )

            if body_visible:
                left_shoulder_x, left_shoulder_y = keypoints[LEFT_SHOULDER].tolist()
                right_shoulder_x, right_shoulder_y = keypoints[RIGHT_SHOULDER].tolist()
                left_hip_x, left_hip_y = keypoints[LEFT_HIP].tolist()
                right_hip_x, right_hip_y = keypoints[RIGHT_HIP].tolist()

                hip_x = (left_hip_x + right_hip_x) / 2 / frame.shape[1]
                hip_y = (left_hip_y + right_hip_y) / 2 / frame.shape[0]
                shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
                shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                torso_angle = math.degrees(math.atan2(
                    abs((left_hip_y + right_hip_y) / 2 - shoulder_y),
                    abs((left_hip_x + right_hip_x) / 2 - shoulder_x),
                ))
                lying_down = torso_angle <= LYING_TORSO_ANGLE_MAX_DEGREES

                if lying_down != previous_lying_down:
                    print(f"lying_down={lying_down}", flush=True)
                    previous_lying_down = lying_down

                bed_x1, bed_y1, bed_x2, bed_y2 = BED_REGION
                current_bed_state = (
                    bed_x1 <= hip_x <= bed_x2
                    and bed_y1 <= hip_y <= bed_y2
                )

                if person_in_bed is None:
                    person_in_bed = current_bed_state
                elif current_bed_state != person_in_bed:
                    if candidate_bed_state == current_bed_state:
                        candidate_state_frames += 1
                    else:
                        candidate_bed_state = current_bed_state
                        candidate_state_frames = 1

                    if candidate_state_frames >= TRANSITION_FRAMES:
                        last_bed_event = (
                            "GETTING_IN_BED" if current_bed_state
                            else "GETTING_OUT_OF_BED"
                        )
                        print(
                            f"event={last_bed_event} lying_down={lying_down}",
                            flush=True,
                        )
                        person_in_bed = current_bed_state
                        candidate_bed_state = None
                        candidate_state_frames = 0
                else:
                    candidate_bed_state = None
                    candidate_state_frames = 0

                posture_text = "Lying: YES" if lying_down else "Lying: NO"
                posture_color = (0, 255, 0) if lying_down else (0, 0, 255)
                cv2.putText(
                    frame,
                    posture_text,
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    posture_color,
                    2,
                )

        bed_x1, bed_y1, bed_x2, bed_y2 = BED_REGION
        cv2.rectangle(
            frame,
            (int(bed_x1 * frame.shape[1]), int(bed_y1 * frame.shape[0])),
            (int(bed_x2 * frame.shape[1]), int(bed_y2 * frame.shape[0])),
            (255, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "BED REGION",
            (
                int(bed_x1 * frame.shape[1]) + 10,
                int(bed_y1 * frame.shape[0]) + 30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

        if last_bed_event:
            cv2.putText(
                frame,
                last_bed_event,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0) if person_in_bed else (255, 0, 255),
                2,
            )

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
