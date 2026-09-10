import math

import cv2
import mediapipe as mp
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
BED_REGION = (0.5, 0.73, 1.0, 1.0)
TRANSITION_FRAMES = 10
LYING_TORSO_ANGLE_MAX_DEGREES = 35

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)


if not cap.isOpened():
    raise RuntimeError("Camera not available at index 1. Check macOS camera permissions or try another index.")

mp_hands = mp.solutions.hands.Hands()
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils
yolo_model = YOLO("yolov8n.pt")
previous_human_present = None
previous_pose_detected = None
previous_lying_down = None
person_in_bed = None
candidate_bed_state = None
candidate_state_frames = 0
last_bed_event = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    yolo_result = yolo_model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)[0]
    human_present = yolo_result.boxes is not None and len(yolo_result.boxes) > 0

    # if human_present != previous_human_present:
    #     print(f"human_present={human_present}", flush=True)
    #     previous_human_present = human_present

    if yolo_result.boxes is not None:
        for box, confidence in zip(yolo_result.boxes.xyxy, yolo_result.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            confidence_text = f"person {float(confidence):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # status_text = "Human present: YES" if human_present else "Human present: NO"
    # status_color = (0, 255, 0) if human_present else (0, 0, 255)
    # cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
    #             1, status_color, 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = mp_hands.process(rgb)
    pose_results = mp_pose.process(rgb)

    pose_detected = pose_results.pose_landmarks is not None
    # if pose_detected != previous_pose_detected:
    #     print(f"pose_detected={pose_detected}", flush=True)
    #     previous_pose_detected = pose_detected

    if pose_detected:
        mp_draw.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
        )

        landmarks = pose_results.pose_landmarks.landmark
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        body_visible = min(
            left_hip.visibility,
            right_hip.visibility,
            left_shoulder.visibility,
            right_shoulder.visibility,
        ) >= 0.5

        if body_visible:
            hip_x = (left_hip.x + right_hip.x) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            bed_x1, bed_y1, bed_x2, bed_y2 = BED_REGION
            person_in_region = (
                bed_x1 <= hip_x <= bed_x2
                and bed_y1 <= hip_y <= bed_y2
            )
            torso_angle = math.degrees(math.atan2(
                abs(hip_y - shoulder_y),
                abs(hip_x - shoulder_x),
            ))
            lying_down = torso_angle <= LYING_TORSO_ANGLE_MAX_DEGREES

            if lying_down != previous_lying_down:
                print(f"lying_down={lying_down}", flush=True)
                previous_lying_down = lying_down

            current_bed_state = person_in_region

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
            cv2.putText(frame, posture_text, (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, posture_color, 2)

    bed_x1, bed_y1, bed_x2, bed_y2 = BED_REGION
    cv2.rectangle(
        frame,
        (int(bed_x1 * frame.shape[1]), int(bed_y1 * frame.shape[0])),
        (int(bed_x2 * frame.shape[1]), int(bed_y2 * frame.shape[0])),
        (255, 0, 255),
        2,
    )
    cv2.putText(frame, "BED REGION", (
        int(bed_x1 * frame.shape[1]) + 10,
        int(bed_y1 * frame.shape[0]) + 30,
    ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if last_bed_event:
        cv2.putText(frame, last_bed_event, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0) if person_in_bed else (255, 0, 255), 2)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()