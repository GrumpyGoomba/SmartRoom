import math

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5
TRANSITION_FRAMES = 4
VERTICAL_ANGLE_TOLERANCE_DEGREES = 20
HORIZONTAL_ANGLE_TOLERANCE_DEGREES = 25

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def angle_from_horizontal(start_point, end_point):
    delta_x = abs(end_point[0] - start_point[0])
    delta_y = abs(end_point[1] - start_point[1])
    if delta_x == 0 and delta_y == 0:
        return None
    return math.degrees(math.atan2(delta_y, delta_x))


def keypoint_is_visible(keypoint_confidence, index):
    return float(keypoint_confidence[index]) >= KEYPOINT_CONFIDENCE_THRESHOLD


def point_from_keypoints(keypoints, index):
    return tuple(float(value) for value in keypoints[index].tolist())


def detect_bed_posture(keypoints, keypoint_confidence):
    torso_keypoints = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not all(keypoint_is_visible(keypoint_confidence, index) for index in torso_keypoints):
        return None

    shoulder_center = (
        (float(keypoints[LEFT_SHOULDER][0]) + float(keypoints[RIGHT_SHOULDER][0])) / 2,
        (float(keypoints[LEFT_SHOULDER][1]) + float(keypoints[RIGHT_SHOULDER][1])) / 2,
    )
    hip_center = (
        (float(keypoints[LEFT_HIP][0]) + float(keypoints[RIGHT_HIP][0])) / 2,
        (float(keypoints[LEFT_HIP][1]) + float(keypoints[RIGHT_HIP][1])) / 2,
    )
    torso_angle = angle_from_horizontal(shoulder_center, hip_center)
    torso_is_vertical = (
        torso_angle is not None
        and abs(torso_angle - 90) <= VERTICAL_ANGLE_TOLERANCE_DEGREES
    )

    leg_is_in_bed_posture = False
    for hip_index, knee_index, ankle_index in (
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ):
        leg_keypoints = (hip_index, knee_index, ankle_index)
        if not all(keypoint_is_visible(keypoint_confidence, index) for index in leg_keypoints):
            continue

        hip = point_from_keypoints(keypoints, hip_index)
        knee = point_from_keypoints(keypoints, knee_index)
        ankle = point_from_keypoints(keypoints, ankle_index)
        thigh_angle = angle_from_horizontal(hip, knee)
        shin_angle = angle_from_horizontal(knee, ankle)
        thigh_is_horizontal = (
            thigh_angle is not None
            and thigh_angle <= HORIZONTAL_ANGLE_TOLERANCE_DEGREES
        )
        shin_is_vertical = (
            shin_angle is not None
            and abs(shin_angle - 90) <= VERTICAL_ANGLE_TOLERANCE_DEGREES
        )
        leg_is_in_bed_posture |= thigh_is_horizontal and shin_is_vertical

    return torso_is_vertical and leg_is_in_bed_posture


def select_largest_person(pose_result):
    if pose_result.boxes is None or pose_result.keypoints is None:
        return None

    selected_person = None
    largest_person_area = 0
    for person_index, box in enumerate(pose_result.boxes.xyxy):
        x1, y1, x2, y2 = box.tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > largest_person_area:
            largest_person_area = area
            selected_person = person_index
    return selected_person


def main():
    picam2 = Picamera2(camera_num=0)
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (1280, 720), "format": "BGR888"},
            controls={"FrameRate": 1},
        )
    )
    picam2.start()

    yolo_pose_model = YOLO("yolov8n-pose.pt")
    is_in_bed_posture = False
    candidate_bed_posture = None
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
            selected_person = select_largest_person(pose_result)

            if selected_person is not None:
                keypoints = pose_result.keypoints.xy[selected_person]
                keypoint_confidence = pose_result.keypoints.conf[selected_person]
                posture_detected = detect_bed_posture(keypoints, keypoint_confidence)

                if posture_detected is not None:
                    if posture_detected != is_in_bed_posture:
                        if candidate_bed_posture == posture_detected:
                            candidate_state_frames += 1
                        else:
                            candidate_bed_posture = posture_detected
                            candidate_state_frames = 1

                        if candidate_state_frames >= TRANSITION_FRAMES:
                            is_in_bed_posture = posture_detected
                            last_bed_event = (
                                "Got In Bed" if is_in_bed_posture else "Got Out of Bed"
                            )
                            print(last_bed_event, flush=True)
                            candidate_bed_posture = None
                            candidate_state_frames = 0
                    else:
                        candidate_bed_posture = None
                        candidate_state_frames = 0

            posture_text = "Bed posture: YES" if is_in_bed_posture else "Bed posture: NO"
            posture_color = (0, 255, 0) if is_in_bed_posture else (0, 0, 255)
            cv2.putText(
                frame,
                posture_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                posture_color,
                2,
            )

            if last_bed_event:
                cv2.putText(
                    frame,
                    last_bed_event,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0) if is_in_bed_posture else (255, 0, 255),
                    2,
                )

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
