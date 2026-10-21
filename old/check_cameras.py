import cv2

print("Scanning camera devices...\n")

for index in range(10):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Camera {index}: not available")
        continue

    ok, frame = cap.read()
    print(f"Camera {index}: available")
    print(f"  read_frame_success={ok}")
    if frame is not None:
        print(f"  frame_shape={frame.shape}")

    cap.release()

print("\nDone. Try the camera index that shows a real image instead of OBS.")
