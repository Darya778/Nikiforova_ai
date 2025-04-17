import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def dist(p1, p2) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def angle(a, b, c) -> int:
    aa = dist(b, c)
    bb = dist(a, c)
    cc = dist(a, b)

    cosB = (aa ** 2 + cc ** 2 - bb ** 2) / (2 * aa * cc)
    cosB = np.clip(cosB, -1.0, 1.0)
    B = np.arccos(cosB)
    return np.rad2deg(B)

def process(image, keypoints):
    left_shoulder = keypoints[5]
    left_elbow = keypoints[7]
    left_wrist = keypoints[9]
    right_shoulder = keypoints[6]
    right_elbow = keypoints[8]
    right_wrist = keypoints[10]

    try:
        if right_elbow and not left_elbow:
            elbow_angle = angle(right_shoulder, right_elbow, right_wrist)
            x, y = int(right_elbow[0]) + 10, int(right_elbow[1]) + 10
        else:
            elbow_angle = angle(left_shoulder, left_elbow, left_wrist)
            x, y = int(left_elbow[0]) + 10, int(left_elbow[1]) + 10
        cv2.putText(image, f"{int(elbow_angle)}", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)
        return elbow_angle
    except ZeroDivisionError:
        return None

path = Path(__file__).parent
model_path = path / "yolo11n-pose.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
writer =  cv2.VideoWriter("out1.mp4", cv2.VideoWriter_fourcc(*"avc1"), 10, (640, 480))

last_time = time.perf_counter()
is_down = False
counter = 0
last_down = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    copy = frame.copy()
    cur_time = time.perf_counter()

    results = model(frame, verbose=False, conf=0.5)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if results:
        result = results[0]
        keypoints = result.keypoints.xy.tolist()
        if not keypoints:
            continue

        keypoints = keypoints[0]
        if not keypoints:
            continue

        annotator = Annotator(copy)
        annotator.kpts(result.keypoints.data[0],
                       result.orig_shape, 5, True)
        annotated = annotator.result()

        cv2.putText(annotated, f"fps: {(1 / (cur_time - last_time)):.1f}",
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 255, 0), 1)
        last_time = cur_time

        angle_ = process(annotated, keypoints)
        if angle_ is not None:
            old_value = is_down
            is_down = angle_ < 90
            if old_value and not is_down:
                counter += 1
                last_down = time.perf_counter()

        if time.perf_counter() - last_down > 50:
            counter = 0

        cv2.putText(frame, f"Push-ups: {counter}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
        writer.write(frame)
        cv2.imshow("Pose", annotated)
    cv2.imshow("YOLO", frame)

writer.release()
cap.release()
cv2.destroyAllWindows()
