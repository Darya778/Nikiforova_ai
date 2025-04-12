import ultralytics
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2
import time

path = Path(__file__).parent
model_path = path / "best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
# image = cv2.imread("scirock.jpg")
state = "igle" # wait, result
prev_time = 0
curr_time = 0
player1_hand = ""
player2_hand = ""
timer = 5
game_result = ""

while cap.isOpened():
    ret, frame = cap.read()
    cv2.putText(frame, f"{state} - {5 - timer}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # frame = image
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    results = model(frame)

    if not results:
        continue

    result = results[0]
    boxes = result.boxes.xyxy.tolist()
    if not boxes:
        continue

    boxes = boxes[0]
    if not boxes:
        continue

    if len(result.boxes.xyxy) == 2:
        labels = []
        for label, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            x1, y1, x2, y2 = xyxy.numpy().astype("int")
            print(result.boxes.cls)
            print(result.names)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            labels.append(result.names[label.item()].lower())

            cv2.putText(frame, f"{labels[-1]}", (x1 + 15, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        player1_hand, player2_hand = labels
        if player1_hand == "rock" and player2_hand == "rock" and state == "igle":
            state = "wait"
            prev_time = time.time()

    if state == "wait":
        timer = round(time.time() - prev_time, 1)
    if timer >= 5:
        timer = 5
        if state == "wait":
            state = "result"
            if player1_hand == player2_hand:
                game_result = "draw"
            elif player1_hand == "scissors":
                if player2_hand == "rock":
                    game_result = "win player 2"
                else:
                    game_result = "win player 1"
            elif player2_hand == "scissors":
                if player1_hand == "rock":
                    game_result = "win player 1"
                else:
                    game_result = "win player 2"
            elif player1_hand == "rock" and player2_hand == "paper":
                game_result = "win player 2"
            elif player2_hand == "rock" and player1_hand == "paper":
                game_result = "win player 1"
            curr_time = time.time()
        cv2.putText(frame, f"{game_result}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if state == "result" and (time.time() - curr_time) >= 5:
        state = "igle"
        game_result = ""


    cv2.imshow("YOLO", frame)

cap.release()
cv2.destroyAllWindows()
