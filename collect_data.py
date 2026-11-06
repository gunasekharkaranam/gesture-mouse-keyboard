"""
collect_data.py
---------------
Record our own gesture dataset with the webcam.

Every saved sample = one frame with a detected hand:
  * a row in data/landmarks.csv : user, label, handedness, 63 keypoint values
  * a 96x96 hand crop in data/images/<label>/   (for the CNN baseline)

How to use:
    python collect_data.py --user guna
    - press 0..6 to choose the gesture (see list on screen)
    - press SPACE to start / stop recording that gesture
    - move your hand around, change distance and angle while recording
    - press q to quit
Aim for about 300-400 samples per gesture per person (~15-20 s each).
Each group member should record with their own --user name, so that we
can test on a person the model has never seen (leave-one-user-out).
"""
import argparse
import csv
import os
import time

import cv2

import config
from hand_tracker import HandTracker, draw_hand, crop_hand


def count_existing(path, user):
    """How many samples per gesture this user has already saved."""
    counts = {g: 0 for g in config.GESTURES}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["user"] == user and row["label"] in counts:
                    counts[row["label"]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="name of the person recording")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--save-images", type=int, default=1, help="1 = also save hand crops")
    ap.add_argument("--image-every", type=int, default=2, help="save a crop every N frames")
    args = ap.parse_args()

    new_file = not os.path.exists(config.LANDMARK_CSV)
    f = open(config.LANDMARK_CSV, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        header = ["user", "label", "handedness"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
        writer.writerow(header)

    counts = count_existing(config.LANDMARK_CSV, args.user)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    tracker = HandTracker()

    current, recording, frame_id = 0, False, 0
    print("Keys: 0-6 choose gesture | SPACE start/stop | q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not found."); break
        frame = cv2.flip(frame, 1)        # mirror view (feels natural, same as the app)
        clean = frame.copy()
        hand = tracker.process(frame)
        frame_id += 1

        if hand is not None:
            draw_hand(frame, hand)
            if recording:
                label = config.GESTURES[current]
                writer.writerow([args.user, label, hand.handedness] +
                                [f"{v:.5f}" for v in hand.landmarks.flatten()])
                counts[label] += 1
                if args.save_images and frame_id % args.image_every == 0:
                    crop = crop_hand(clean, hand)
                    if crop is not None:
                        d = os.path.join(config.IMAGE_DIR, label)
                        os.makedirs(d, exist_ok=True)
                        cv2.imwrite(os.path.join(d, f"{args.user}_{int(time.time()*1000)}.jpg"), crop)

        # ---------------- on-screen help ----------------
        cv2.rectangle(frame, (0, 0), (330, 30 + 24 * config.NUM_CLASSES), (30, 30, 30), -1)
        for i, g in enumerate(config.GESTURES):
            col = (0, 255, 255) if i == current else (220, 220, 220)
            cv2.putText(frame, f"{i}: {g:<13} {counts[g]:>5}", (10, 25 + 24 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        status = "REC" if recording else "paused (SPACE)"
        cv2.putText(frame, f"user={args.user}  {status}", (10, 25 + 24 * config.NUM_CLASSES),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if recording else (200, 200, 200), 2)
        if hand is None:
            cv2.putText(frame, "no hand", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Collect gesture data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            recording = not recording
            f.flush()
        if ord("0") <= key < ord("0") + config.NUM_CLASSES:
            current, recording = key - ord("0"), False

    f.close()
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Saved counts:", counts)


if __name__ == "__main__":
    main()
