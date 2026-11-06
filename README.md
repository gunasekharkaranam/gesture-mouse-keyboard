# Mouse & Keyboard Control using Hand Gesture Recognition (EE 604 course project)

**Report:** [EE604_Gesture_Control_Report.pdf](EE604_Gesture_Control_Report.pdf) (LaTeX source in `report/`)  ·  **Slides:** `presentation/Gesture_Control_EE604_minimal.pptx`  ·  **Results:** `results/`

Team: Guna Sekhar Subbarao Karanam, Vooba Rajkumar, G. Bhuvana Reddy (EE 604 Image Processing, IIT Kanpur, 2026)

Control the computer mouse and type on an "air keyboard" using only a normal webcam and hand gestures.

**Input:** live webcam video (RGB).
**Output:** operating-system mouse and keyboard events: cursor movement, left and right click, double click, scroll, drag, key presses.
**Constraints:** real time on a laptop CPU (no GPU), no gloves or markers, normal room lighting, one hand.

## Pipeline

```
webcam frame -> MediaPipe Hand Landmarker (21 keypoints)        [existing model]
             -> feature extraction (mirror, translate, scale, rotate, angles, distances)   [ours]
             -> Gesture MLP (deep learning classifier, 7 classes)                          [ours]
             -> confidence + temporal voting (stable gesture)                              [ours]
             -> action logic:  cursor (stable anchor + One-Euro filter) / click / scroll /
                drag / mode switch / air keyboard with hover-lock                          [ours]
             -> pyautogui -> operating system
```

## Novelty (what is new compared with the usual "MediaPipe + finger-up rules" projects)

1. **Learned gesture classifier on invariant keypoint features.** The features don't change when the hand moves, gets bigger or smaller, rotates, or switches between left and right. We also use data augmentation and test on users the model has never seen (leave-one-user-out). We compare against the rule-based method, SVM, Random Forest, kNN and an image CNN.
2. **Stable cursor anchor.** The cursor follows the index-finger knuckle (landmark 5), not the fingertip. So the cursor does not jump away from the target at the moment you pinch to click. This is measured as "anchor drift".
3. **Speed-adaptive smoothing (One-Euro filter).** We compare it with no filter, a moving average and a Kalman filter, measuring jitter and lag.
4. **Confidence + temporal voting.** This removes one-frame wrong predictions and so prevents false clicks. It is measured as flicker and false clicks.
5. **Few-shot personalization.** Only the last layer is re-trained, using K samples per gesture from a new user (`personalize.py`).
6. **Air keyboard with hover-lock.** The key typed is the one you pointed at just *before* the pinch started.

## Gestures

| id | Gesture | Mouse mode | Keyboard mode |
|---|---|---|---|
| 0 | open palm | pause (cursor frozen) | – |
| 1 | point (index up) | move cursor | hover a key |
| 2 | pinch thumb+index | left click (twice = double click) | type key |
| 3 | pinch thumb+middle (ring and pinky up) | right click | – |
| 4 | two fingers up (V) | scroll (move hand up/down) | – |
| 5 | fist | drag | – |
| 6 | thumbs up, hold 1 s | switch to keyboard | switch to mouse |

## Setup (Windows)

```bat
cd C:\EE604\Project\code
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Use Python 3.10, 3.11 or 3.12 (MediaPipe has no wheels for newer versions yet). The first run downloads the MediaPipe hand model (about 7.5 MB) into `models/`.


## One command for everything (after data and logs are recorded)

```bat
run_all.bat          (or: python run_all.py)
```
This runs all the training and analysis steps below, in the right order. It then builds:
- `report/` and `report_overleaf.zip`: the CVPR-style write-up with your real numbers, tables, figures and the full code appendix. Upload the zip to Overleaf (New Project → Upload Project) and click Recompile.
- `presentation/Gesture_Control_EE604.pptx`: the slides, with speaker notes.

Before you run it, edit `team.json` (names, roll numbers, emails, GitHub link, hardware).
`VIVA_QA.md` has likely viva questions with answers, plus the demo script.

## Step-by-step (what run_all.py does, if you want to run parts by hand)

| Step | Command | Output |
|---|---|---|
| 1. Try it at once (rule-based, no training yet) | `python app.py --rules --dry-run` | live window |
| 2. Record data (**each member**) | `python collect_data.py --user guna` | `data/landmarks.csv`, `data/images/` |
| 3. Train + ablation + baselines | `python train.py` | `results/ablation.md`, confusion matrix, model |
| 4. Unseen-user test | `python train.py --loso` | `results/loso.md` |
| 5. Image-CNN baseline | `python train_cnn.py` | `results/cnn_baseline.md` |
| 6. Personalization | `python personalize.py --user guna` | `results/personalization_guna.md/.png` |
| 7. Retrain final model on the random split | `python train.py` | `models/gesture_mlp.pt` |
| 8. Use it | `python app.py` | controls your PC |
| 9. Record logs for experiments | `python app.py --log --dry-run` (or press `l`) | `logs/session_*.csv` |
| 10. Filter / voting / anchor analysis | `python experiments/analyze_logs.py logs\session_*.csv` | `results/filter_comparison.md`, `voting_comparison.md`, `anchor_drift.md`, `performance.md` |
| 11. Click test | `python experiments/click_test.py --method mouse --user guna`, then `--method gesture` with `app.py` running | `results/click_test_summary.md` |
| 12. Typing test | `python experiments/typing_test.py --user guna` with `app.py` in keyboard mode | `results/typing_summary.md` |

Note: step 4 (`--loso`) does **not** overwrite the deployed model, but step 6 is best run after step 3. Run step 3 once more at the end so that `models/gesture_mlp.pt` is the random-split model.

### Tips for recording data (step 2)
- About 300–400 samples per gesture per person (15–20 seconds of recording each).
- While recording, move your hand around: near and far, a little tilted, at different places in the frame.
- Record with your normal hand. If possible, also record a few seconds with the other hand.
- Record in two lighting conditions if you can (for example, room light and window light).

### Recording logs for step 10
Run `python app.py --log --dry-run` three times, about 1 minute each:
1. **still**: point and keep your hand as still as you can.
2. **move**: move the cursor around, using big and small movements.
3. **clicks**: point, then pinch. Repeat **exactly 30 times**.

### Keys in the app window
`q`/`Esc` quit · `c` control on/off · `k` switch mode · `f` change filter · `a` change anchor point · `v` voting on/off · `r` MLP ↔ rule-based · `l` start/stop logging

For the demo, press `f`, `a`, `v` and `r` live to show the difference each novelty makes.

## Files

| File | What it does |
|---|---|
| `config.py` | all settings |
| `hand_tracker.py` | MediaPipe wrapper, drawing, hand crop |
| `features.py` | invariant features + augmentation |
| `rules.py` | rule-based baseline classifier |
| `model.py` | Gesture MLP, saving/loading, live predictor |
| `voting.py` | confidence + temporal voting |
| `filters.py` | none / moving average / Kalman / One-Euro |
| `virtual_keyboard.py` | air keyboard with hover-lock |
| `controller.py` | sends mouse/keyboard events (pyautogui) |
| `app.py` | main application + UI + logging |
| `collect_data.py` | dataset recording |
| `train.py` | MLP training, ablation, baselines, leave-one-user-out |
| `train_cnn.py` | image-CNN baseline |
| `personalize.py` | few-shot personalization |
| `experiments/analyze_logs.py` | filter, voting, anchor and FPS analysis |
| `experiments/click_test.py` | Fitts'-law click test |
| `experiments/typing_test.py` | air-keyboard WPM test |

## Existing work used
- MediaPipe Hands / Hand Landmarker (Google): https://github.com/google-ai-edge/mediapipe
- One-Euro filter: Casiez, Roussel, Vogel, "1€ Filter", CHI 2012
- pyautogui: https://github.com/asweigart/pyautogui
- Dataset: our own recordings (3 users). Optional public dataset for extra tests: HaGRID, https://github.com/hukenovs/hagrid
