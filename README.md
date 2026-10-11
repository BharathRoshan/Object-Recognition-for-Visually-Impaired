# Object Recognition for the Visually Impaired

A computer-vision prototype that detects common objects with OpenCV's DNN interface to YOLOv3 and can announce detected object classes using text-to-speech. The aim is to explore a simple visual-assistance workflow; this is **not** a validated navigation aid.

## Features

- Webcam, video-file, or stream input.
- YOLOv3 detection with confidence filtering and non-maximum suppression.
- Bounding boxes and confidence labels in a live preview.
- Optional spoken class names, handled on a background thread.
- Configurable model paths and thresholds.

## Setup

Use Python 3.10 or 3.11 and run commands from the repository root.

```bash
git clone https://github.com/BharathRoshan/Object-Recognition-for-Visually-Impaired.git
cd Object-Recognition-for-Visually-Impaired
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The detector expects the YOLOv3 configuration, COCO weights, and labels at the paths below. The labels file is in `yolov3-coco/coco-labels`; make sure `yolov3.cfg` and `yolov3.weights` are also present. If the weights are not present, obtain the official YOLOv3 COCO weights from their original source and place them in the expected directory. Do not download model files from untrusted mirrors.

## Run

Webcam preview:

```bash
python "object detection.py" --source 0
```

Webcam with spoken detections:

```bash
python "object detection.py" --source 0 --speak
```

Video file:

```bash
python "object detection.py" --source path/to/video.mp4 --speak
```

Adjust the confidence threshold:

```bash
python "object detection.py" --source 0 --confidence 0.6 --nms-threshold 0.4
```

For headless use, `--no-display` prints detected classes to the terminal. Speech requires a working local text-to-speech backend; `pyttsx3` support differs across operating systems.

## How it works

1. Read a frame from the selected video source.
2. Convert it to a 416×416 YOLO input blob.
3. Run inference and filter low-confidence detections.
4. Apply non-maximum suppression to reduce overlapping boxes.
5. Draw detections and, if enabled, announce object classes with a cooldown to limit repeated speech.

## Limitations and responsible use

- Object detection can miss objects, mislabel them, and perform poorly in low light, clutter, or unfamiliar environments.
- The COCO model does not provide reliable distance, collision risk, or safe crossing decisions.
- Do not rely on this prototype to navigate roads, stairs, traffic, or other hazardous settings. Keep a cane, guide dog, human assistance, or other established mobility aid as appropriate.
- Speech output may be delayed and should not be treated as a real-time safety signal.
- Test with consent, avoid recording bystanders, and consider privacy before using a camera in public.

## Troubleshooting

- **Weights/config not found:** check the `--weights` and `--config` paths and run from the repository root.
- **Camera won't open:** close other camera apps or select another camera index with `--source 1`.
- **No speech:** install `pyttsx3` (included in requirements) and verify that your operating system has a working speech engine.
- **Slow inference:** CPU inference may be slow; reduce the input resolution only if you also update the model input handling and validate detection quality.
