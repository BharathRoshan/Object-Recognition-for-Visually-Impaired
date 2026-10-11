"""Real-time YOLOv3 object detection with optional spoken labels.

This is an assistive-technology prototype, not a safety-critical navigation system.
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                        help="Camera index (default: 0), video file, or stream URL.")
    parser.add_argument("--config", default="yolov3-coco/yolov3.cfg")
    parser.add_argument("--weights", default="yolov3-coco/yolov3.weights")
    parser.add_argument("--labels", default="yolov3-coco/coco-labels")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--speak", action="store_true",
                        help="Speak newly detected classes (requires pyttsx3).")
    parser.add_argument("--announce-cooldown", type=float, default=5.0,
                        help="Minimum seconds between announcements for a class.")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable the OpenCV preview window.")
    return parser.parse_args()


def load_labels(path):
    labels_path = Path(path)
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. Run from the repository root or pass --labels."
        )
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


def load_detector(config_path, weights_path):
    config = Path(config_path)
    weights = Path(weights_path)
    if not config.is_file():
        raise FileNotFoundError(f"YOLO config not found: {config}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. Download the official YOLOv3 COCO weights "
            "or pass the correct --weights path."
        )
    net = cv2.dnn.readNetFromDarknet(str(config), str(weights))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers


def detect_objects(frame, net, output_layers, labels, confidence_threshold, nms_threshold):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1 / 255.0, size=(416, 416),
        mean=(0, 0, 0), swapRB=True, crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            if scores.size == 0:
                continue
            class_id = int(scores.argmax())
            confidence = float(scores[class_id])
            if confidence < confidence_threshold or class_id >= len(labels):
                continue
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            box_width = int(detection[2] * width)
            box_height = int(detection[3] * height)
            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)
            boxes.append([x, y, box_width, box_height])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold
    )
    detections = []
    if len(indices):
        for idx in indices.flatten():
            x, y, box_width, box_height = boxes[int(idx)]
            class_id = class_ids[int(idx)]
            score = confidences[int(idx)]
            detections.append({
                "label": labels[class_id],
                "confidence": score,
                "box": (x, y, box_width, box_height),
            })
    return detections


def draw_detections(frame, detections):
    for item in detections:
        x, y, width, height = item["box"]
        label = f'{item["label"]} {item["confidence"]:.0%}'
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)
        text_y = max(y - 8, 20)
        cv2.putText(
            frame, label, (max(x, 0), text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA,
        )
    return frame


def speech_worker(messages):
    """Run speech on a background thread so playback doesn't block frame processing."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
    except Exception as exc:
        print(f"Speech output unavailable: {exc}")
        return

    while True:
        message = messages.get()
        if message is None:
            break
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as exc:
            print(f"Could not speak detection: {exc}")
    try:
        engine.stop()
    except Exception:
        pass


def main():
    args = parse_args()
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be greater than 0 and at most 1.")
    if not 0 <= args.nms_threshold <= 1:
        raise SystemExit("--nms-threshold must be between 0 and 1.")
    if args.announce_cooldown < 0:
        raise SystemExit("--announce-cooldown cannot be negative.")

    labels = load_labels(args.labels)
    net, output_layers = load_detector(args.config, args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    messages = queue.Queue()
    speech_thread = None
    if args.speak:
        speech_thread = threading.Thread(target=speech_worker, args=(messages,), daemon=True)
        speech_thread.start()

    last_spoken = {}
    print("Detection running. Press Q in the preview window to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Video source ended or a frame could not be read.")
                break

            detections = detect_objects(
                frame, net, output_layers, labels,
                args.confidence, args.nms_threshold,
            )
            now = time.monotonic()
            current_labels = sorted({item["label"] for item in detections})
            for label in current_labels:
                if args.speak and now - last_spoken.get(label, -1e9) >= args.announce_cooldown:
                    messages.put(label)
                    last_spoken[label] = now

            if args.no_display:
                if current_labels:
                    print("Detected:", ", ".join(current_labels))
            else:
                preview = draw_detections(frame, detections)
                cv2.imshow("Assistive Vision - YOLOv3", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        print("Stopping detection...")
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if speech_thread is not None:
            messages.put(None)
            speech_thread.join(timeout=2)


if __name__ == "__main__":
    main()
