# detection.py

import torch
from ultralytics import YOLO

def get_model(model_path="yolov8l_100e.pt"):
    """Load the YOLOv8 model from a given path and move to appropriate device."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    return model

def detect_faces(model, images, box_format='xywh', th=0.6):
    """
    Detect faces from a list of PIL images using YOLOv8.
    Returns bounding boxes in desired format with scores and classes.
    """
    results = model(images, verbose=False)

    all_boxes = []
    all_scores = []
    all_classes = []

    for result in results:
        if result.boxes is None:
            all_boxes.append([])
            all_scores.append([])
            all_classes.append([])
            continue

        # Choose box format
        if box_format == 'xywh':
            boxes = result.boxes.xywh.cpu().numpy()
        elif box_format == 'xyxy':
            boxes = result.boxes.xyxy.cpu().numpy()
        else:
            raise ValueError(f"Unsupported box_format: {box_format}")

        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # Filter by confidence threshold
        f_boxes = []
        f_scores = []
        f_classes = []

        for box, score, cls in zip(boxes, scores, classes):
            if score >= th:
                f_boxes.append(box)
                f_scores.append(score)
                f_classes.append(cls)

        all_boxes.append(f_boxes)
        all_scores.append(f_scores)
        all_classes.append(f_classes)

    return results, all_boxes, all_scores, all_classes
