# detection.py
"""
Face detection helper module utilizing YOLOv8 models.
Supports bounding box extraction in xyxy and xywh formats with confidence filtering.
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np

def get_model(model_path="yolov8m_200e.pt"):
    """
    Load the YOLOv8 model from a given file path and transfer to GPU if available.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    print(f"[YOLO] Loaded model '{model_path}' on device '{device}'")
    return model

def detect_faces(model, images, box_format='xyxy', th=0.5):
    """
    Detect faces from a list of images (PIL Images or numpy arrays).
    
    Args:
        model: Loaded YOLO model.
        images: List of images (numpy BGR/RGB or PIL).
        box_format: 'xyxy' [x1, y1, x2, y2] or 'xywh' [x_center, y_center, w, h].
        th: Confidence threshold.
        
    Returns:
        results: Raw YOLO results list.
        all_boxes: Filtered bounding box arrays per image.
        all_scores: Confidence scores per image.
        all_classes: Class IDs per image.
    """
    results = model(images, verbose=False)

    all_boxes = []
    all_scores = []
    all_classes = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            all_boxes.append([])
            all_scores.append([])
            all_classes.append([])
            continue

        if box_format == 'xyxy':
            boxes = result.boxes.xyxy.cpu().numpy()
        elif box_format == 'xywh':
            boxes = result.boxes.xywh.cpu().numpy()
        else:
            raise ValueError(f"Unsupported box_format: {box_format}")

        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

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
