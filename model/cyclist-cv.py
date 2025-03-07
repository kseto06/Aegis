# Imports
import os
import random

import numpy as np
import torch
from typing import List, Tuple, Type
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

import colorsys
from PIL import Image, ImageFont, ImageDraw
import imghdr

from ultralytics import YOLO

#Webcam:
import cv2

class YOLO_Detection():
    def __init__(self, model_path: str='yolo/yolo11n.pt'):
        self.CLASSES: list[str] = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']
        
        self.model = YOLO(model_path)

    def filter_boxes(self, box_confidence: torch.Tensor, boxes: torch.Tensor, box_class_probs: torch.Tensor, threshold: float = .6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        This function filters boxes using confidence and class probabilities and seeing if they lie above the certain threshold.
        '''

        # Compute the score of a box as the confidence that there's some object * the probability of it being in a certain class
        box_scores = box_confidence * box_class_probs

        box_classes = torch.argmax(box_scores, dim=-1)
        box_class_scores, _ = torch.max(box_scores, dim=-1, keepdim=False)
        filtering_mask = (box_class_scores >= threshold) # Only filter & keep boxes above the threshold

        # Convert scores to boolean values using the filtering mask
        scores = torch.masked_select(box_class_scores[filtering_mask])
        boxes = torch.masked_select(boxes[filtering_mask])
        classes = torch.masked_select(box_classes[filtering_mask])

        return scores, boxes, classes

    def iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        '''
        Design IOU for non-max suppression (NMS) -- we want to use NMS to only select the most accurate (highest probability of the 3 boxes)
        '''
        (box1_x1, box1_y1, box1_x2, box1_y2) = box1
        (box2_x1, box2_y1, box2_x2, box2_y2) = box2

        # Compute intersections
        xi1 = np.maximum(box1[0], box2[0])
        yi1 = np.maximum(box1[1], box2[1])
        xi2 = np.minimum(box1[2], box2[2])
        yi2 = np.minimum(box1[3], box2[3])
        intersection_width = xi2 - xi1
        intersection_height = yi2 - yi1
        intersection_area = max(intersection_width, 0) * max(intersection_height, 0) #Case where areas do not intersect

        # Compute Union Area and return the iou
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        return float(intersection_area) / float(union_area)

    def non_max_suppression(self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor, max_boxes: int = 10, iou_threshold: float = 0.5) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        '''
        Non-max suppression: Select the highest-score box, overlap the box and remove boxes that overlap significantly
        '''
        nms_detections: list = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        nms_detections = nms_detections[:max_boxes]

        return scores[nms_detections], boxes[nms_detections], classes[nms_detections]

    def train(self):
        '''
        Finetune the pre-trained model on the cyclist dataset using Ultralytics YOLO
        '''
        # Load the dataset
        dataset = DataProcessor()
        dataset.split_data()
        train_images, train_labels = dataset.get_train_data()
        val_images, val_labels = dataset.get_val_data()

        # Load the model
        train_data = TensorDataset(torch.stack(train_images), torch.stack(train_labels))
        val_data = TensorDataset(torch.stack(val_images), torch.stack(val_labels))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

        # Finetune pre-trained model
        self.model.train(data=train_loader, val_data=val_loader, epochs=50, imgsz=640, batch=16)
        self.model.export(format='onnx')

class Inference(): 
    # Pass in a yolo class and model path
    def __init__(self, yolo: Type[object], model_path: str = 'yolo/yolo11s.onnx'):
        self.yolo = yolo
        self.model = YOLO(model_path)
        self.CLASSES = yolo.CLASSES
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, video_src=0, score_threshold=0.6, iou_threshold=0.5, max_boxes=10, use_webcam=False):
        if use_webcam:
            capture = cv2.VideoCapture(f'http://192.168.205.149:8080/video') #IP when connected to hotspot data
        else:
            capture = cv2.VideoCapture(video_src)

        if not capture.isOpened():
            if use_webcam:
                print("Local hosted server could not be opened, reverting to computer webcam")
            capture = cv2.VideoCapture(0)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Run model prediction
            prediction = self.model(frame)

            # Draw the bounding boxes
            # self.plot_bboxes(prediction);
            scores: np.ndarray = prediction[0].boxes.conf.numpy() # probabilities
            classes: np.ndarray = prediction[0].boxes.cls.numpy() # predicted classes
            boxes: np.ndarray = prediction[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
            self.draw_boxes(prediction[0].orig_img, frame, scores, classes, boxes, self.CLASSES, self.generate_colors(self.CLASSES))
            cv2.imshow("Cyclist Detection", frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
        capture.release()
        cv2.destroyAllWindows()

    '''
    Helper functions for YOLO inference, drawing on webcam:
    '''
    def generate_colors(self, class_names):
        '''
        Generates random HSV --> RGB colors for each class
        '''
        hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101) # Fixed seed for consistent colors across runs
        random.shuffle(colors)  # Shuffle colors
        random.seed(None)
        return colors

    def draw_boxes(self, img, frame, scores, classes, boxes, names, colors):
        '''
        This function draws the bounding box with class labels/scores over the frame.
        '''
        thickness = (frame.shape[0] + frame.shape[1]) // 300

        for score, cls, bbox in zip(scores, classes, boxes):
            class_label = names[int(cls)] # class name
            label = f"{class_label} : {score:0.2f}" # bbox label
            lbl_margin = 3 #label margin
            img = cv2.rectangle(img, (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                color=colors[int(score.item())],
                                thickness=thickness)
            label_size = cv2.getTextSize(label,
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                        fontScale=1, thickness=thickness)
            lbl_w, lbl_h = label_size[0]
            lbl_w += 2 * lbl_margin 
            lbl_h += 2 * lbl_margin
            img = cv2.rectangle(img, (bbox[0], bbox[1]),
                                (bbox[0]+lbl_w, bbox[1]-lbl_h),
                                color=colors[int(score.item())], 
                                thickness=-thickness)
            cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(255, 255, 255),
                        thickness=3)
        return img

if __name__ == '__main__':
    yolo = YOLO_Detection()
    train = False

    if train:
        yolo.train()

    inference = Inference(yolo, model_path='yolo/yolo11n.pt')
    inference.predict(video_src=0, score_threshold=0.6, iou_threshold=0.5, max_boxes=10, use_webcam=True)