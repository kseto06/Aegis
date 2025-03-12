# Imports
import random

import numpy as np
import torch
from typing import Tuple, Type
from sklearn.model_selection import train_test_split

import colorsys

from ultralytics import YOLO

#Webcam:
import cv2
import threading
from LapSRN.LapSRN import SuperResolution

# YOLO Class
class YOLO_Detection():
    def __init__(self, model_path: str='yolo/yolo11n.pt'):
        self.CLASSES: list[str] = ['cyclist', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush'] #Custom cyclist class + default COCO 80 classes
        
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

# Running cyclist inference
class Inference(): 
    # Pass in a yolo class and model path
    def __init__(self, yolo: Type[object], model_path: str, super_res_path):
        self.yolo = yolo
        self.model = YOLO(model_path)
        self.CLASSES = yolo.CLASSES
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SuperRes = SuperResolution(MODEL_PATH=super_res_path)
        self.SuperRes.set_model(scale=2) #Adjust based on LapSRN x<num> scale

    def predict(self, video_src: int = 0, score_threshold: float = 0.6, iou_threshold: float = 0.5, max_boxes: int = 10, zoom: int = 1, use_webcam: bool = False, use_super_res: bool = False):
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

            # Apply super resolution in a separate thread
            if use_super_res:
                upscale_thread = threading.Thread(target=self.SuperRes.upscale_worker, args=(frame,))
                upscale_thread.start()

            # Adjust the zoom
            height, width, _ = frame.shape
            frame = frame[int(height / 2 - height / (2 * zoom)):int(height / 2 + height / (2 * zoom)),
                          int(width / 2 - width / (2 * zoom)):int(width / 2 + width / (2 * zoom))]
            
            if not self.SuperRes.upscaled_queue.empty() and use_super_res:
                upscaled_img = self.SuperRes.upscaled_queue.get()
                frame = cv2.resize(frame, (upscaled_img.shape[1], upscaled_img.shape[0]), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC to maximize quality
            else:
                cv2.resize(frame, (width, height))

            # Run model prediction
            prediction = self.model(frame)

            # Draw the bounding boxes
            # self.plot_bboxes(prediction);
            scores: np.ndarray = prediction[0].boxes.conf.numpy() # probabilities
            classes: np.ndarray = prediction[0].boxes.cls.numpy() # predicted classes
            boxes: np.ndarray = prediction[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
            self.draw_boxes(prediction[0].orig_img, frame, scores, classes, boxes, self.CLASSES, self.generate_colors(self.CLASSES), score_threshold)
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

    def draw_boxes(self, img, frame, scores, classes, boxes, names, colors, score_threshold):
        '''
        This function draws the bounding box with class labels/scores over the frame.
        '''
        thickness = (frame.shape[0] + frame.shape[1]) // 300

        for score, cls, bbox in zip(scores, classes, boxes):
            if score > score_threshold:
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
    inference = Inference(yolo, model_path='yolo/TrainedCTModels/CT_model.onnx', super_res_path='LapSRN/LapSRN_x2.pb')
    inference.predict(video_src=0, score_threshold=0.3, iou_threshold=0.5, max_boxes=10, zoom=1.2, use_webcam=True, use_super_res=True)