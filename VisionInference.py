# Imports
import random
import subprocess

import numpy as np
import torch
from typing import List, Tuple, Type, Union
from sklearn.model_selection import train_test_split

import colorsys

from ultralytics import YOLO
import cv2
from cv2.typing import MatLike

import multiprocessing
from model.LapSRN.LapSRN import LapSRNInference
from model.FastSRGAN.SRGAN import FastSRGANInference
from model.SwiftSRGAN.SRGAN import SwiftSRGANInference

from pyfirmata import Arduino, util
import threading
import time

from lights.youtube_light import launch_video

# YOLO Class
class YOLO_Detection():
    def __init__(self, model_path: str='model/yolo/yolo11n.pt'):
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
    def __init__(self, yolo: Type[object], model_path: str = 'model/yolo/TrainedCTCIMATModels/CTCIMAT.onnx', super_res_model_path: str = None, super_res_config_path: str = None, ARDUINO_PORT: str = '/dev/cu.usbmodem21401'):
        self.yolo = yolo
        self.model = YOLO(model_path)
        self.CLASSES = yolo.CLASSES
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optional Super Resolution models
        if 'LapSRN' in super_res_model_path:
            self.SuperRes = LapSRNInference()

        elif 'FastSRGAN' in super_res_model_path:
            self.SuperRes = FastSRGANInference(MODEL_PATH=super_res_model_path, CONFIG_PATH=super_res_config_path)
                
        elif 'SwiftSRGAN' in super_res_model_path:
            self.SuperRes = SwiftSRGANInference(MODEL_PATH=super_res_model_path, CONFIG_PATH=None)

        # Compile the C (Sound) file to play the action
        subprocess.Popen(["gcc", "./arduino/Sound.c", "-o", "./arduino/Sound"])

        try:
            self.board = Arduino(ARDUINO_PORT)
            self.led_pin = 13
            self.last_detection_time = 0
        except Exception as e:
            print("Arduino initialization failed: ", e)
            self.board = None

        # Boolean to ensure only youtube_light opens once
        self.youtube_opened: bool = False
        
    def predict(self, video_src: int = 0, score_threshold: float = 0.2, iou_threshold: float = 0.5, max_boxes: int = 10, zoom: int = 1, resolution: Union[Tuple[int, int], None] = None, use_webcam: bool = False, 
                use_super_res: bool = False, super_res_model: str = None,
                use_sound: bool = True, use_arduino: bool = False, use_youtube_light: bool = False):
        '''
        This function runs live inference on a connected camera (default: webcam) with optional Super Resolution
        '''
        if use_webcam:
            if resolution is None:
                capture = cv2.VideoCapture(f'http://192.168.205.149:8080/video') #IP when connected to hotspot data
            else:
                capture = cv2.VideoCapture(f'http://192.168.205.149:8080/video', cv2.CAP_DSHOW)
        else:
            if resolution is None:
                capture = cv2.VideoCapture(video_src)
            else:
                capture = cv2.VideoCapture(video_src, cv2.CAP_DSHOW)

        if not capture.isOpened():
            if use_webcam:
                print("Local hosted server could not be opened, reverting to computer webcam")
            capture = cv2.VideoCapture(0)

        if resolution is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, max(resolution))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, min(resolution))
        
        if use_super_res:
            multiprocessing.set_start_method("fork")

        queue = multiprocessing.Queue()  # Create a queue to store the upscaled frame
        sr_process = None
        upscaled_img = None

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Adjust the zoom
            height, width, _ = frame.shape
            frame = frame[int(height / 2 - height / (2 * zoom)):int(height / 2 + height / (2 * zoom)),
                          int(width / 2 - width / (2 * zoom)):int(width / 2 + width / (2 * zoom))]
            
            if use_super_res and super_res_model == 'LapSRN':
                # Apply super resolution in a separate multithread
                if sr_process is None or not sr_process.is_alive():
                    sr_process = multiprocessing.Process(target=self.super_res_worker, args=(frame, queue))
                    sr_process.start()

                # Check if the super resolution process has finished
                if not queue.empty():
                    upscaled_img = queue.get()

                if upscaled_img is not None:
                    # Resize with super-resolution applied frame and Inter-Cubic interpolation
                    frame = cv2.resize(frame, (upscaled_img.shape[1], upscaled_img.shape[0]), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC to maximize quality
                else:
                    frame = cv2.resize(frame, (width, height))

            elif use_super_res and (super_res_model == 'FastSRGAN' or super_res_model == 'SwiftSRGAN'):
                # Apply super resolution in a separate multithread
                if sr_process is None or not sr_process.is_alive():
                    sr_process = multiprocessing.Process(target=self.super_res_worker, args=(frame, queue))
                    sr_process.start()

                # Check if the super resolution process has finished
                if not queue.empty():
                    upscaled_img = queue.get()

                # Resize with super-resolution applied frame
                if upscaled_img is not None:
                    frame = cv2.resize(upscaled_img, (width, height))
                else:
                    frame = cv2.resize(frame, (width, height))

            else:
                cv2.resize(frame, (width, height))

            # Run model prediction
            prediction = self.model(frame)

            # Draw the bounding boxes
            # self.plot_bboxes(prediction);
            scores: np.ndarray = prediction[0].boxes.conf.numpy() # probabilities
            classes: np.ndarray = prediction[0].boxes.cls.numpy() # predicted classes
            boxes: np.ndarray = prediction[0].boxes.xyxy.numpy() # bboxes
            # self.draw_boxes(prediction[0].orig_img, frame, scores, classes, boxes, self.CLASSES, self.generate_colors(self.CLASSES), score_threshold)
            self.draw_boxes(prediction[0].orig_img, boxes, scores, classes, score_threshold)

            # Alert system on prediction
            if len(prediction[0].boxes) > 0:
                self.action(use_sound, use_arduino, use_youtube_light)

            cv2.imshow("Cyclist Detection", frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
        capture.release()
        cv2.destroyAllWindows()

    def action(self, use_sound: bool = True, use_arduino: bool = False, use_youtube_light = False):
        '''
        Using subprocess to open and make a response action
        '''
        # Sound: 
        if use_sound:
            subprocess.Popen(["./arduino/Sound"])

        # YouTube light:
        if use_youtube_light and self.youtube_opened is not True:
            try:
                threading.Thread(target=launch_video, daemon=True).start()
                self.youtube_opened = True
            except Exception as e:
                print(f"Error launching video: {e}")

        # Arduino light:
        def _action_thread():
            self.last_detection_time = time.time()
            self.board.digital[self.led_pin].write(1) # Turning on the LED before the loop

            while time.time() - self.last_detection_time < 10: # Keep the light on for 10s on detection
                time.sleep(1)
            
            self.board.digital[self.led_pin].write(0) # Turning off the LED after the loop

        if self.board is None or not use_arduino:
            return 
        else:
            threading.Thread(target=_action_thread, daemon=True).start()

    def super_res_worker(self, frame: MatLike, queue: multiprocessing.Queue):
        upscaled_img = self.SuperRes.upscale_worker(frame)
        return upscaled_img
    
    def video_predict(self, video_path: str = 'tests/cyclist_vid1.mp4', output_dir: str = 'tests'):
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise Exception("Prediction for video did not init successfully")
            exit()

        # Get video properties
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video writer
        output_path = f"{output_dir}/output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = capture.read()
            if not ret:
                break 

            # Run YOLO inference
            predictions = self.model(frame)
            predictions = [predictions] if not isinstance(predictions, list) else predictions

            # Process detection results
            for prediction in predictions:
                boxes = prediction.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                scores = prediction.boxes.conf.cpu().numpy()  # Get confidence scores
                classes = prediction.boxes.cls.cpu().numpy() # bboxes

                # Draw bounding boxes
                self.draw_boxes(frame, boxes, scores, classes)

            # Show output frame
            cv2.imshow("YOLO Video Inference", frame)

            # Save output frame
            out.write(frame)

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

        capture.release()
        out.release()
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
    
    def draw_boxes(self, frame: MatLike, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, score_threshold: float = 0.3):
        for box, score, cls in zip(boxes, scores, classes):
            if score >= score_threshold:
                x1, y1, x2, y2 = map(int, box)
                label = f"Cyclist: {score:.2f}"
                colors = self.generate_colors(self.CLASSES)
                colors = list(map(lambda x: (int(x[2] * 255), int(x[1] * 255), int(x[0] * 255)), colors))  # Convert to BGR for OpenCV
                
                # Draw bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(cls)], thickness=2*2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5*1.75, colors[int(cls)], thickness=3)

        return frame

if __name__ == '__main__':
    yolo = YOLO_Detection()
    inference = Inference(yolo=yolo, model_path='model/yolo/TrainedCTCIMATModels/CTCIMAT.onnx', super_res_model_path='model/SwiftSRGAN/model/swift_srgan_2x.pth.tar', super_res_config_path=None)
    inference.predict(video_src=0, score_threshold=0.40, iou_threshold=0.5, max_boxes=10, zoom=1, resolution=(1080, 720), use_webcam=True, 
                      use_super_res=False, super_res_model='SwiftSRGAN', 
                      use_sound=False, use_arduino=False, use_youtube_light=True)
    # inference.video_predict()