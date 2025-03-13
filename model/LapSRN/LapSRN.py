import cv2
from cv2 import dnn_superres
import numpy as np
import torch
import queue
import os
from warnings import warn

# Super Resolution object for better camera res to detect
@DeprecationWarning
class LapSRNInference():
    '''
    This class implements LapSRN -- Deep Laplacian Pyramid Networks for efficient and high-quality image super-resolution 
    LapSRN uses CNN-based Super Resolution algorithms where there are:
    - Feature-embedding sub-networks for extracting non-linear features
    - Transposed convolutional layers for upsampling feature maps and images
    - A convolutional layer for predicting the sub-band residuals
    - Weights of each components are shared across pyramid levels to reduce NN parameters
    
    In the context of this project, the super resolution will make image processing clearer and therefore make it easier to 
    detect cyclists with higher-quality webcam images.
    '''

    def __init__(self, MODEL_PATH: str = 'LapSRN_x2.pb', CONFIG_PATH: str = None):
        self.SR = dnn_superres.DnnSuperResImpl_create()
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        self.SR.readModel(MODEL_PATH)

        # Get the scale based on LapSRN x<num> scale
        for char in MODEL_PATH:
            if char.isdigit():
                self.SR.set_model(scale=int(char))
                break

        self.upscaled_queue = queue.Queue(maxsize=1)
        warn("LapSRN is DEPRECATED -- Super Resolution using LapSRN is too slow for real-time inference")

    def set_model(self, scale: int = 2):
        self.SR.setModel('lapsrn', scale)

        if torch.cuda.is_available():
            self.SR.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.SR.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        return self.SR.upsample(image)
    
    # Multithreading to prevent lag
    def upscale_worker(self, frame: np.ndarray) -> np.ndarray:
        upscaled_frame = self.SR.upsample(frame)
        if self.upscaled_queue.full():
            return self.upscaled_queue.get()
        self.upscaled_queue.put(upscaled_frame)