'''
Implements the Fast-SRGAN model for fast upsampling at 30fps.
'''

# Imports
import torch
from torch import nn
from torch.nn.utils import prune
import numpy as np

from omegaconf import OmegaConf
from PIL import Image
import queue

from cv2.typing import MatLike
from typing import Tuple, Union


# INFERENCE:
class SRGANInference():
    """
    Super-resolves images using Fast-SRGAN in real-time before running YOLO predictions
    Uses threading to avoid webcam lag during inference (might not be needed though)
    """
    def __init__(self, MODEL_PATH: str = 'model/srgan.pt', CONFIG_PATH: str = 'configs/config.yaml'):
        # Initialization setup
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Initialize the model with weights
        config = OmegaConf.load(CONFIG_PATH)
        self.model = Generator(config=config.generator)
        self.prune_generator(self.model)
        weights = torch.load(f=MODEL_PATH, map_location='cpu')
        self.model.load_state_dict({layer.replace("_orig_mod.", ""): weight for layer, weight in weights.items()}) # Load correct layer names and weights
        self.model.to(self.device).eval()
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8) # INT8 Quantization
        self.upscaled_queue = queue.Queue(maxsize=1) # Multithreading

    def upscale(self, image):
        '''
        Upscales a single image
        '''
        img = np.array(Image.fromarray(image).convert("RGB"))
        img = (torch.from_numpy(img) / 127.5) / 1.0
        img = img.permute(2, 0, 1).unsqueeze(dim=0).to(device=self.device)

        # Process a super-res image using the model
        img = self.model(img.half()).cpu()
        img = (img + 1.0) / 2.0
        img = img.permute(0, 2, 3, 1).squeeze()
        img = (img * 255).numpy().astype(np.uint8)

    def upscale_worker(self, frame: MatLike):
        '''
        Create a separate multithread to perform super-resolution to try to prevent lag
        '''
        with torch.no_grad():
            # Get the low-res/unprocessed image
            img = np.array(Image.fromarray(frame).convert("RGB"))
            img = (torch.from_numpy(img) / 127.5) / 1.0
            img = img.permute(2, 0, 1).unsqueeze(dim=0).to(device=self.device)

            # Process a super-res image using the model
            img = self.model(img).cpu()
            img = (img + 1.0) / 2.0
            img = img.permute(0, 2, 3, 1).squeeze()
            img = (img * 255).numpy().astype(np.uint8)

            # Queue the image
            if self.upscaled_queue.full():
                self.upscaled_queue.get()
            self.upscaled_queue.put(img)

    def prune_generator(self, generator, pruning_factor=0.2):
        """
        Function to prune all the convolutional layers in the Generator model by a given pruning factor.
        """
        
        # Function to prune convolutional layers
        def prune_conv_layer(conv_layer, pruning_factor=pruning_factor):
            prune.l1_unstructured(conv_layer, name="weight", amount=pruning_factor)
            if conv_layer.bias is not None:
                prune.l1_unstructured(conv_layer, name="bias", amount=pruning_factor)
                prune.remove(conv_layer, 'bias')
            prune.remove(conv_layer, 'weight')
        
        # Pruning over the entire Generator NN:
        prune_conv_layer(generator.neck[0], pruning_factor)

        for block in generator.stem:
            prune_conv_layer(block.conv1, pruning_factor)
            prune_conv_layer(block.conv2, pruning_factor)

        prune_conv_layer(generator.bottleneck[0], pruning_factor)

        prune_conv_layer(generator.head[0], pruning_factor)

        for upsample_block in generator.upsampling:
            prune_conv_layer(upsample_block.conv, pruning_factor)

# GENERATOR:
class Generator(nn.Module):
    '''
    Define the class for the Generator, creating fake data (Super Resolution images) with discriminator feedback
    '''
    def __init__(self, config):
        super().__init__()

        # Define Sequential networks to be the Generator of the GAN
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=config.n_filters, kernel_size=3, padding=1), 
            nn.PReLU()
        )

        self.stem = nn.Sequential(
            *[ResidualBlock(
                in_channels=config.n_filters, 
                out_channels=config.n_filters
            ) for _ in range(config.n_layers)]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(config.n_filters)
        )

        self.upsampling = nn.Sequential(
            UpsampleBlock(config=config),
            UpsampleBlock(config=config)
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.n_filters,
                out_channels=3,
                kernel_size=3,
                padding=1
            ),
            nn.Tanh()
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = self.neck(x)
        x = self.stem(residual)
        x = self.bottleneck(x) + residual
        x = self.upsampling(x)
        x = self.head(x)
        return x

class UpsampleBlock(nn.Module):
    '''
    Define the class for an upsampling block, which increases spatial resolution (SuperRes) of images
    '''
    def __init__(self, config):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=config.n_filters,
            out_channels=config.n_filters * 4,
            kernel_size=3,
            padding=1
        )
        self.phase_shift = nn.PixelShuffle(upscale_factor=2)
        self.relu = nn.PReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.relu(self.phase_shift(self.conv(x)))
    
class ResidualBlock(nn.Module):
    '''
    Define the class for a Residual block:
    - Address the vanishing gradient problem, relaying information via residual connnection
    - Enable training of very deep NNs by incorporating skip connections
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Convolutional
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # Batch normalization
        self.bn1 = nn.InstanceNorm2d(out_channels)

        #PReLU function
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = self.relu1(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

# DISCRMINATOR:
class Discriminator(nn.Module):
    '''
    Defines the class for the GAN discriminator, which distinguishes fake and real data to provide feedback for Generator performance
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.neck = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=config.n_filters,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(
                negative_slope=0.2
            )
        )

        self.stem = nn.Sequential(
            DefaultBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters,
                stride=2,
            ),
            DefaultBlock(
                in_channels=config.n_filters,
                out_channels=config.n_filters * 2,
                stride=1,
            ),
            DefaultBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 2,
                stride=2,
            ),
            DefaultBlock(
                in_channels=config.n_filters * 2,
                out_channels=config.n_filters * 4,
                stride=1,
            ),
            DefaultBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 4,
                stride=2,
            ),
            DefaultBlock(
                in_channels=config.n_filters * 4,
                out_channels=config.n_filters * 8,
                stride=1,
            ),
            DefaultBlock(
                in_channels=config.n_filters * 8,
                out_channels=config.n_filters * 8,
                stride=2,
            ),
            torch.nn.Conv2d(
                in_channels=config.n_filters * 8, out_channels=1, kernel_size=1, padding=0, stride=1
            ),
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.neck(x)
        x = self.stem(x)
        return x

class DefaultBlock(nn.Module):
    '''
    Defines the class for a default GAN block/structure
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: Union[Tuple[int, int], int]):
        super.__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False
        )
        self.bn = nn.InstanceNorm2d(out_channels)
        self.LReLU = nn.LeakyReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.LReLU(self.bn(self.conv(x)))


        
