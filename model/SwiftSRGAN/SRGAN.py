import torch
from torch import nn
import numpy as np

from PIL import Image
import queue

from cv2.typing import MatLike
from typing import Tuple, Union

'''
Swift-SRGAN aims to implement a model that is able to achieve the results of normal SRGAN, but much faster
- Done using depthwise separable convolution in the GAN network, achieving a performance model that is comparible to normal SRGAN
  while speeding up computation 74x faster and is also 1/8 of original model size
'''
class SwiftSRGANInference():
    """
    Super-resolves images using Fast-SRGAN in real-time before running YOLO predictions
    Uses threading to avoid webcam lag during inference (might not be needed though)
    """
    def __init__(self, MODEL_PATH: str = 'model/srgan.pt', CONFIG_PATH: Union[str, None] = None):
        # Initialization setup
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # NOTE: updated -- multiprocessing only works on cpu:
        self.device = 'cpu'

        # Initialize the model with weights
        self.model = Generator(in_channels=3, upscale_factor=2) #Change upscale factor based on model used x<scale>
        checkpoint = torch.load(f=MODEL_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device).eval()
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8) # INT8 Quantization
        self.upscaled_queue = queue.Queue(maxsize=1) #Multiprocessing queue

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

class Generator(nn.Module):
    """
    Swift-SRGAN Generator, creating fake data (Super Resolution images) with discriminator feedback
    """

    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_blocks: int = 16, upscale_factor: int = 4):
        super(Generator, self).__init__()
        
        self.initial = ConvBlock(
            in_channels=in_channels, 
            out_channels=num_channels, 
            kernel_size=9, 
            stride=1, 
            padding=4, 
            use_bn=False)
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)] #Unpack list comprehension
        )
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsampler = nn.Sequential(
            *[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)]
        )
        self.final_conv = SeparableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x) + initial
        x = self.upsampler(x)
        return (torch.tanh(self.final_conv(x)) + 1) / 2
    
class Discriminator(nn.Module):
    """
    Defines the class for the Swift-SRGAN discriminator:
    - Distinguishes fake and real data to provide feedback for Generator performance
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: Tuple[int] = (64, 64, 128, 128, 256, 256, 512, 512),
    ) -> None:
        super(Discriminator, self).__init__()

        blocks = []
        for i, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + i % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if i == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

class SeparableConv2d(nn.Module):
    '''
    This class implements a Depthwise Separable Convolution
    - Two steps: Depthwise convolution and point-wise convolution
        - Depthwise conv applied to one single image channel at a time
        - Point-wise conv (1x1) conv operation performed on M image channels
    - Attempts to reduce the number of parameters to change compared to normal conv
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1, bias: bool = True):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias,
            padding=padding
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    '''
    Defines the class for a Swift-SRGAN convolutional layer-block that utilizes depthwise separable convolution for faster inference
    '''
    def __init__(self, in_channels: int, out_channels: int, use_act: bool=True, use_bn: bool=True, discriminator: bool = False, **kwargs):
        super(ConvBlock, self).__init__()

        self.use_act = use_act
        self.cnn = SeparableConv2d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   **kwargs, 
                                   bias=not use_bn)
        self.bn = nn.BatchNorm2d(num_features=out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.use_act:
            x = self.cnn(x)
            x = self.bn(x)
            x = self.act(x)
            return x
        else:
            x = self.cnn(x)
            x = self.bn(x)
            return x
    
class UpsampleBlock(nn.Module):
    '''
    Define the class for an upsampling block, which increases spatial resolution (SuperRes) of images
    '''
    def __init__(self, in_channels: int, scale_factor: int):
        super(UpsampleBlock, self).__init__()
        
        self.conv = SeparableConv2d(in_channels=in_channels, 
                                    out_channels=in_channels * scale_factor**2, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1)
        
        self.ps = nn.PixelShuffle(scale_factor) # (in_channels * 4, H, W) -> (in_channels, H*2, W*2)
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        x = self.ps(x)
        x = self.act(x)
        return x
        
class ResidualBlock(nn.Module):
    '''
    Define the class for a Residual block:
    - Address the vanishing gradient problem, relaying information via residual connnection
    - Enable training of very deep NNs by incorporating skip connections
    '''
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.block1(x)
        out = self.block2(out)
        return out + x