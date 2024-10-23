import torch
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
import logging

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)

# Retrieve the logger instance
logger = logging.getLogger()

logging.info("Downloading PyTorch model")

model = mobilenet_v2(weights=MobileNet_V2_Weights)

logging.info(f"Model loaded: {model}")
