import torch
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
import logging

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)

# Retrieve the logger instance
logger = logging.getLogger()

logging.info("Downloading PyTorch model")

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)


# Download an example image from the pytorch website
import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

logging.info("Downloaded sample image")
