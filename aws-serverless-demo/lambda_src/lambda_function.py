# We should be using loggers instead of print statements
# https://stackoverflow.com/questions/37703609/using-python-logging-with-aws-lambda

# MobileteNetV2 inference
# https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
# https://pytorch.org/vision/stable/models.html

import torch
import os
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import logging

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)

# Retrieve the logger instance
logger = logging.getLogger()

# Define MobileNetV2 weights
weights = MobileNet_V2_Weights.DEFAULT

# Define MobileNetV2 transforms
preprocess = weights.transforms()

logging.info("Loading PyTorch model")

model = mobilenet_v2(weights=MobileNet_V2_Weights)

logging.info(f"Model loaded: {model}")

# Read imagenet classes
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Load a sample image just to test the model locally
input_image = Image.open("dog.jpg")

input_tensor = preprocess(input_image)

# Create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

# Set model to eval mode
model.eval()


def lambda_handler(event, context):
    print(event)
    print(context.__dict__)

    logging.info("Received event: " + str(event))

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_catid = torch.topk(probabilities, 3)

    return {
        "statusCode": 200,
        "body": "Hello World",
        "top3predictions": {
            "1": {
                "class": categories[top3_catid[0]],
                "probability": top3_prob[0].item(),
            },
            "2": {
                "class": categories[top3_catid[1]],
                "probability": top3_prob[1].item(),
            },
            "3": {
                "class": categories[top3_catid[2]],
                "probability": top3_prob[2].item(),
            },
        },
    }


if __name__ == "__main__":
    print("Hello World")
