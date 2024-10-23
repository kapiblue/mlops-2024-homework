# We should be using loggers instead of print statements
# https://stackoverflow.com/questions/37703609/using-python-logging-with-aws-lambda

import torch
import os
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
import logging

# Initialize you log configuration using the base class
logging.basicConfig(level=logging.INFO)

# Retrieve the logger instance
logger = logging.getLogger()

logging.info("Loading PyTorch model")

model = mobilenet_v2(weights=MobileNet_V2_Weights)

logging.info(f"Model loaded: {model}")


def lambda_handler(event, context):
    print(event)
    print(context.__dict__)

    logging.info("Received event: " + str(event))

    # prediction = ...

    return {
        "statusCode": 200,
        "body": "Hello World"
        # 'prediction': prediction
    }


if __name__ == "__main__":
    print("Hello World")
