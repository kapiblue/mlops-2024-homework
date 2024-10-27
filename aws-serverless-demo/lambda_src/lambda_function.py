import torch
import boto3
import os
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import logging
import json
from io import BytesIO

# Initialize logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Define the path to the weights
model_path = "model/mobilenet_v2.pth"

# Load the model
logging.info("Loading MobileNetV2 model.")
model = mobilenet_v2()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
logging.info("Model loaded successfully.")

# Define MobileNetV2 weights and transformations
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()

# Read imagenet classes
with open("model/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Initialize S3 client
s3_client = boto3.client("s3")

def lambda_handler(event, context):
    # Log the received event
    # logger.info("Received event: %s", json.dumps(event))
    
    # Get bucket name and object key from the event
    bucket_name = event["Records"][0]["s3"]["bucket"]["name"]
    object_key = event["Records"][0]["s3"]["object"]["key"]
    
    try:
        # Download image from S3
        logger.info(f"Downloading image {object_key} from bucket {bucket_name}")
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response["Body"].read()
        
        # Load image
        input_image = Image.open(BytesIO(image_data))
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create mini-batch
        
        # Move input and model to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            model.to("cuda")
        
        # Perform inference
        with torch.no_grad():
            output = model(input_batch)
        
        # Compute probabilities and get top 3 predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        
        predictions = [
            {"class": categories[top3_catid[i]], "probability": top3_prob[i].item()}
            for i in range(3)
        ]
        
        # WARN: Do not save the results back to the S3 to avoid infinite loop lambda calling
        # Save results back to S3 as a JSON file
        result_key = f"{os.path.splitext(object_key)[0]}_inference.json"
        result_data = json.dumps({"predictions": predictions})
        s3_client.put_object(
            Bucket=bucket_name,
            Key=result_key,
            Body=result_data,
            ContentType="application/json"
        )
        
        logger.info(f"Inference results saved to {result_key}")
        
        return {
            "statusCode": 200,
            "body": json.dumps(predictions)
        }
    
    except Exception as e:
        logger.error("Error processing the image: %s", e)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
