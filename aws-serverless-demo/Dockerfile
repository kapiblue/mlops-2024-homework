# Start from the official AWS Lambda Python 3.8 image
FROM public.ecr.aws/lambda/python:3.8

# Upgrade pip
RUN python3.8 -m pip install --upgrade pip
# Install boto3 for interacting with AWS services
RUN python3.8 -m pip install boto3

# Install model dependencies
RUN python3.8 -m pip install torch==2.4.1 torchvision==0.19.1

# Copy the entire directory (where lambda_function.py and other imports are) into the container
COPY ./lambda_src/ ${LAMBDA_TASK_ROOT}

# Command to run the Lambda function when invoked by AWS Lambda
CMD ["lambda_function.lambda_handler"]
