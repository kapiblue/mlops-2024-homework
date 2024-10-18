# Start from the official AWS Lambda Python 3.8 image
FROM public.ecr.aws/lambda/python:3.8

# Install boto3 for interacting with AWS services and any other dependencies listed in requirements.txt (if present)
RUN pip install boto3

# If you have a requirements.txt file for additional dependencies, uncomment the following line:
# RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the entire directory (where lambda_function.py and other imports are) into the container
COPY ./lambda_src/ ${LAMBDA_TASK_ROOT}

# RUN python my_script.py 
# Run a python script that will predownload the model. 

# Command to run the Lambda function when invoked by AWS Lambda
CMD ["lambda_function.lambda_handler"]
