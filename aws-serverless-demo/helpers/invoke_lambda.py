import boto3
import json
import requests

REGION = "eu-west-2"

client = boto3.client("lambda")


def invoke_lambda_using_boto3(function_name: str, payload: dict):
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )

    response_payload = json.loads(response["Payload"].read())
    print("Response from Lambda:", response_payload)


def invoke_lambda(lambda_url: str, payload: dict, headers=None):
    if headers is None:
        headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(lambda_url, json=payload, headers=headers)
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Error invoking Lambda: {response.status_code} - {response.text}")

    except requests.RequestException as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    payload = {}
    # function_name = 'hello_world_lambda'
    # invoke_lambda_using_boto3(function_name, payload)

    lambda_url = "https://jtlek3cslf3z5u6ief35fv76uy0yuhlv.lambda-url.eu-west-2.on.aws/"
    invoke_lambda(lambda_url, payload)
