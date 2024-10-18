# We should be using loggers instead of print statements
# https://stackoverflow.com/questions/37703609/using-python-logging-with-aws-lambda

def lambda_handler(event, context):
    print(event)
    print(context.__dict__)

    return {
        'statusCode': 200,
        'body': 'Hello, World!'
    }



if __name__ == "__main__":
    print("Hello World")
