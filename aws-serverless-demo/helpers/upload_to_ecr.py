import boto3
import docker
import subprocess

REGION = "eu-west-2"

ecr_client = boto3.client("ecr")
docker_client = docker.from_env()
aws_account_id = boto3.client("sts").get_caller_identity().get("Account")


def create_or_get_ecr_repository(repository_name):
    try:
        response = ecr_client.create_repository(repositoryName=repository_name)
        print(f"Repository {repository_name} created.")
        repository_uri = response["repository"]["repositoryUri"]
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository {repository_name} already exists.")
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository_uri = response["repositories"][0]["repositoryUri"]
    return repository_uri


# Authenticate Docker to ECR
def authenticate_ecr(region, aws_account_id):
    token = ecr_client.get_authorization_token()["authorizationData"][0]
    # ecr_password = token['authorizationToken']
    ecr_url = token["proxyEndpoint"]
    print(ecr_url)
    login_command = f"aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin {ecr_url}"
    print(login_command)
    subprocess.run(login_command, shell=True, check=True)
    print("Docker authenticated with ECR")


def build_and_push_docker_image(repository_uri, repository_name, image_tag):
    print("Building Docker image:")
    docker_client.images.build(
        path="/home/ubuntu/janek/aws-serverless-demo/",
        tag=f"{repository_name}:{image_tag}",
    )
    print("Docker image success")

    full_image_name = f"{repository_uri}:{image_tag}"
    docker_client.images.get(f"{repository_name}:{image_tag}").tag(full_image_name)
    print(f"Tagged image: {full_image_name}")

    print("Pushing to ECR:")
    for line in docker_client.images.push(
        repository_uri, tag=image_tag, stream=True, decode=True
    ):
        print(line)
    


if __name__ == "__main__":
    repository_name = "hello-world-lambda"
    image_tag = "latest"

    repository_uri = create_or_get_ecr_repository(repository_name)
    # print(f"Repository URI: {repository_uri}")
    authenticate_ecr(REGION, aws_account_id)

    build_and_push_docker_image(repository_uri, repository_name, image_tag)
