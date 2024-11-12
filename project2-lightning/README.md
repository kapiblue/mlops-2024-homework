# Environment setup
1. Python 3.11
2. `python -m venv venv`
3. `source activate venv/bin/activate`
4. `pip install optuna`
5. `pip install -r requirements.txt`

# Data 
I created the custom, very small and simple dataset based on my (own ;)) previous topography project. The dataset contains 200 images from two classes (binary classification).
1. `cd data && gdown https://drive.google.com/uc?id=1-YFEQ4ZSLHr7DceqqcCPmw2PwRR1OkNm`
2. `unzip mlops_dataset.zip`


# Docker

1. Download data according to the instructions above.
2. Run `wandb init` in the `project2-lightning` directory and configure wandb.
3. Build your docker `docker build -t ptl_train .`
4. Run the container with wandb.

```bash
wandb docker-run --rm --name ptl_cont ptl_train
```

https://docs.wandb.ai/ref/cli/wandb-docker-run
