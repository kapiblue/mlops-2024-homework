from train_module import ResNetClassifier
from data_module import ImageDataModule
from torchvision import models
import torch.nn as nn
import pandas as pd
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('run_name', type=str, help='WandB run name')

DATA_PATH = "data/"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

wandb.init(
    project="MLOps2",
    entity="jankowskidaniel06-put",
    name=parser.parse_args().run_name,
)

logger = WandbLogger(
    project="MLOps2",
)

# Model architecture
architecture = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

classifier = nn.Sequential(
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.ReLU(), nn.Linear(512, 1)
)

architecture.fc = classifier

# Initialize the LightningModule
model = ResNetClassifier(architecture)


df = pd.read_csv(DATA_PATH + "moved_parameters_mlops.csv")
dm = ImageDataModule(main_path="data/", data=df, batch_size=BATCH_SIZE, num_workers=0)

trainer = L.Trainer(max_epochs=1, accelerator=DEVICE, logger=logger)
trainer.fit(model, dm)
