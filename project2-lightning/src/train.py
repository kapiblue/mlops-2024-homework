from train_module import ResNetClassifier
from data_module import ImageDataModule
from torchvision import models
import torch.nn as nn
import pandas as pd
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import argparse
import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('run_name', type=str, help='WandB run name')

DATA_PATH = "data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    num_epochs = 2 # trial.suggest_int("num_epochs", 3, 10)

    wandb_logger = WandbLogger(
        project="MLOps2",
        name=f"trial_{trial.number}",
        entity="jankowskidaniel06-put",
    )

    architecture = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 1)
    )
    architecture.fc = classifier

    model = ResNetClassifier(architecture, learning_rate=learning_rate)

    df = pd.read_csv(DATA_PATH + "moved_parameters_mlops.csv")
    dm = ImageDataModule(
        main_path=DATA_PATH,
        data=df,
        batch_size=batch_size,
        num_workers=0,
    )

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/loss")

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator=DEVICE,
        logger=wandb_logger,
        callbacks=[pruning_callback],
        enable_checkpointing=False,
        log_every_n_steps=50
    )

    trainer.fit(model, dm)

    val_loss = trainer.callback_metrics.get("val/loss")
    if val_loss is None:
        return float('inf')

    return val_loss.item()

if __name__ == "__main__":

    study = optuna.load_study(
        study_name="distributed-example", storage="mysql://root@localhost/example"
    )
    study.optimize(objective, n_trials=3)
    print("Number of ended trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Validation loss: ", trial.value)
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
