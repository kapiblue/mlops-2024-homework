import time
import os
import torch
import lightning as L
import torch.nn as nn
import torchmetrics
from src.conv6 import Conv6, weight_init
from src.get_data import get_dataloaders
from src.utils import create_folders


BATCH_SIZE = 60
LR = 3e-4
INPUT_CHANNELS = 3
N_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = os.path.join(os.getcwd(), "models", "model_float32.pth")


class CifarClassifier(L.LightningModule):
    def __init__(self, model, learning_rate=LR):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            DEVICE
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            DEVICE
        )
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
            DEVICE
        )
        self.test_f1 = torchmetrics.F1Score(
            num_classes=10, average="macro", task="multiclass"
        ).to(DEVICE)

        # Variable to store batch inference times
        self.batch_times = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.val_acc(y_hat, y)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Measure inference time
        start_time = time.time()
        y_hat = self.model(x)
        end_time = time.time()

        # Record batch inference time
        self.batch_times.append(end_time - start_time)

        loss = self.criterion(y_hat, y)
        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        return loss

    def on_test_epoch_end(self):
        # Print metrics after test
        acc = self.test_acc.compute()
        f1 = self.test_f1.compute()
        avg_time = sum(self.batch_times) / len(self.batch_times)

        self.test_results = {
            "accuracy": acc.item(),
            "f1_score": f1.item(),
            "average_inference_time": avg_time,
        }

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Average Batch Inference Time: {avg_time:.6f} seconds")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":

    create_folders(["models", "plots", "results"])
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    architecture = Conv6(INPUT_CHANNELS)
    weight_init(architecture)

    model = CifarClassifier(architecture)

    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    print("Training the Original Model...")
    trainer.fit(model, train_loader, val_loader)
    print("Saving the model to state_dict...")
    torch.save(model.model.state_dict(), weights_path)
