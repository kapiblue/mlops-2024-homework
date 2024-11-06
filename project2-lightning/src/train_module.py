import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision import models

# Define the model architecture
model = models.resnet18(pretrained=True)

classifier = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, 1)
)

model.fc = classifier


# Define the LightningModule
class ResNetClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_hat = self.model(x)
        
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)