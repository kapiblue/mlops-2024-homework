import torch
import torch.nn as nn
import lightning as L


# Define the LightningModule
class ResNetClassifier(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()

        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('train_steps/loss', loss, on_step=True, on_epoch=False)
        self.log('train/loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()

        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('val_steps/loss', loss, on_step=True, on_epoch=False)
        self.log('val/loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)