import time
import torch
from conv6 import Conv6, weight_init
from get_data import get_dataloaders
import lightning as L
import torch.nn as nn
import torchmetrics
import json
from q_utils import quantize_model
import matplotlib.pyplot as plt


BATCH_SIZE = 60
LR = 3e-4
INPUT_CHANNELS = 3
N_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

architecture = Conv6(INPUT_CHANNELS)
weight_init(architecture)


class CifarClassifier(L.LightningModule):
    def __init__(self, model, learning_rate=LR):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        self.test_f1 = torchmetrics.F1Score(num_classes=10, average="macro", task="multiclass").to(DEVICE)

        # Variable to store batch inference times
        self.batch_times = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
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
        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
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


model = CifarClassifier(architecture)

trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

print("Training the Original Model...")
trainer.fit(model, train_loader, val_loader)
print("Saving the model to state_dict...")
torch.save(model.model.state_dict(), "../../models/model_float32.pth")


print("Testing the Original Model...")
DEVICE = "cpu"
architecture = Conv6(INPUT_CHANNELS)
architecture.load_state_dict(torch.load("../../models/model_float32.pth"))
model = CifarClassifier(architecture)
trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

_, _, test_loader = get_dataloaders(BATCH_SIZE)
trainer.test(model, test_loader)

float_model_results = model.test_results




print("Quantizing the Model...")

q_model = Conv6(INPUT_CHANNELS, quantize=True)
q_model.load_state_dict(torch.load("../../models/model_float32.pth"))

DEVICE = "cpu"
quantized_model = quantize_model(q_model, train_loader, DEVICE)

print("Testing the Quantized Model...")
quantized_classifier = CifarClassifier(quantized_model)
quantized_classifier.batch_times = []  # Reset batch times
_, _, test_loader = get_dataloaders(BATCH_SIZE)
trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)
trainer.test(quantized_classifier, test_loader)

quantized_model_results = quantized_classifier.test_results


results = {
    "float32": float_model_results,
    "quantized": quantized_model_results,
}

with open("../../models/quantization_results.json", "w") as f:
    json.dump(results, f)

float_accuracy = results['float32']['f1_score']
int8_accuracy = results['quantized']['f1_score']
float_inference_time = results['float32']['average_inference_time']
int8_inference_time = results['quantized']['average_inference_time']

# Labels
labels = ['Float32', 'Int8']

# Create a figure with two subplots (one row, two columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Bar plot for accuracy
ax1.bar(labels, [float_accuracy, int8_accuracy], color=['blue', 'orange'])
ax1.set_title('Model F1_score')
ax1.set_ylabel('f1_score')
ax1.set_ylim(0, 1)  # Assuming accuracy is between 0 and 1

# Bar plot for inference time
ax2.bar(labels, [float_inference_time, int8_inference_time], color=['blue', 'orange'])
ax2.set_title('Average Inference Time')
ax2.set_ylabel('Inference Time (seconds)')
ax2.set_ylim(0, max(float_inference_time, int8_inference_time) * 1.1)  # Scale y-axis based on max time
fig.suptitle('Float32 vs. int8', fontsize=16)
# Adjust layout and show plot
plt.tight_layout()
plt.savefig('float32_vs_int8.png')


