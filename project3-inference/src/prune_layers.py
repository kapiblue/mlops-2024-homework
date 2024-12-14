import json
import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune
import lightning as L
import os
from src.conv6 import Conv6
from src.q_utils import quantize_model
from src.get_data import get_dataloaders
from src.train import CifarClassifier
from src.utils import print_conv6_sparsity
import torch.nn as nn
import torch
import torch.nn.init as init

N_EPOCHS = 10
BATCH_SIZE = 60
INPUT_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = os.path.join(os.getcwd(), "models", "model_float32.pth")


class Conv6PrunedFC(nn.Module):
    def __init__(self, input_channels: int, quantize: bool = False):
        super(Conv6PrunedFC, self).__init__()
        self.quantize = quantize
        self.quant = torch.ao.quantization.QuantStub()

        # Six convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # Flatten
        self.flatten = nn.Flatten()
        # Three fully-connected layers
        self.fc1 = nn.Linear(4096, 256)
        # self.fc2 = nn.Linear(256, 256) # Prune this layer
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x)) # Prune this layer
        x = self.fc3(x)
        if self.quantize:
            x = self.dequant(x)
        return x


class Conv6PrunedCNN(nn.Module):
    def __init__(self, input_channels: int, quantize: bool = False):
        super(Conv6PrunedCNN, self).__init__()
        self.quantize = quantize
        self.quant = torch.ao.quantization.QuantStub()

        # Six convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.relu2 = nn.ReLU()  # Prune this layer
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # Flatten
        self.flatten = nn.Flatten()
        # Three fully-connected layers
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 256) # Prune this layer
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.relu1(self.conv1(x))
        # x = self.relu2(self.conv2(x)) # Prune this layer
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) # Prune this layer
        x = self.fc3(x)
        if self.quantize:
            x = self.dequant(x)
        return x


class Conv6PrunedCNNFC(nn.Module):
    def __init__(self, input_channels: int, quantize: bool = False):
        super(Conv6PrunedCNNFC, self).__init__()
        self.quantize = quantize
        self.quant = torch.ao.quantization.QuantStub()

        # Six convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.relu2 = nn.ReLU()  # Prune this layer
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # Flatten
        self.flatten = nn.Flatten()
        # Three fully-connected layers
        self.fc1 = nn.Linear(4096, 256)
        # self.fc2 = nn.Linear(256, 256) # Prune this layer
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.relu1(self.conv1(x))
        # x = self.relu2(self.conv2(x)) # Prune this layer
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x)) # Prune this layer
        x = self.fc3(x)
        if self.quantize:
            x = self.dequant(x)
        return x

def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

if __name__ == "__main__":


    ############### CNN LAYER PRUNING ################
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6PrunedCNN(INPUT_CHANNELS)
    # architecture.load_state_dict(torch.load(weights_path), strict=False)
    weight_init(architecture)

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    print("Training the Model...")
    trainer.fit(model, train_loader, val_loader)

    DEVICE = "cpu"
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)
    print("Testing the pruned model...")
    trainer.test(model, test_loader)

    results = {"layer_cnn_pruning": model.test_results}

    results_path = os.path.join(os.getcwd(), "results", "cnn_layer_pruning.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    
    ############### FC LAYER PRUNING ################
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6PrunedFC(INPUT_CHANNELS)
    # architecture.load_state_dict(torch.load(weights_path), strict=False)
    weight_init(architecture)

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    print("Training the Model...")
    trainer.fit(model, train_loader, val_loader)

    DEVICE = "cpu"
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)
    print("Testing the pruned model...")
    trainer.test(model, test_loader)

    results = {"layer_fc_pruning": model.test_results}

    results_path = os.path.join(os.getcwd(), "results", "fc_layer_pruning.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    

    ############### CNN & FC LAYER PRUNING ################
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6PrunedCNNFC(INPUT_CHANNELS)
    # architecture.load_state_dict(torch.load(weights_path), strict=False)
    weight_init(architecture)

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    print("Training the Model...")
    trainer.fit(model, train_loader, val_loader)

    DEVICE = "cpu"
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)
    print("Testing the pruned model...")
    trainer.test(model, test_loader)

    results = {"layer_fc_cnn_pruning": model.test_results}

    results_path = os.path.join(os.getcwd(), "results", "fc_cnn_layer_pruning.json")
    with open(results_path, "w") as f:
        json.dump(results, f)