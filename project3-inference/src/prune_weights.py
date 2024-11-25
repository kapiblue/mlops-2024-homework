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

N_EPOCHS = 10
BATCH_SIZE = 60
INPUT_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = os.path.join(os.getcwd(), "models", "model_float32.pth")
results_path = os.path.join(os.getcwd(), "results", "weights_pruning_results.json")


if __name__ == "__main__":

    _, _, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6(INPUT_CHANNELS)
    architecture.load_state_dict(torch.load(weights_path))

    parameters_to_prune = (
        (architecture.conv1, 'weight'),
        (architecture.conv2, 'weight'),
        (architecture.conv3, 'weight'),
        (architecture.conv4, 'weight'),
        (architecture.conv5, 'weight'),
        (architecture.conv6, 'weight'),
        (architecture.fc1, 'weight'),
        (architecture.fc2, 'weight'),
        (architecture.fc3, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    trainer.test(model, test_loader)

    results = {
        "global_pruning" : model.test_results
    }

    with open(results_path, "w") as f:
        json.dump(results, f)