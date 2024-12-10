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

N_EPOCHS = 10
BATCH_SIZE = 60
INPUT_CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = os.path.join(os.getcwd(), "models", "model_float32.pth")


def get_params_to_prune(architecture):
    params_to_prune = (
        (architecture.conv1, "weight"),
        (architecture.conv2, "weight"),
        (architecture.conv3, "weight"),
        (architecture.conv4, "weight"),
        (architecture.conv5, "weight"),
        (architecture.conv6, "weight"),
        (architecture.fc1, "weight"),
        (architecture.fc2, "weight"),
        (architecture.fc3, "weight"),
    )
    return params_to_prune


if __name__ == "__main__":

    _, _, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6(INPUT_CHANNELS)
    architecture.load_state_dict(torch.load(weights_path))

    parameters_to_prune = get_params_to_prune(architecture)

    ################### GLOBAL L1 PRUNING ###################

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )

    print_conv6_sparsity(architecture, pruning_method="global L1")

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    trainer.test(model, test_loader)

    results = {"global_l1_pruning": model.test_results}

    results_path = os.path.join(os.getcwd(), "results", "l1_pruning_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    ################### GLOBAL RANDOM PRUNING ###################

    architecture = Conv6(INPUT_CHANNELS)
    architecture.load_state_dict(torch.load(weights_path))
    parameters_to_prune = get_params_to_prune(architecture)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=0.2,
    )

    print_conv6_sparsity(architecture, pruning_method="global random")

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    trainer.test(model, test_loader)

    results = {"global_random_pruning": model.test_results}

    results_path = os.path.join(os.getcwd(), "results", "random_pruning_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

        ################### PRUNING ACCORDING TO TYPE ###################

    architecture = Conv6(INPUT_CHANNELS)
    architecture.load_state_dict(torch.load(weights_path))

    for name, module in architecture.named_modules():
        # prune 15% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", n=2, dim=0, amount=0.1)
        # prune 30% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", n=2, dim=0, amount=0.3)

    print_conv6_sparsity(architecture, pruning_method="type aware structured")

    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    trainer.test(model, test_loader)

    results = {"type_aware_pruning": model.test_results}

    results_path = os.path.join(
        os.getcwd(), "results", "type_aware_pruning_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f)
