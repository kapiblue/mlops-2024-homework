import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import torch

from src.conv6 import Conv6


def plot_sample_images(images):
    plt.figure(figsize=(15, 6))
    plt.axis("off")
    plt.imshow(make_grid(images, nrow=15).permute((1, 2, 0)))
    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), "plots", "sample_images.png")
    plt.savefig(save_path)


def plot_ltf_results(histories: list, p_weights_remaining: list):
    fig, (ax1, ax2) = plt.subplots(figsize=(20, 7), ncols=2)
    early_stop_iters = [x["early_stop_iter"] for x in histories]
    test_acc = [x["test_accuracy"] for x in histories]
    p_weights_remaining = ["{:.2f}".format(x["p_weights_remaining"]) for x in histories]
    ax1.plot(p_weights_remaining, early_stop_iters)
    ax1.set_xlabel("Percent of weights remaining")
    ax1.set_ylabel("Early stopping iteration (val)")
    ax2.plot(p_weights_remaining, test_acc)
    ax2.set_xlabel("Percent of weights remaining")
    ax2.set_ylabel("Accuracy on the test set")
    fig.tight_layout()
    plt.legend()
    save_path = os.path.join(os.getcwd(), "plots", "lth_results.png")
    plt.savefig(save_path)


# Write a function to create folders if they don't exist
def create_folders(folders: list):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def print_conv6_sparsity(model: Conv6, pruning_method: str = ""):

    # Put modules in a dictionary
    modules = {
        "conv1": model.conv1,
        "conv2": model.conv2,
        "conv3": model.conv3,
        "conv4": model.conv4,
        "conv5": model.conv5,
        "conv6": model.conv6,
        "fc1": model.fc1,
        "fc2": model.fc2,
        "fc3": model.fc3,
    }
    # Calculate sparsity in a loop and collect results
    sparsities = {}
    total_n_zero_elements = 0
    total_n_elements = 0
    for module_name, module in modules.items():
        n_zero_elements = torch.sum(module.weight == 0)
        n_elements = module.weight.nelement()
        sparsity = 100.0 * float(n_zero_elements) / float(n_elements)
        sparsities[module_name] = sparsity
        total_n_zero_elements += n_zero_elements
        total_n_elements += n_elements

    global_sparsity = 100.0 * float(total_n_zero_elements) / float(total_n_elements)

    # Print results
    for module_name, sparsity in sparsities.items():
        print(f"Sparsity in {module_name} : {sparsity:.2f}%")

    print(f"Global sparsity for {pruning_method} pruning: {global_sparsity:.2f}%")

    save_path = os.path.join(
        os.getcwd(), "plots", f"local_sparsities_{pruning_method}.png"
    )
    plt.bar(sparsities.keys(), sparsities.values())
    plt.ylabel("Sparsity (%)")
    plt.title(
        f"Local sparsities for {pruning_method} pruning.\nGlobal sparsity: {global_sparsity:.2f}%"
    )
    plt.savefig(save_path)
    plt.close()
