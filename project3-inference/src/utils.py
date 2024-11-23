import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


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
