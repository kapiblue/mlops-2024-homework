import torch.optim as optim
import torch
import torch.nn as nn
from torchinfo import summary
import json

from src.get_data import get_dataloaders
from src.conv6 import Conv6
from src.lottery_ticket_finder import LotteryTicketFinder
from src.utils import plot_ltf_results, plot_sample_images, create_folders


BATCH_SIZE = 60

# Define the training parameters
conv6_params = {
    "learning_rate": 3e-4,
    "batch_size": BATCH_SIZE,
    "model_type": "conv6",
    "train_iterations": 1e4,  # 3e4
}

history = {
    "train_loss": [],
    "valid_loss": [],
    "train_accuracy": [],
    "valid_accuracy": [],
    "test_accuracy": 0,
    "early_stop_iter": 0,
    "p_weights_remaining": 100,
}

input_channels = 3
pruning_iterations = 5
validation_frequency = 750
prune_amount = 0.2

if __name__ == "__main__":
    create_folders(["weights", "plots"])

    print("Running the Lottery Ticket Hypothesis experiment")
    # Get dataloaders for CIFAR-10
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # Save visualization of sample training images
    images, _ = next(iter(train_loader))
    plot_sample_images(images)

    # Initialize the model
    model = Conv6(input_channels)
    # Print the model summary
    summary(model, input_size=(BATCH_SIZE, input_channels, 32, 32))

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Lottery Ticket Finder
    ltf = LotteryTicketFinder(
        model, conv6_params, train_loader, val_loader, test_loader, criterion, device
    )

    ltf.run(
        pruning_iterations=pruning_iterations,
        validation_frequency=validation_frequency,
        history=history,
        prune_amount=prune_amount,
        early_stop_N=2,
    )

    file_path = "lth_results.json"

    # Write the list of dictionaries to the json file
    with open(file_path, "w") as f:
        for entry in ltf.histories:
            f.write(json.dumps(entry) + "\n")

    histories = []
    with open(file_path, "r") as f:
        for entry in f:
            histories.append(json.loads(entry))

    x = 100
    p_weights_remaining = [
        x,
    ]
    for _ in range(pruning_iterations - 1):
        x *= 1 - prune_amount
        p_weights_remaining.append("{:.2f}".format(x))

    plot_ltf_results(histories, p_weights_remaining)
