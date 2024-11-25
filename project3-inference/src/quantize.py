import json
import matplotlib.pyplot as plt
import torch
import lightning as L
import os
from src.conv6 import Conv6
from src.q_utils import quantize_model
from src.get_data import get_dataloaders
from src.train import CifarClassifier

N_EPOCHS = 10
BATCH_SIZE = 60
INPUT_CHANNELS = 3
DEVICE = "cpu"

weights_path = os.path.join(os.getcwd(), "models", "model_float32.pth")
results_path = os.path.join(os.getcwd(), "results", "quantization_results.json")

if __name__ == "__main__":

    print("Testing the Original Model...")
    train_loader, _, test_loader = get_dataloaders(BATCH_SIZE)
    architecture = Conv6(INPUT_CHANNELS)
    architecture.load_state_dict(torch.load(weights_path))
    model = CifarClassifier(architecture)
    trainer = L.Trainer(max_epochs=N_EPOCHS, accelerator=DEVICE)

    trainer.test(model, test_loader)

    float_model_results = model.test_results

    print("Quantizing the Model...")

    q_model = Conv6(INPUT_CHANNELS, quantize=True)
    q_model.load_state_dict(torch.load(weights_path))

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

    with open(results_path, "w") as f:
        json.dump(results, f)

    float_accuracy = results["float32"]["f1_score"]
    int8_accuracy = results["quantized"]["f1_score"]
    float_inference_time = results["float32"]["average_inference_time"]
    int8_inference_time = results["quantized"]["average_inference_time"]

    # Labels
    labels = ["Float32", "Int8"]

    # Create a figure with two subplots (one row, two columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Bar plot for accuracy
    ax1.bar(labels, [float_accuracy, int8_accuracy], color=["blue", "orange"])
    ax1.set_title("Model F1_score")
    ax1.set_ylabel("f1_score")
    ax1.set_ylim(0, 1)  # Assuming accuracy is between 0 and 1

    # Bar plot for inference time
    ax2.bar(
        labels, [float_inference_time, int8_inference_time], color=["blue", "orange"]
    )
    ax2.set_title("Average Inference Time")
    ax2.set_ylabel("Inference Time (seconds)")
    ax2.set_ylim(
        0, max(float_inference_time, int8_inference_time) * 1.1
    )  # Scale y-axis based on max time
    fig.suptitle("Float32 vs. int8", fontsize=16)
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "plots", "float32_vs_int8.png"))
