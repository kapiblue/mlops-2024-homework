from torch.quantization import prepare, convert
import torch

# Calibrate function for quantization
def calibrate_model(model, data_loader, device):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            x, _ = batch
            model(x.to(device))
            break


# Quantize the trained model
def quantize_model(trained_model, data_loader, device):
    print("\nQuantizing the model with static quantization...")

    # Set quantization config
    trained_model.qconfig = torch.quantization.get_default_qconfig(
        "x86"
    )  # fbgemm is optimized for x86 CPUs

    # Fuse layers for better quantization (Conv + ReLU)
    fused_model = torch.quantization.fuse_modules(
        trained_model,
        [
            ["conv1", "relu1"],
            ["conv2", "relu2"],
            ["conv3", "relu3"],
            ["conv4", "relu4"],
            ["conv5", "relu5"],
            ["conv6", "relu6"],
        ],
    )

    # Prepare the model for static quantization
    prepare(fused_model, inplace=True)

    # Calibrate the model using training/validation data
    calibrate_model(fused_model, data_loader, device)

    # Convert the model to quantized format
    quantized_model = convert(fused_model, inplace=False)

    return quantized_model
