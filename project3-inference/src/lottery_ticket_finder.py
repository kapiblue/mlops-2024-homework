import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
from src.conv6 import weight_init


class LotteryTicketFinder:
    def __init__(
        self,
        model: nn.Module,
        train_params: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        # Initialize model weights
        weight_init(model)
        self.device = device
        self.model = model.to(self.device)
        # Save initial weights
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        self.weights_folder = "weights"
        torch.save(
            {"state_dict": self.initial_state_dict},
            os.path.join(os.getcwd(), self.weights_folder, "initial_state_dict.pth"),
        )
        self.learning_rate = train_params["learning_rate"]
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=train_params["learning_rate"]
        )
        self.criterion = criterion

        self.train_loader = train_loader
        self.train_loader_iter = iter(train_loader)
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_iterations = int(train_params["train_iterations"])

        self.masks = None

    def run(
        self,
        pruning_iterations: int,
        validation_frequency: int,
        history: dict,
        prune_amount: float = 0.2,
        early_stop_N: int = 1,
    ):
        """
        Runs the lottery ticket experiment.
        """

        histories = [history.copy() for _ in range(pruning_iterations)]

        # Track the percent of remaining weights
        p_weights_remaining = 100
        for prune_iter in range(pruning_iterations):
            # Pruning before training
            if not prune_iter == 0:
                print(f"Pruning the model at iter {prune_iter}")
                self.masks = self.prune_by_percentile(amount=prune_amount)
                p_weights_remaining *= 1.0 - prune_amount

                # Reinitialize model weights to original ones after pruning
                print("Reinitialize model weights to original ones after pruning...")
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(
                            os.getcwd(), self.weights_folder, f"initial_state_dict.pth"
                        )
                    )["state_dict"]
                )
                self.model = self.model.to(self.device)

                print("Applying the mask to the pruned model...")
                self.apply_mask()
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.learning_rate
                )

            # Train and validate
            pbar = tqdm(range(self.train_iterations))
            # Track best validation accuracy
            best_acc = 0
            is_early_stopping = False
            early_stop_counter = 0
            save_model_filename = None
            for train_iter in pbar:
                is_last_iter = train_iter == self.train_iterations - 1
                # Train one iteration
                loss, acc = self.train_one_iter()
                pbar.set_description(
                    f"Iteration {train_iter}/{self.train_iterations} Training Loss {loss:.6f} Training Accuracy {acc:.2f}"
                )
                # Evaluate model
                if (
                    train_iter % validation_frequency == 0 and train_iter > 0
                ) or is_last_iter:
                    # Store training loss and accuracy
                    histories[prune_iter]["train_loss"].append(loss)
                    histories[prune_iter]["train_accuracy"].append(acc)
                    # Evaluate on the validation set
                    val_loss, val_acc = self.evaluate(self.val_loader)
                    print(
                        f"\nValidation at {train_iter}, Loss {val_loss}, Accuracy {val_acc}."
                    )
                    histories[prune_iter]["valid_loss"].append(val_loss)
                    histories[prune_iter]["valid_accuracy"].append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        early_stop_counter = 0
                    # Check early stopping criterion
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stop_N:
                            is_early_stopping = True
                            print(f"\nEarly stopping at iteration {train_iter}")
                            save_model_filename = (
                                f"model_pi_{prune_iter}_ti_{train_iter}.pth"
                            )
                            self.save_model(prune_iter, train_iter, save_model_filename)
                # Save when stopping or in the last iteration
                if is_early_stopping or is_last_iter:
                    test_loss, test_acc = self.evaluate(
                        self.test_loader, save_model_filename
                    )
                    print(
                        f"\nTesting the model at iteration {train_iter}, Test Accuracy {test_acc}."
                    )
                    histories[prune_iter]["test_accuracy"] = test_acc
                    histories[prune_iter]["early_stop_iter"] = train_iter
                    histories[prune_iter]["p_weights_remaining"] = p_weights_remaining
                    # Save the model in the last iteration
                    if not is_early_stopping:
                        save_model_filename = (
                            f"model_pi_{prune_iter}_ti_{train_iter}.pth"
                        )
                        self.save_model(prune_iter, train_iter, save_model_filename)
                    break

        self.histories = histories

    def train_one_iter(self):
        """
        Trains for one iteration (one batch)
        """

        # get the inputs; data is a list of [inputs, labels]
        try:
            inputs, labels = next(self.train_loader_iter)
        except StopIteration:
            # reinitialize train loader iterator once it reached the end
            self.train_loader_iter = iter(self.train_loader)
            inputs, labels = next(self.train_loader_iter)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        train_acc = (torch.max(outputs.data, 1)[1] == labels).sum().item() / len(labels)
        return train_loss, train_acc * 100

    def evaluate(self, data_loader, model_path=None):
        if model_path is not None:
            model = self.model_class()
            model.load_state_dict(torch.load(model_path))
            model.cuda()
        else:
            model = self.model
        model.eval()
        valid_batch_losses = []
        valid_batch_acc = []
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

                acc = (torch.max(outputs.data, 1)[1] == labels).sum().item() / len(
                    labels
                )
                valid_batch_losses.append(loss.item())
                valid_batch_acc.append(acc)

            valid_loss = np.sum(valid_batch_losses) / len(valid_batch_losses)
            valid_acc = np.sum(valid_batch_acc) / len(valid_batch_acc)
            return valid_loss, valid_acc * 100

    def save_model(self, prune_iter: int, train_iter: int, save_model_filename: str):
        print(f"Model saved at training iteration {train_iter}")
        torch.save(
            self.model.state_dict(),
            os.path.join(os.getcwd(), self.weights_folder, save_model_filename),
        )

    def prune_by_percentile(self, amount=0.2):
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        # Get the weights and calculate the threshold
        all_weights = torch.cat(
            [torch.flatten(module.weight.data) for module, _ in parameters_to_prune]
        )
        threshold = torch.quantile(torch.abs(all_weights), amount)

        # Create a mask and prune the weights
        masks = {}
        for module, param in parameters_to_prune:
            mask = torch.abs(module.weight.data) > threshold
            masks[module] = mask
            module.weight.data.mul_(mask.float())

        return masks

    def apply_mask(self):
        if self.masks is not None:
            for module, mask in self.masks.items():
                module.weight.data.mul_(mask.float())
