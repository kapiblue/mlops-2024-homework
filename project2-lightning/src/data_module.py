import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the ImageDataset class
class ImageDataset(Dataset):
    def __init__(self, path: str, data: pd.DataFrame, transform=None) -> None:
        self.image_paths = np.array([path + filename for filename in data['filename'].to_numpy()])
        self.labels = data['epsilon'].to_numpy()
        self.transform = transform

    def __getitem__(self, inx: int) -> tuple:
        image_path = self.image_paths[inx]
        target = self.labels[inx]
        image = Image.open(image_path)
        image = np.array(image)
        # repeat grayscale value three times for all RGB channels
        image = np.repeat(image[..., np.newaxis], 3, -1)
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(target, dtype=torch.float32)
        return image, target

    def __len__(self) -> int:
        return len(self.image_paths)

# Define the LightningDataModule class
class ImageDataModule(L.LightningDataModule):
    def __init__(self, main_path: str, data: pd.DataFrame, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.main_path = main_path
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define the transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        train, valid = train_test_split(self.data, test_size=0.2, random_state=12, shuffle=True, stratify=self.data['epsilon'])
        
        # Create the training and validation datasets
        self.train_dataset = ImageDataset(path=self.main_path, data=train, transform=self.transform)
        self.valid_dataset = ImageDataset(path=self.main_path, data=valid, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
