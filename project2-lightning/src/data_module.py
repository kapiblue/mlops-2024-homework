import lightning.pytorch as pl
import pandas as pd


# Define the data module
class TopographyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, params_df: pd.DataFrame):
        super().__init__()
        self.data_dir = data_dir

    def train_dataloader(self):
        # Return the training dataloader
        pass
