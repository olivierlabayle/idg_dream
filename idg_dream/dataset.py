import pandas as pd
from torch.utils.data import DataLoader, Dataset


class IDGDataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.size = len(self.data)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        features, target = sample
        features = self.transform(features)
        return features, target




