import pandas as pd
from torch.utils.data import DataLoader, Dataset


def load_data(path, nrows=None):
    df = pd.read_csv(path, sep=",", header=0, nrows=nrows)
    df.dropna(how="all", subset=["compound_id", "standard_inchi_key"], inplace=True)
    df.dropna(how="all", subset=["target_id", "gene_names"], inplace=True)
    return df


def
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


if __name__ == '__main__':
    path = "/home/olivier/data/idg_dream/DTC_data.csv"
    df = load_data(path, 1000)
    print(df.head())



