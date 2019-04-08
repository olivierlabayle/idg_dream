from skorch.dataset import CVSplit
from torch.optim import *

CONFIG = {
    "baseline_net":
        {
            "kmer_size": 3,
            "radius": 4,
            "ecfp_dim": 2 ** 20,
            "embedding_dim": 20,
            "lr": 0.1,
            "weight_decay":0.1,
            "max_epochs": 50,
            "device": None,
            "loaders": False,
            "optimizer": Adam,
            "train_split": CVSplit(3)
        },
    "bilstm_fingerprint":
        {
            "engine": None,
            "loaders": False,
            "kmer_size": 3,
            "radius": 4,
            "ecfp_dim": 2 ** 20,
            "hidden_size": 20,
            "mlp_sizes": (20, 10),
            "embedding_dim": 20,
            "max_epochs": 10,
            "lr": 1,
            "optimizer": Adam,
            "weight_decay":0.1,
            "lstm_dropout":0.2,
            "device": None,
            "train_split": CVSplit(3)
        }
}
