from skorch.dataset import CVSplit
from torch.optim import *

CONFIG = {
    "baseline_net":
        {
            "kmer_size": 3,
            "radius": 4,
            "ecfp_dim": 2 ** 20,
            "embedding_dim": 200,
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "max_epochs": 500,
            "device": None,
            "loaders": False,
            "optimizer": Adam,
            "train_split": CVSplit(0.2, random_state=0)
        },
    "bilstm_fingerprint":
        {
            "engine": None,
            "loaders": False,
            "kmer_size": 3,
            "radius": 4,
            "ecfp_dim": 2 ** 20,
            "hidden_size": 100,
            "mlp_sizes": (100, 100),
            "embedding_dim": 100,
            "max_epochs": 500,
            "lr": 1e-5,
            "optimizer": Adam,
            "weight_decay": 1e-5,
            "lstm_dropout": 0,
            "device": None,
            "train_split": CVSplit(0.2, random_state=0)
        }
}
