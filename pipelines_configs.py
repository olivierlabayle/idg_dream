from skorch.dataset import CVSplit
from torch.optim import *

CONFIG = {
    "baseline":
        {
            "kmer_size": 3,
            "radius": 2,
            "ecfp_dim": 2 ** 10,
            "embedding_dim": 10,
            "lr": 0.1,
            "max_epochs": 5,
            "device": None,
            "loaders": False,
            "train_split": None
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
            "device": None,
            "train_split": CVSplit(3)
        }
}
