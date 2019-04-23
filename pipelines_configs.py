from skorch.dataset import CVSplit
from torch.optim import *

CONFIG = {
    "BaselineNetFactory":
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
    "LinearRegressionFactory":
        {'alpha': 50,
         'ecfp_dim': 2**22,
         'radius': 6,
         'kmer_size': 4}
    ,
    "BiLSTMFingerprintFactory":
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
        },
    "GraphBiLSTMFactory":
        {
            "kmer_size": 3,
            "graph_hidden_dim": 50,
            "embedding_dim": 50,
            "lstm_hidden_size": 50,
            "mlp_sizes": (50,),
            "dropout": 0,
            "graph_layers": 1,
            "max_epochs": 100,
            "lr": 1e-4,
            "optimizer": SGD,
            "device": None,
            "weight_decay": 0,
            "train_split":  CVSplit(0.2, random_state=0)
        }
}
