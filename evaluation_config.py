GRIDS = {
    "baseline": [
        {
            "kmer_size": [3, 5, 7],
            "radius": [2, 3],
            "ecfp_dim": [2 ** 10],
            "embedding_dim": [10, 20],
            "lr": 0.1,
            "max_epochs": [5, 10],
            "device": 'cpu',
            "loaders": False,
            "train_split": None
        }
    ],
    "linear_regression": [{""}]
}