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
    "linear_regression": [
        {
            "linear_regression__alpha": [1e-3, 1e-1, 1, 10, 100],
            "encode_proteins__kmer_size": [3],
            "encode_ecfp__radius": [2],
            "encode_ecfp__dim": [2 ** 10]
        }
    ]
}
