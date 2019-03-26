GRIDS = {
    "baseline_net": [
        {
            "encode_proteins__kmer_size": [3, 5, 7],
            "encode_ecfp__radius": [2, 3],
            "encode_ecfp__dim": [2 ** 10],
            "baseline_net__module__embedding_dim": [10, 20],
            "baseline_net__optimizer__lr": [0.1, 1],
            "baseline_net__max_epochs": [10, 50],
        }
    ],
    "linear_regression": [
        {
            "linear_regression__alpha": [1e-3, 1e-1, 1, 10, 100, 1000],
            "sparse_encoding__encode_proteins__kmer_size": [3, 5, 7],
            "sparse_encoding__encode_ecfp__radius": [2, 3, 4],
            "sparse_encoding__encode_ecfp__dim": [2 ** 10, 2**15, 2**20]
        }
    ]
}
