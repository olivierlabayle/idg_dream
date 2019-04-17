from torch.optim import *

GRIDS = {
    "BaselineNetFactory": [
        {
            "encode_proteins__kmer_size": [3, 5],
            "encode_ecfp__radius": [4],
            "encode_ecfp__dim": [2 ** 10],
            "baseline_net__module__num_fingerprints": [2 ** 10],
            "baseline_net__optimizer": [Adam],
            "baseline_net__module__embedding_dim": [50],
            "baseline_net__optimizer__lr": [1e-4],
            "baseline_net__max_epochs": [100],
        }
    ],
    "LinearRegressionFactory": [
        {
            "sparse_encoding__encode_proteins__kmer_size": [3],
            "sparse_encoding__encode_ecfp__radius": [6],
            "sparse_encoding__encode_ecfp__dim": [2 ** 22],
            "n_neighbors": [1, 2, 3, 4, 5],
            "metric": ['minkowski'],
            "weights": ['unifom'],
        }
    ],
    "NNFactory": [
        {
            "sparse_encoding__encode_proteins__kmer_size": [3],
            "sparse_encoding__encode_ecfp__radius": [6],
            "sparse_encoding__encode_ecfp__dim": [2 ** 22]
        }
    ],
    "BiLSTMFingerprintFactory": [
        {"encode_ecfp__radius": [4],
         "encode_ecfp__dim": [2 ** 20],
         "bilstm_fingerprint__module__num_fingerprints": [2 ** 20],
         "bilstm_fingerprint__optimizer": [Adam],
         "bilstm_fingerprint__lr": [1e-4],
         "bilstm_fingerprint__module__lstm_dropout": [0.2, 0.3],
         "bilstm_fingerprint__optimizer__weight_decay": [1e-3],
         "bilstm_fingerprint__max_epochs": [200]}
    ]
}
