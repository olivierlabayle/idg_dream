from torch.optim import *

GRIDS = {
    "BaselineNetFactory": [
        {
            "encode_proteins__kmer_size": [3, 5, 7],
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
            "linear_regression__alpha": [20, 30, 40, 50, 60, 70],
            "sparse_encoding__encode_proteins__kmer_size": [4],
            "sparse_encoding__encode_ecfp__radius": [6],
            "sparse_encoding__encode_ecfp__dim": [2 ** 22]
        }
    ],
    "NNFactory": [
        {
            "nn__k": [5, 10],
            "nn__weights": ['average'],
            "nn__radius": [6],
            "nn__ecfp_dim": [2 ** 22]
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
