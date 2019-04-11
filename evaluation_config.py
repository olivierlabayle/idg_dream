from torch.optim import *

GRIDS = {
    "baseline_net": [
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
    "linear_regression": [
        {
            "linear_regression__alpha": [1e-2, 1e-1, 1, 10],
            "sparse_encoding__encode_proteins__kmer_size": [3, 5, 7],
            "sparse_encoding__encode_ecfp__radius": [4],
            "sparse_encoding__encode_ecfp__dim": [2 ** 20]
        }
    ],
    "bilstm_fingerprint": [
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
