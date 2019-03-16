from sklearn.pipeline import Pipeline
from skorch.regressor import NeuralNetRegressor
from idg_dream.models import Baseline
from idg_dream.transformers import InchiLoader, SequenceLoader, ProteinEncoder, ECFPEncoder, Splitter


def baseline(engine, kmer_size=3, radius=2, ecfp_dim=2 ** 20, embedding_dim=10, lr=0.1, max_epochs=5,
                      device='cpu'):
    protein_encoder = ProteinEncoder(kmer_size=kmer_size)
    num_kmers = len(protein_encoder.kmers_mapping)
    net = NeuralNetRegressor(module=Baseline,
                             module__num_kmers=num_kmers,
                             module__num_fingerprints=ecfp_dim,
                             module__embedding_dim=embedding_dim,
                             max_epochs=max_epochs,
                             lr=lr,
                             device=device,
                             )
    return Pipeline(
        steps=[
            ('load_inchis', InchiLoader(engine)),
            ('load_sequences', SequenceLoader(engine)),
            ('encode_proteins', protein_encoder),
            ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim)),
            ('split', Splitter(compound_column='ecfp_encoding', protein_column='kmers_encoding')),
            ('baseline_model', net)
        ]
    )


