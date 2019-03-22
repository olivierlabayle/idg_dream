from sklearn.pipeline import Pipeline
from skorch.regressor import NeuralNetRegressor
from idg_dream.models import Baseline
from idg_dream.transformers import InchiLoader, SequenceLoader, ProteinEncoder, ECFPEncoder, DfToDict
from functools import partial

from idg_dream.utils import collate_to_sparse_tensors


def baseline(engine=None, kmer_size=3, radius=2, ecfp_dim=2 ** 10, embedding_dim=10, lr=0.1, max_epochs=5,
             device='cpu', loaders=False, train_split=None):
    """
    This pipeline is a neural net baseline using sparsed input fingerprints for both the compound (ecfp) and the
    enzyme (k-mers).
    :param engine: If loaders is true, the you should also specify an engine
    :param kmer_size: The k-mer size used for the enzyme's descriptor
    :param radius: The radius used in ecfp
    :param ecfp_dim: The dimension of the byte space used by ecfp algorithm
    :param embedding_dim: Both enzyme and compounds are embedded in the neural net in a space of the same size
    :param lr: the neural net base learning rate
    :param max_epochs: Maximum number of epochs to run
    :param device: The device on which computation will take place
    :param loaders: If the sequence of the enzyme and the inchi of the compound is not present in the base dataset,
                    use the idg_dream database to load them, the engine argument should then be specified
    :param train_split: if None, no internal cross validation is made, else a skorch CVSplit object
    :return: sklearn.pipeline.Pipeline
    """
    protein_encoder = ProteinEncoder(kmer_size=kmer_size)
    num_kmers = len(protein_encoder.kmers_mapping)
    collate_fn = partial(collate_to_sparse_tensors, protein_input_size=num_kmers, compound_input_size=ecfp_dim)
    net = NeuralNetRegressor(module=Baseline,
                             module__num_kmers=num_kmers,
                             module__num_fingerprints=ecfp_dim,
                             module__embedding_dim=embedding_dim,
                             max_epochs=max_epochs,
                             lr=lr,
                             device=device,
                             iterator_train__collate_fn=collate_fn,
                             iterator_valid__collate_fn=collate_fn,
                             train_split=train_split
                             )
    steps = [('encode_proteins', protein_encoder),
             ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim)),
             ('to_dict', DfToDict(protein_colname='kmers_encoding', compound_colname='ecfp_encoding', )),
             ('baseline_model', net)]
    if loaders:
        steps = [('load_inchis', InchiLoader(engine)),
                 ('load_sequences', SequenceLoader(engine))] + steps
    return Pipeline(
        steps=steps
    )
