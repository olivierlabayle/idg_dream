import torch
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from skorch.regressor import NeuralNetRegressor
from sklearn.linear_model import Ridge
from idg_dream.models import Baseline, SiameseBiLSTMFingerprints
from idg_dream.transformers import InchiLoader, SequenceLoader, KmersCounter, ECFPEncoder, DfToDict, KmerEncoder
from functools import partial

from idg_dream.utils import collate_to_sparse_tensors, update_sparse_data_from_list, to_sparse


def add_loader(cond, steps, engine):
    if cond is True:
        return [('load_inchis', InchiLoader(engine)),
                ('load_sequences', SequenceLoader(engine))] + steps
    return steps


def baseline_net(engine=None, kmer_size=3, radius=2, ecfp_dim=2 ** 10, embedding_dim=10, lr=0.1, max_epochs=5,
                 device=None, loaders=False, train_split=None):
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
    if torch.cuda.is_available() and device is not 'cpu':
        device = "cuda"
    else:
        device = "cpu"

    protein_encoder = KmersCounter(kmer_size=kmer_size)
    num_kmers = len(protein_encoder.kmers_mapping)
    collate_fn = partial(collate_to_sparse_tensors,
                         protein_input_size=num_kmers, compound_input_size=ecfp_dim, device=torch.device(device))
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
             ('to_dict', DfToDict({'protein_input': 'kmers_counts', 'compound_input': 'ecfp_encoding'})),
             ('baseline_net', net)]
    steps = add_loader(loaders, steps, engine)
    return Pipeline(steps=steps)


def linear_regression(engine=None, loaders=False, kmer_size=3, radius=2, ecfp_dim=2 ** 10, alpha=0):
    steps = [
        ('sparse_encoding', FeatureUnion(n_jobs=-1, transformer_list=[
            ('encode_proteins', KmersCounter(kmer_size=kmer_size, sparse_output=True)),
            ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim, sparse_output=True))
        ])),
        ('linear_regression', Ridge(alpha=alpha))]
    steps = add_loader(loaders, steps, engine)
    return Pipeline(steps=steps)


def bilstm_fingerprint(engine=None, loaders=False, kmer_size=3, radius=2, ecfp_dim=2 ** 10, hidden_size=10,
                       mlp_sizes=(10,), embedding_dim=10, max_epochs=10, lr=1, device=None, train_split=None):
    if torch.cuda.is_available() and device is not 'cpu':
        device = "cuda"
    else:
        device = "cpu"

    def collate_fn(batch):
        """
        need to sort by length
        :return:
        """
        protein_inputs, protein_lengths, compound_inputs, targets = [], [], [], []
        for x, y in batch:
            protein_inputs.append(x['protein_input'])
            protein_lengths.append(x['protein_lengths'])
            compound_inputs.append(x['compound_input'])
            targets.append(y)
        order = np.argsort(protein_lengths)[::-1]
        protein_lengths = np.array(protein_lengths)[order]
        protein_inputs = np.array(protein_inputs)[order]
        compound_inputs = np.array(compound_inputs)[order]
        targets = np.array(targets)[order]
        return (
            {
                "compound_input": to_sparse(compound_inputs, update_sparse_data_from_list, ecfp_dim,
                                            return_torch=True).to(device),
                "protein_input": torch.from_numpy(protein_inputs).to(device),
                "protein_lengths": torch.from_numpy(protein_lengths).to(device)
            },
            torch.from_numpy(targets).to(device)
        )

    kmers_encoder = KmerEncoder(kmer_size=kmer_size, pad=True)
    net = NeuralNetRegressor(module=SiameseBiLSTMFingerprints,
                             module__num_kmers=kmers_encoder.dim,
                             module__num_fingerprints=ecfp_dim,
                             module__embedding_dim=embedding_dim,
                             module__hidden_size=hidden_size,
                             module__mlp_sizes=mlp_sizes,
                             max_epochs=max_epochs,
                             lr=lr,
                             device=device,
                             iterator_train__collate_fn=collate_fn,
                             iterator_valid__collate_fn=collate_fn,
                             train_split=train_split
                             )
    steps = [
        ('encode_proteins', kmers_encoder),
        ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim, sparse_output=False)),
        ('to_dict', DfToDict(
            {'protein_input': 'kmers_encoding', 'compound_input': 'ecfp_encoding', 'protein_lengths': 'encoding_len'})),
        ('bilstm_fingerprint', net)]
    steps = add_loader(loaders, steps, engine)

    return Pipeline(steps=steps)
