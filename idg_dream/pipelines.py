import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from skorch.regressor import NeuralNetRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from torch.optim import SGD

from idg_dream.models import Baseline, SiameseBiLSTMFingerprints, GraphBiLSTM
from idg_dream.transformers import InchiLoader, SequenceLoader, KmersCounter, ECFPEncoder, DfToDict, KmerEncoder, \
    InchiToDG
from functools import partial

from idg_dream.utils import collate_to_sparse_tensors, collate_bilstm_fingerprint, collate_graph_bilstm, \
    DGLHackedNeuralNetRegressor, NB_AMINO_ACID


def add_loader(cond, steps, engine):
    if cond is True:
        return [('load_inchis', InchiLoader(engine)),
                ('load_sequences', SequenceLoader(engine))] + steps
    return steps


class PipelineFactory():
    def add_loader(self, cond, steps, engine):
        if cond is True:
            return [('load_inchis', InchiLoader(engine)),
                    ('load_sequences', SequenceLoader(engine))] + steps
        return steps

    def parse_device(self, **kwargs):
        if torch.cuda.is_available() and kwargs.get('device') is not 'cpu':
            kwargs['device'] = "cuda"
        else:
            kwargs['device'] = "cpu"
        return kwargs

    def get_steps(self, **kwargs):
        raise NotImplementedError

    def __call__(self, engine=None, loaders=None, **kwargs):
        kwargs = self.parse_device(**kwargs)
        steps = self.get_steps(**kwargs)
        steps = add_loader(loaders, steps, engine)
        return Pipeline(steps=steps)


class BaselineNetFactory(PipelineFactory):
    def get_steps(self,
                  kmer_size=3,
                  radius=2,
                  ecfp_dim=2 ** 10,
                  embedding_dim=10,
                  lr=0.1,
                  max_epochs=5,
                  device=None,
                  train_split=None,
                  optimizer=SGD,
                  weight_decay=0,
                  dropout=0):
        """
        This pipeline is a neural net baseline using sparsed input fingerprints for both the compound (ecfp) and the
        enzyme (k-mers).
        :param kmer_size: The k-mer size used for the enzyme's descriptor
        :param radius: The radius used in ecfp
        :param ecfp_dim: The dimension of the byte space used by ecfp algorithm
        :param embedding_dim: Both enzyme and compounds are embedded in the neural net in a space of the same size
        :param lr: the neural net base learning rate
        :param max_epochs: Maximum number of epochs to run
        :param device: The device on which computation will take place
        :param train_split: if None, no internal cross validation is made, else a skorch CVSplit object
        :return: sklearn.pipeline.Pipeline
        """
        kmers_counter = KmersCounter(kmer_size=kmer_size)
        num_kmers = NB_AMINO_ACID ** kmer_size
        collate_fn = partial(collate_to_sparse_tensors,
                             protein_input_size=num_kmers, compound_input_size=ecfp_dim, device=torch.device(device))
        net = NeuralNetRegressor(module=Baseline,
                                 module__num_kmers=num_kmers,
                                 module__num_fingerprints=ecfp_dim,
                                 module__embedding_dim=embedding_dim,
                                 module__dropout=dropout,
                                 max_epochs=max_epochs,
                                 lr=lr,
                                 optimizer=optimizer,
                                 optimizer__weight_decay=weight_decay,
                                 device=device,
                                 iterator_train__collate_fn=collate_fn,
                                 iterator_train__shuffle=True,
                                 iterator_valid__collate_fn=collate_fn,
                                 train_split=train_split
                                 )
        return [('encode_proteins', kmers_counter),
                ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim)),
                ('to_dict', DfToDict({'protein_input': 'kmers_counts', 'compound_input': 'ecfp_encoding'})),
                ('baseline_net', net)]


class LinearRegressionFactory(PipelineFactory):
    def get_steps(self, kmer_size=3, radius=2, ecfp_dim=2 ** 10, alpha=0, device=None):
        return [
            ('sparse_encoding', FeatureUnion(n_jobs=-1, transformer_list=[
                ('encode_proteins', KmersCounter(kmer_size=kmer_size, sparse_output=True)),
                ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim, sparse_output=True))
            ])),
            ('linear_regression', Ridge(alpha=alpha))]


class NNFactory(PipelineFactory):
    def get_steps(self,
                  n_neighbors=1,
                  metric='minkowski',
                  weights='unifom',
                  kmer_size=3,
                  radius=2,
                  ecfp_dim=2 ** 10,
                   device=None):
        return [
            ('sparse_encoding', FeatureUnion(n_jobs=-1, transformer_list=[
                ('encode_proteins', KmersCounter(kmer_size=kmer_size, sparse_output=True)),
                ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim, sparse_output=True))
            ])),
            ('nn', KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric, weights=weights))]


class BiLSTMFingerprintFactory(PipelineFactory):
    def get_steps(self,
                  kmer_size=3,
                  radius=2,
                  ecfp_dim=2 ** 20,
                  hidden_size=10,
                  mlp_sizes=(10,),
                  embedding_dim=10,
                  max_epochs=10,
                  lr=1,
                  optimizer=SGD,
                  device=None,
                  train_split=None,
                  weight_decay=0,
                  lstm_dropout=0):
        """
        :param kmer_size:
        :param radius:
        :param ecfp_dim:
        :param hidden_size:
        :param mlp_sizes:
        :param embedding_dim:
        :param max_epochs:
        :param lr:
        :param optimizer:
        :param device:
        :param train_split:
        :param weight_decay:
        :param lstm_dropout:
        :return:
        """

        collate_fn = partial(collate_bilstm_fingerprint, device=torch.device(device), ecfp_dim=ecfp_dim)
        kmers_encoder = KmerEncoder(kmer_size=kmer_size, pad=True)
        net = NeuralNetRegressor(module=SiameseBiLSTMFingerprints,
                                 module__num_kmers=kmers_encoder.dim + 1,
                                 module__num_fingerprints=ecfp_dim,
                                 module__embedding_dim=embedding_dim,
                                 module__hidden_size=hidden_size,
                                 module__mlp_sizes=mlp_sizes,
                                 module__lstm_dropout=lstm_dropout,
                                 max_epochs=max_epochs,
                                 lr=lr,
                                 optimizer=optimizer,
                                 optimizer__weight_decay=weight_decay,
                                 device=device,
                                 iterator_train__shuffle=True,
                                 iterator_train__collate_fn=collate_fn,
                                 iterator_valid__collate_fn=collate_fn,
                                 train_split=train_split
                                 )
        return [
            ('encode_proteins', kmers_encoder),
            ('encode_ecfp', ECFPEncoder(radius=radius, dim=ecfp_dim, sparse_output=False)),
            ('to_dict', DfToDict(
                {'protein_input': 'kmers_encoding', 'compound_input': 'ecfp_encoding',
                 'protein_lengths': 'encoding_len'})),
            ('bilstm_fingerprint', net)]


class GraphBiLSTMFactory(PipelineFactory):
    def get_steps(self,
                  kmer_size=3,
                  graph_hidden_dim=10,
                  embedding_dim=10,
                  lstm_hidden_size=10,
                  mlp_sizes=(10,),
                  dropout=0,
                  graph_layers=1,
                  max_epochs=10,
                  lr=1e-1,
                  optimizer=SGD,
                  device=None,
                  weight_decay=1e-1,
                  train_split=None):
        kmers_encoder = KmerEncoder(kmer_size=kmer_size, pad=True)
        collate_fn = partial(collate_graph_bilstm, device=torch.device(device))
        net = DGLHackedNeuralNetRegressor(module=GraphBiLSTM,
                                          module__num_kmers=kmers_encoder.dim + 1,
                                          module__graph_in_dim=118,
                                          module__graph_hidden_dim=graph_hidden_dim,
                                          module__embedding_dim=embedding_dim,
                                          module__lstm_hidden_size=lstm_hidden_size,
                                          module__mlp_sizes=mlp_sizes,
                                          module__dropout=dropout,
                                          max_epochs=max_epochs,
                                          lr=lr,
                                          optimizer=optimizer,
                                          optimizer__weight_decay=weight_decay,
                                          device=device,
                                          iterator_train__shuffle=True,
                                          iterator_train__collate_fn=collate_fn,
                                          iterator_valid__collate_fn=collate_fn,
                                          train_split=train_split
                                          )
        return [('encode_proteins', kmers_encoder),
                ('graph_encoding', InchiToDG(device=torch.device(device))),
                ('to_dict', DfToDict(
                    {'protein_input': 'kmers_encoding', 'compound_input': 'dg_graph',
                     'protein_lengths': 'encoding_len'})),
                ('bilstm_fingerprint', net)]
