import pandas as pd
import torch
from rdkit.Chem import MolFromInchi, AllChem
from sklearn.base import TransformerMixin, BaseEstimator
from sqlalchemy import text
from idg_dream.utils import update_sparse_data_from_list, update_sparse_data_from_dict, to_sparse, inchi_to_graph, \
    get_kmers_mapping


class NoFitterTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self


class SequenceLoader(NoFitterTransformer):
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def build_query(uniprot_ids):
        return text(f"""SELECT uniprot_id AS target_id, sequence 
                            FROM uniprot_proteins 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{uniprot_ids}) AS uniprot_id) query_ids
                                USING(uniprot_id);""")

    def transform(self, X):
        query = self.build_query(X['target_id'].unique().tolist())
        sequences = pd.read_sql(query, self.engine)
        return X.merge(sequences, how="left", on="target_id")


class InchiLoader(NoFitterTransformer):
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def build_query(inchi_keys):
        return text(f"""SELECT DISTINCT standard_inchi_key, standard_inchi FROM
                            compound_structures 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{inchi_keys}) AS standard_inchi_key) query_compounds
                            USING(standard_inchi_key);""")

    def transform(self, X, y=None):
        query = self.build_query(X["standard_inchi_key"].unique().tolist())
        inchis = pd.read_sql(query, self.engine)
        return X.merge(inchis, how="left", on="standard_inchi_key")


class KmersCounter(NoFitterTransformer):
    def __init__(self, kmer_size=3, sparse_output=False):
        self.kmer_size = kmer_size
        self.sparse_output = sparse_output
        self.fitted_ = False

    def fit(self, X, y=None):
        if not self.fitted_:
            self.kmers_mapping_ = get_kmers_mapping(self.kmer_size)
            self.dim_ = len(self.kmers_mapping_)
            self.fitted_ = True
        return self

    def _transform(self, sequence):
        n = len(sequence)
        last_amino_acid_index = n - n % self.kmer_size
        freqs = {}
        for i in range(0, last_amino_acid_index, self.kmer_size):
            kmer_id = self.kmers_mapping_[sequence[i:i + self.kmer_size]]
            freqs[kmer_id] = freqs.setdefault(kmer_id, 0) + 1
        return freqs

    def transform(self, X):
        Xt = X.copy()
        Xt['kmers_counts'] = Xt['sequence'].apply(self._transform)
        if self.sparse_output:
            return to_sparse(Xt['kmers_counts'], update_sparse_data_from_dict, self.dim_)
        return Xt


class KmerEncoder(NoFitterTransformer):
    def __init__(self, kmer_size=3, pad=True):
        self.kmer_size = kmer_size
        self.pad = pad
        self.init_kmers_mapping()

    def init_kmers_mapping(self):
        self.kmers_mapping = get_kmers_mapping(self.kmer_size)
        self.dim = len(self.kmers_mapping)

    def set_params(self, **params):
        super().set_params()
        self.init_kmers_mapping()

    def _transform(self, sequence, max_length):
        n = len(sequence)
        last_amino_acid_index = n - n % self.kmer_size
        kmers_ids = [self.kmers_mapping[sequence[i:i + self.kmer_size]] for i in
                     range(0, last_amino_acid_index, self.kmer_size)]
        encoding_len = len(kmers_ids)
        if max_length is not None:
            kmers_ids += [self.dim] * (max_length - encoding_len)
        return pd.Series({'kmers_encoding': kmers_ids, 'encoding_len': encoding_len})

    def transform(self, X):
        Xt = X.copy()
        max_length = None
        if self.pad:
            max_length = Xt["sequence"].str.len().max() // self.kmer_size
        Xt[['kmers_encoding', 'encoding_len']] = Xt['sequence'].apply(self._transform, args=(max_length,))
        return Xt


class ECFPEncoder(NoFitterTransformer):
    def __init__(self, radius, dim=2 ** 20, sparse_output=False):
        self.radius = radius
        self.sparse_output = sparse_output
        self.dim = dim

    def _transform(self, inchi):
        mol = MolFromInchi(inchi)
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.dim, bitInfo=info)
        return list(info.keys())

    def transform(self, X):
        Xt = X.copy()
        Xt['ecfp_encoding'] = Xt['standard_inchi'].apply(self._transform)
        if self.sparse_output:
            return to_sparse(Xt['ecfp_encoding'], update_sparse_data_from_list, self.dim)
        return Xt


class DfToDict(NoFitterTransformer):
    def __init__(self, columns_mapping):
        self.columns_mapping = columns_mapping

    def transform(self, X):
        return dict((key, X[value].values) for key, value in self.columns_mapping.items())


class InchiToDG(NoFitterTransformer):
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def transform(self, X):
        Xt = X.copy()
        Xt['dg_graph'] = Xt['standard_inchi'].apply(inchi_to_graph, device=self.device)
        return Xt
