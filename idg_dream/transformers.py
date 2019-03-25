import itertools

import pandas as pd
import scipy.sparse
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from rdkit.Chem import MolFromInchi, AllChem
from sklearn.base import TransformerMixin, BaseEstimator
from sqlalchemy import text
from idg_dream.utils import update_sparse_data_from_list, update_sparse_data_from_dict, to_sparse


class SequenceLoader(TransformerMixin, BaseEstimator):
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def build_query(uniprot_ids):
        return text(f"""SELECT uniprot_id AS target_id, sequence 
                            FROM uniprot_proteins 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{uniprot_ids}) AS uniprot_id) query_ids
                                USING(uniprot_id);""")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        query = self.build_query(X['target_id'].unique().tolist())
        sequences = pd.read_sql(query, self.engine)
        return X.merge(sequences, how="left", on="target_id")


class InchiLoader(TransformerMixin, BaseEstimator):
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def build_query(inchi_keys):
        return text(f"""SELECT DISTINCT standard_inchi_key, standard_inchi FROM
                            compound_structures 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{inchi_keys}) AS standard_inchi_key) query_compounds
                            USING(standard_inchi_key);""")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        query = self.build_query(X["standard_inchi_key"].unique().tolist())
        inchis = pd.read_sql(query, self.engine)
        return X.merge(inchis, how="left", on="standard_inchi_key")


class ProteinEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, kmer_size, sparse_output=False):
        self.kmer_size = kmer_size
        self.sparse_output = sparse_output
        self.init_kmers_mapping()

    def init_kmers_mapping(self):
        self.kmers_mapping = dict((''.join(letters), index) for index, letters in
                                  enumerate(itertools.product(ExtendedIUPACProtein.letters, repeat=self.kmer_size)))
        self.dim = len(self.kmers_mapping)

    def set_params(self, **params):
        super().set_params()
        self.init_kmers_mapping()

    def _transform(self, sequence):
        n = len(sequence)
        last_amino_acid_index = n - n % self.kmer_size
        freqs = {}
        for i in range(0, last_amino_acid_index, self.kmer_size):
            kmer_id = self.kmers_mapping[sequence[i:i + self.kmer_size]]
            freqs[kmer_id] = freqs.setdefault(kmer_id, 0) + 1
        return freqs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xt = X.copy()
        Xt['kmers_encoding'] = Xt['sequence'].apply(self._transform)
        if self.sparse_output:
            return to_sparse(Xt['kmers_encoding'], update_sparse_data_from_dict, self.dim)
        return Xt


class ECFPEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, radius, dim=2 ** 20, sparse_output=False):
        self.radius = radius
        self.sparse_output = sparse_output
        self.dim = dim

    def _transform(self, inchi):
        mol = MolFromInchi(inchi)
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.dim, bitInfo=info)
        return list(info.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xt = X.copy()
        Xt['ecfp_encoding'] = Xt['standard_inchi'].apply(self._transform)
        if self.sparse_output:
            return to_sparse(Xt['ecfp_encoding'], update_sparse_data_from_list, self.dim)
        return Xt


class DfToDict(TransformerMixin, BaseEstimator):
    def __init__(self, protein_colname, compound_colname):
        self.protein_colname = protein_colname
        self.compound_colname = compound_colname

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return {'protein_input': X[self.protein_colname].values, 'compound_input': X[self.compound_colname].values}
