import itertools

import pandas as pd
import scipy.sparse
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from rdkit.Chem import MolFromInchi, AllChem
from sklearn.base import TransformerMixin
from sqlalchemy import text
from idg_dream.utils import update_sparse_data_from_list, update_sparse_data_from_dict


class SequenceLoader(TransformerMixin):
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


class InchiLoader(TransformerMixin):
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


class ColumnFilter(TransformerMixin):
    def __init__(self, colnames):
        self.colnames = colnames

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.colnames]


class Splitter(TransformerMixin):
    def __init__(self, compound_column, protein_column):
        self.compound_column = compound_column
        self.protein_column = protein_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.protein_column], X[self.compound_column]


class ProteinEncoder(TransformerMixin):
    def __init__(self, kmer_size):
        self.kmer_size = kmer_size
        self.kmers_mapping = dict((''.join(letters), index) for index, letters in
                                  enumerate(itertools.product(ExtendedIUPACProtein.letters, repeat=self.kmer_size)))

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
        X['kmers_encoding'] = X['sequence'].apply(self._transform)
        return X


class ECFPEncoder(TransformerMixin):
    def __init__(self, radius, dim=2 ** 20):
        self.radius = radius
        self.dim = dim

    def _transform(self, inchi):
        mol = MolFromInchi(inchi)
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.dim, bitInfo=info)
        return list(info.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['ecfp_encoding'] = X['standard_inchi'].apply(self._transform)
        return X


class SparseJoin(TransformerMixin):
    def __init__(self, protein_colname, compound_colname, protein_dim, compound_dim):
        self.protein_colname = protein_colname
        self.compound_colname = compound_colname
        self.protein_dim = protein_dim
        self.compound_dim = compound_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p_data, p_row_indexes, p_col_indexes = [], [], []
        c_data, c_row_indexes, c_col_indexes = [], [], []
        n = len(X)
        for index, row in enumerate(X[[self.compound_colname, self.protein_colname]].itertuples()):
            update_sparse_data_from_dict(p_data, p_row_indexes, p_col_indexes, row.kmers_encoding, index)
            update_sparse_data_from_list(c_data, c_row_indexes, c_col_indexes, row.ecfp_encoding, index)

        protein_sparse = scipy.sparse.csr_matrix((p_data, (p_row_indexes, p_col_indexes)), shape=(n, self.protein_dim))
        compound_sparse = scipy.sparse.csr_matrix((c_data, (c_row_indexes, c_col_indexes)), shape=(n, self.compound_dim))
        return scipy.sparse.hstack((compound_sparse, protein_sparse))


class DfToDict(TransformerMixin):
    def __init__(self, protein_colname, compound_colname):
        self.protein_colname = protein_colname
        self.compound_colname = compound_colname

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return {'protein_input': X[self.protein_colname].values, 'compound_input': X[self.compound_colname].values}
