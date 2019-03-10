import itertools

import pandas as pd
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from sqlalchemy import text


class SequenceLoader:
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


class InchiLoader:
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def build_query(inchi_keys):
        return text(f"""SELECT DISTINCT standard_inchi_key, standard_inchi FROM
                            compound_structures 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{inchi_keys}) AS standard_inchi_key) query_compounds
                            USING(standard_inchi_key);""")

    def transform(self, X):
        query = self.build_query(X["standard_inchi_key"].unique().tolist())
        inchis = pd.read_sql(query, self.engine)
        return X.merge(inchis, how="left", on="standard_inchi_key")


class ColumnFilter:
    def __init__(self, colnames):
        self.colnames = colnames

    def transform(self, X):
        return X[self.colnames]


class Splitter:
    def __init__(self, compound_column, protein_column):
        self.compound_column = compound_column
        self.protein_column = protein_column

    def transform(self, X):
        return X[self.compound_column], X[self.protein_column]


class ProteinEncoder:
    def __init__(self, kmer_size):
        self.kmer_size = kmer_size
        self.kmers_mapping = dict((''.join(letters), index) for index, letters in
                                  enumerate(itertools.product(ExtendedIUPACProtein.letters, repeat=self.kmer_size)))

    def _transform(self, sequence):
        n = len(sequence)
        last_amino_acid_index = n - n % self.kmer_size
        return [self.kmers_mapping[sequence[i:i + self.kmer_size]] for i in
                range(0, last_amino_acid_index, self.kmer_size)]

    def transform(self, X):
        return X.apply(self._transform)


class ECFPTransformer:
    def __init__(self, radius):
        self.radius = radius

    def transform(self, X):
        return X


if __name__ == '__main__':
    pass
