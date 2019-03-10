import pandas as pd
from sqlalchemy import text, create_engine


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


if __name__ == '__main__':
    pass
