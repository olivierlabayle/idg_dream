import pandas as pd
from sqlalchemy import text, create_engine


class SequenceLoader:
    def __init__(self, engine):
        self.engine = engine

    def build_query(self, uniprot_ids):
        return text(f"""SELECT uniprot_id, sequence 
                            FROM uniprot_proteins 
                            INNER JOIN 
                                (SELECT unnest(ARRAY{uniprot_ids}) AS uniprot_id) query_ids
                                USING(uniprot_id);""")

    def transform(self, X):
        query = self.build_query(X['target_id'].unique().tolist())
        sequences = pd.read_sql(query, self.engine)
        return X.merge(sequences, how="inner", left_on="target_id", right_on="uniprot_id")


class ColumnFilter:
    def __init__(self, colnames):
        self.colnames = colnames

    def transform(self, X):
        return X[self.colnames]


if __name__ == '__main__':
    pass
