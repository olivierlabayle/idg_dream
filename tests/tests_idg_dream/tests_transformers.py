import unittest
import pandas as pd
from sqlalchemy import create_engine
from idg_dream.settings.test import DB_PORT

from idg_dream.transforms import SequenceLoader, ColumnFilter


class TestSequenceLoader(unittest.TestCase):
    engine = create_engine(f'postgresql+pg8000://idg_dream@127.0.0.1:{DB_PORT}/idg_dream', echo=False)
    transformer = SequenceLoader(engine=engine)

    @classmethod
    def setUpClass(cls):
        uniprot_data = pd.DataFrame([['Q9Y4K4', 'ACGTG'],
                                     ['Q9Y478', 'GTGTGG'],
                                     ['Q9UL54', 'CCTAGTAA']], columns=["uniprot_id", 'sequence'])
        uniprot_data.to_sql('uniprot_proteins', cls.engine, if_exists='replace')

    @classmethod
    def tearDownClass(cls):
        cls.engine.execute("DROP TABLE IF EXISTS uniprot_proteins;")

    def get_X(self):
        return pd.DataFrame([['Q9Y4K4', 0],
                             ['Q9Y478', 1],
                             ['Q9Y4K4', 2],
                             ['Q9UL54', 3],
                             ["UNKNOWN_ID", 4]],
                            columns=["target_id", "index_column"])

    def test_transform(self):
        X_transformed = self.transformer.transform(self.get_X())
        expected_df = pd.DataFrame([['Q9Y4K4', 0, 'Q9Y4K4', 'ACGTG'],
                                    ['Q9Y478', 1, 'Q9Y478', 'GTGTGG'],
                                    ['Q9Y4K4', 2, 'Q9Y4K4', 'ACGTG'],
                                    ['Q9UL54', 3, 'Q9UL54', 'CCTAGTAA']],
                                   columns=['target_id', 'index_column', 'uniprot_id', 'sequence'])
        X_transformed = X_transformed.sort_values('index_column')
        X_transformed.index = list(range(len(X_transformed)))
        pd.testing.assert_frame_equal(X_transformed, expected_df)


class TestColumnFilter(unittest.TestCase):
    transformer = ColumnFilter(colnames=['sequence', 'compound_id'])

    def get_X(self):
        return pd.DataFrame([["ACGTG", 0, "Q9Y4K4"],
                             ["GGTGACG", 1, "XRFTG4"]],
                            columns=["sequence", "compound_id", "uniprot_id"])

    def test_transform(self):
        X_transformed = self.transformer.transform(self.get_X())
        pd.testing.assert_frame_equal(
            X_transformed,
            pd.DataFrame([['ACGTG', 0],
                          ['GGTGACG', 1]],
                         columns=["sequence", "compound_id"])
        )
