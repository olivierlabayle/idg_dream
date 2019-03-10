import unittest
import pandas as pd
from sqlalchemy import create_engine
from idg_dream.settings.test import DB_PORT

from idg_dream.transforms import SequenceLoader, ColumnFilter, InchiLoader


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

    @staticmethod
    def get_X():
        return pd.DataFrame([['Q9Y4K4', 0],
                             ['Q9Y478', 1],
                             ['Q9Y4K4', 2],
                             ['Q9UL54', 3],
                             ["UNKNOWN_ID", 4]],
                            columns=["target_id", "index_column"])

    def test_transform(self):
        X_transformed = self.transformer.transform(self.get_X())
        expected_df = pd.DataFrame([['Q9Y4K4', 0, 'ACGTG'],
                                    ['Q9Y478', 1, 'GTGTGG'],
                                    ['Q9Y4K4', 2, 'ACGTG'],
                                    ['Q9UL54', 3, 'CCTAGTAA']],
                                   columns=['target_id', 'index_column', 'sequence'])
        X_transformed = X_transformed.sort_values('index_column')
        X_transformed.index = list(range(len(X_transformed)))
        pd.testing.assert_frame_equal(X_transformed, expected_df)


class TestInchiLoader(unittest.TestCase):
    engine = create_engine(f'postgresql+pg8000://idg_dream@127.0.0.1:{DB_PORT}/idg_dream', echo=False)
    transformer = InchiLoader(engine)

    @classmethod
    def setUpClass(cls):
        uniprot_data = pd.DataFrame([['OWRSAHYFSSNENM-UHFFFAOYSA-N',
                                      'InChI=1S/C17H12ClN3O3/c1-10-8-11(21-17(24)20-15(22)9-19-21)6-7-12(10)16(23)13-4-2-3-5-14(13)18/h2-9H,1H3,(H,20,22,24)'],
                                     ['ZJYUMURGSZQFMH-UHFFFAOYSA-N',
                                      'InChI=1S/C18H12N4O3/c1-11-8-14(22-18(25)21-16(23)10-20-22)6-7-15(11)17(24)13-4-2-12(9-19)3-5-13/h2-8,10H,1H3,(H,21,23,25)'],
                                     ['YOMWDCALSDWFSV-UHFFFAOYSA-N',
                                      'InChI=1S/C18H16ClN3O3/c1-10-7-14(22-18(25)21-15(23)9-20-22)8-11(2)16(10)17(24)12-3-5-13(19)6-4-12/h3-9,17,24H,1-2H3,(H,21,23,25)']],
                                    columns=["standard_inchi_key", 'standard_inchi'])
        uniprot_data.to_sql('compound_structures', cls.engine, if_exists='replace')

    @classmethod
    def tearDownClass(cls):
        cls.engine.execute("DROP TABLE IF EXISTS compound_structures;")

    @staticmethod
    def get_X():
        return pd.DataFrame([['OWRSAHYFSSNENM-UHFFFAOYSA-N', 0],
                             ['YOMWDCALSDWFSV-UHFFFAOYSA-N', 1],
                             ['OWRSAHYFSSNENM-UHFFFAOYSA-N', 2],
                             ["UNKNOWN_INCHI_KEY", 3]],
                            columns=["standard_inchi_key", "index_column"])

    def test_transform(self):
        X_transformed = self.transformer.transform(self.get_X())
        expected_df = pd.DataFrame(
            [['OWRSAHYFSSNENM-UHFFFAOYSA-N', 0,
              'InChI=1S/C17H12ClN3O3/c1-10-8-11(21-17(24)20-15(22)9-19-21)6-7-12(10)16(23)13-4-2-3-5-14(13)18/h2-9H,1H3,(H,20,22,24)'],
             ['YOMWDCALSDWFSV-UHFFFAOYSA-N', 1,
              'InChI=1S/C18H16ClN3O3/c1-10-7-14(22-18(25)21-15(23)9-20-22)8-11(2)16(10)17(24)12-3-5-13(19)6-4-12/h3-9,17,24H,1-2H3,(H,21,23,25)'],
             ['OWRSAHYFSSNENM-UHFFFAOYSA-N', 2,
              'InChI=1S/C17H12ClN3O3/c1-10-8-11(21-17(24)20-15(22)9-19-21)6-7-12(10)16(23)13-4-2-3-5-14(13)18/h2-9H,1H3,(H,20,22,24)']],
            columns=['standard_inchi_key', 'index_column', 'standard_inchi'])
        X_transformed = X_transformed.sort_values('index_column')
        X_transformed.index = list(range(len(X_transformed)))
        pd.testing.assert_frame_equal(X_transformed, expected_df)


class TestColumnFilter(unittest.TestCase):
    transformer = ColumnFilter(colnames=['sequence', 'compound_id'])

    @staticmethod
    def get_X():
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
