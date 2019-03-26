import unittest
import numpy as np
import pandas as pd
from idg_dream.settings.test import DB_PORT

from idg_dream.transformers import SequenceLoader, InchiLoader, ProteinEncoder, ECFPEncoder
from idg_dream.utils import get_engine


class TestSequenceLoader(unittest.TestCase):
    engine = get_engine(db_port=DB_PORT)
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
                                    ['Q9UL54', 3, 'CCTAGTAA'],
                                    ["UNKNOWN_ID", 4, np.nan]],
                                   columns=['target_id', 'index_column', 'sequence'])
        X_transformed = X_transformed.sort_values('index_column')
        X_transformed.index = list(range(len(X_transformed)))
        pd.testing.assert_frame_equal(X_transformed, expected_df)


class TestInchiLoader(unittest.TestCase):
    engine = get_engine(db_port=DB_PORT)
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
              'InChI=1S/C17H12ClN3O3/c1-10-8-11(21-17(24)20-15(22)9-19-21)6-7-12(10)16(23)13-4-2-3-5-14(13)18/h2-9H,1H3,(H,20,22,24)'],
             ["UNKNOWN_INCHI_KEY", 3, np.nan]],
            columns=['standard_inchi_key', 'index_column', 'standard_inchi'])
        X_transformed = X_transformed.sort_values('index_column')
        X_transformed.index = list(range(len(X_transformed)))
        pd.testing.assert_frame_equal(X_transformed, expected_df)


class TestProteinEncoder(unittest.TestCase):
    transformer = ProteinEncoder(kmer_size=3)

    def test_transform(self):
        X = pd.DataFrame([["ACGTGATAGT"], ["ATCTAGATGGTCTAGTAG"]], columns=['sequence'])
        X_transformed = self.transformer.transform(X)
        pd.testing.assert_frame_equal(
            X_transformed,
            pd.DataFrame([["ACGTGATAGT", {31: 1, 10946: 1, 10821: 1}],
                          ["ATCTAGATGGTCTAGTAG", {417: 1, 10821: 3, 421: 1, 3797: 1}]],
                         columns=['sequence', 'kmers_encoding'])
        )

    def test_transform_k_equal_1(self):
        transformer = ProteinEncoder(kmer_size=1)
        X = pd.DataFrame([["ACG"], ["GGTC"]], columns=['sequence'])
        X_transformed = transformer.transform(X)
        pd.testing.assert_frame_equal(
            X_transformed,
            pd.DataFrame([["ACG", {0: 1, 1: 1, 5: 1}],
                          ["GGTC", {5: 2, 16: 1, 1: 1}]],
                         columns=['sequence', 'kmers_encoding'])
        )

    def test_transform_with_sparse_output(self):
        X = pd.DataFrame([["ACGTGATAGT"], ["ATCTAGATGGTCTAGTAG"]], columns=['sequence'])
        transformer = ProteinEncoder(kmer_size=2, sparse_output=True)
        Xt = transformer.transform(X)
        expected_nonzeros = (np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
                             np.array([1, 130, 146, 416, 5, 16, 42, 135, 146, 416, 417], dtype=np.int32))
        for i, elem in enumerate(Xt.nonzero()):
            np.testing.assert_array_equal(expected_nonzeros[i], elem)
        # Duplicates
        self.assertEqual(Xt[0, 146], 2)
        self.assertEqual(Xt[1, 16], 2)
        self.assertEqual(Xt[1, 5], 2)


class TestECFPEncoder(unittest.TestCase):
    transformer = ECFPEncoder(radius=4)

    def get_X(self):
        return pd.DataFrame([["InChI=1S/CO2/c2-1-3"],
                             ["InChI=1S/C10H10O4/c1-14-9-6-7(2-4-8(9)11)3-5-10(12)13/h2-6,11H,1H3,(H,12,13)/b5-3+"]],
                            columns=['standard_inchi'])

    def test_transform(self):
        X = self.get_X()
        X_transformed = self.transformer.transform(X)
        pd.testing.assert_frame_equal(
            X_transformed,
            pd.DataFrame([["InChI=1S/CO2/c2-1-3", [633848, 899457, 899746, 916106]],
                          ["InChI=1S/C10H10O4/c1-14-9-6-7(2-4-8(9)11)3-5-10(12)13/h2-6,11H,1H3,(H,12,13)/b5-3+",
                           [1773, 9728, 20034, 57369, 57588, 78979, 88049, 95516,
                            107971, 123721, 134214, 167638, 204359, 349540,
                            356383, 378749, 390288, 397092, 431546, 435051,
                            439248, 459409, 495384, 515018, 528633, 529834,
                            547430, 614225, 624875, 635687, 647863, 650023,
                            650051, 654006, 678945, 726962, 830972, 846213,
                            874176, 911985, 916106, 923641, 942272]]],
                         columns=['standard_inchi', 'ecfp_encoding']
                         )
        )

    def test_transform_with_sparse_output(self):
        X = self.get_X()
        transformer = ECFPEncoder(radius=4, sparse_output=True)
        Xt = transformer.transform(X)
        expected_nonzeros = (np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1], dtype=np.int32),
                              np.array([633848, 899457, 899746, 916106, 1773, 9728, 20034, 57369,
                                        57588, 78979, 88049, 95516, 107971, 123721, 134214, 167638,
                                        204359, 349540, 356383, 378749, 390288, 397092, 431546, 435051,
                                        439248, 459409, 495384, 515018, 528633, 529834, 547430, 614225,
                                        624875, 635687, 647863, 650023, 650051, 654006, 678945, 726962,
                                        830972, 846213, 874176, 911985, 916106, 923641, 942272],
                                       dtype=np.int32))
        for i, elem in enumerate(Xt.nonzero()):
            np.testing.assert_array_equal(expected_nonzeros[i], elem)
