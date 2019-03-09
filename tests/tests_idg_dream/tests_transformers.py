import unittest
import pandas as pd
from sqlalchemy import create_engine
from idg_dream.settings import DB_PORT

from idg_dream.transforms import SequenceLoader


class TestSequenceLoader(unittest.TestCase):
    transformer = SequenceLoader(
        engine=create_engine(f'postgresql+pg8000://idg_dream@127.0.0.1:{DB_PORT}/idg_dream', echo=False))

    def get_X(self):
        return pd.DataFrame([['Q9Y4K4', 0],
                             ['Q9Y478', 1],
                             ['Q9Y4K4', 2],
                             ['Q9UL54', 3],
                             ["UNKNOWN_ID", 4]],
                            columns=["target_id", "other_column"])

    def test_transform(self):
        X_transformed = self.transformer.transform(self.get_X())
        pd.testing.assert_frame_equal(X_transformed, [])
