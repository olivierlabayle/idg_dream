import unittest
import torch
import numpy as np

from idg_dream.models import Baseline


class TestBaseline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Baseline(num_kmers=5, num_fingerprints=6, embedding_dim=7)

    def test_forward(self):
        protein_input = torch.LongTensor([[1, 3, 0, 3, 4],
                                          [3, 0, 0, 2, 1]])
        compound_input = torch.LongTensor([[5, 2, 4],
                                           [4, 2, 1]])
        scores = self.model(protein_input, compound_input)
        np.testing.assert_allclose(
            scores.detach(),
            torch.tensor([[0.3311],
                          [0.3179]]),
            rtol=1e-4
        )
