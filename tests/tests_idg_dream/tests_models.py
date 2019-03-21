import unittest
import torch
import numpy as np

from idg_dream.models import Baseline, SparseLinear


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


class TestSparseLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SparseLinear(4, 6)

    def test_forward(self):
        # Let's define the following 2 lines sparse tensor [[0, 1, 0, 0],
        #                                                   [0, 0, 1, 1]]
        indexes = torch.LongTensor([[0, 1, 1],
                                    [1, 2, 3]])
        values = torch.FloatTensor([1, 1, 1])
        input = torch.sparse.FloatTensor(indexes, values, torch.Size([2, 4]))
        output = self.model(input)
        # The first line is defined by
        first_line = self.model.weight.t()[1, :] + self.model.bias
        # The second line is defined by
        second_line = self.model.weight.t()[2, :] + self.model.weight.t()[3, :] + self.model.bias
        # expected output is the concatenation of the 2 lines
        expected_output = torch.stack((first_line, second_line))
        self.assertTrue(torch.all(torch.eq(
            expected_output,
            output
        )))
