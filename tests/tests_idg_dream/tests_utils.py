import unittest
import numpy as np
import torch
from idg_dream.utils import collate_to_sparse_tensors


class TestUtils(unittest.TestCase):
    def test_collate_to_sparse_tensor(self):
        batch = [({"compound_input": np.array([5, 2, 1]), "protein_input": {3: 2, 0: 1}}, np.array([1])),
                 ({"compound_input": np.array([0, 3]), "protein_input": {1: 1, 5: 5}}, np.array([0]))]
        sparse_matrices, target = collate_to_sparse_tensors(batch, protein_input_size=6, compound_input_size=7)
        self.assertTrue(torch.all(torch.eq(
            sparse_matrices['protein_input'].to_dense(),
            torch.FloatTensor([[1, 0, 0, 2, 0, 0],
                               [0, 1, 0, 0, 0, 5]])
        )))
        self.assertTrue(torch.all(torch.eq(
            sparse_matrices['compound_input'].to_dense(),
            torch.FloatTensor([[0, 1, 1, 0, 0, 1, 0],
                               [1, 0, 0, 1, 0, 0, 0]])
        )))
        self.assertTrue(torch.all(torch.eq(
            target,
            torch.FloatTensor([[1],
                              [0]])
        )))
