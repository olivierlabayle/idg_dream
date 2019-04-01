import unittest
import torch
from torch import optim
from torch.nn import MSELoss

from idg_dream.models import Baseline, SparseLinear, GCNLayer, SiameseLSTMFingerprint
from idg_dream.utils import inchi_to_graph


def sparse_input():
    # Let's define the following 2 lines sparse tensor [[0, 1, 0, 0],
    #                                                   [0, 0, 1, 1]]
    indexes = torch.LongTensor([[0, 1, 1],
                                [1, 2, 3]])
    values = torch.FloatTensor([1, 1, 1])
    return torch.sparse.FloatTensor(indexes, values, torch.Size([2, 4]))


class TestBaseline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Baseline(num_kmers=4, num_fingerprints=4, embedding_dim=6)

    def test_forward(self):
        protein_input = sparse_input()
        compound_input = sparse_input()
        out = self.model(protein_input, compound_input)
        self.assertTrue(torch.allclose(
            out,
            torch.FloatTensor([[-0.2538],
                               [-0.2858]]),
            rtol=1e-4
        ))

    def test_backward(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        loss_fn = MSELoss()
        target = torch.FloatTensor([[1e-5],
                                    [3.6e-4]])

        protein_input = sparse_input()
        compound_input = sparse_input()
        old_params = [p.clone().detach() for p in self.model.parameters()]

        out = self.model(protein_input, compound_input)
        loss = loss_fn(out, target)
        loss.backward()

        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

        optimizer.step()

        for i, param in enumerate(self.model.parameters()):
            self.assertFalse(torch.allclose(
                param.data,
                old_params[i]
            ))


class TestSparseLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SparseLinear(4, 6)

    def test_forward(self):
        input = sparse_input()
        output = self.model(input)

        # The first line is defined by
        first_line = self.model.weight[1, :] + self.model.bias
        # The second line is defined by
        second_line = self.model.weight[2, :] + self.model.weight[3, :] + self.model.bias
        # expected output is the concatenation of the 2 lines
        expected_output = torch.stack((first_line, second_line))

        self.assertTrue(torch.all(torch.eq(
            expected_output,
            output
        )))


class TestSiameseLSTMFingerprint(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SiameseLSTMFingerprint(num_kmers=7, num_fingerprints=4, embedding_dim=10, hidden_size=5)

    def get_inputs(self):
        protein_input = torch.tensor([[3, 4, 1, 2, 5, 0],
                                      [2, 0, 0, 6, 6, 6]])
        protein_lengths = torch.LongTensor([5, 3])
        compound_input = sparse_input()
        return protein_input, compound_input, protein_lengths

    def test_forward(self):
        out = self.model(*self.get_inputs())
        print(out)


class TestGCNLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GCNLayer(10, 4)

    def test_forward(self):
        graph = inchi_to_graph("InChI=1S/CO2/c2-1-3", max_atomic_number=10)
        out = self.model(graph)
        self.assertTrue(torch.allclose(
            out,
            torch.tensor([[-0.0395, -0.5909, -0.3447, -0.0130],
                          [0.0104, -0.0371, 0.3398, -0.4799],
                          [0.0104, -0.0371, 0.3398, -0.4799]]),
            atol=1e-4
        ))
