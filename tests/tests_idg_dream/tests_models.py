import unittest

import dgl
import torch
from torch import optim
from torch.nn import MSELoss

from idg_dream.models import Baseline, SparseLinear, GCNLayer, SiameseBiLSTMFingerprints, BiLSTMProteinEmbedder, \
    GraphBiLSTM, GraphCompoundEmbedder
from idg_dream.utils import inchi_to_graph


def sparse_input():
    # Let's define the following 2 lines sparse tensor [[0, 1, 0, 0],
    #                                                   [0, 0, 1, 1]]
    indexes = torch.LongTensor([[0, 1, 1],
                                [1, 2, 3]])
    values = torch.FloatTensor([1, 1, 1])
    return torch.sparse.FloatTensor(indexes, values, torch.Size([2, 4]))


def protein_inputs():
    protein_input = torch.tensor([[3, 4, 1, 2, 5, 0],
                                  [2, 0, 0, 6, 6, 6]])
    protein_lengths = torch.LongTensor([6, 3])
    return protein_input, protein_lengths


def compound_graph_inputs():
    # Carbon dioxyde graph
    co2_graph = dgl.DGLGraph()
    co2_graph.add_nodes(3)
    # index to atom is as follows
    # 0 -> C
    # 1, 2 -> O
    for i in range(1, 3):
        co2_graph.add_edge(0, i)
        co2_graph.add_edge(i, 0)
    co2_graph.ndata['x'] = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    # Acetic acid
    acetic_acid_graph = dgl.DGLGraph()
    acetic_acid_graph.add_nodes(4)
    # index to atom is as follows
    # 0 -> central C
    # 1 -> side C
    # 2, 3 -> O
    for i in range(1, 4):
        acetic_acid_graph.add_edge(0, i)
        acetic_acid_graph.add_edge(i, 0)
    acetic_acid_graph.ndata['x'] = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    return dgl.batch([co2_graph, acetic_acid_graph])


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


class TestBiLSTMProteinEmbedder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BiLSTMProteinEmbedder(num_kmers=7, embedding_dim=10, hidden_size=5, mlp_sizes=(9,))

    def test_forward(self):
        out = self.model(*protein_inputs())
        self.assertTrue(torch.allclose(
            torch.tensor([[0.0677, 0.0000, 0.1308, 0.0000, 0.2197, 0.2629, 0.0000, 0.0000, 0.4213],
                          [0.0816, 0.0000, 0.1149, 0.0000, 0.1836, 0.3153, 0.0598, 0.0000, 0.3552]]),
            out,
            atol=1e-4
        ))

    def test_backward(self):
        model = BiLSTMProteinEmbedder(num_kmers=7, embedding_dim=10, hidden_size=5, mlp_sizes=(9, 1))
        optimizer = optim.SGD(model.parameters(), lr=1)
        loss_fn = MSELoss()
        target = torch.FloatTensor([[1e-5],
                                    [3.6e-4]])

        old_params = [p.clone().detach() for p in model.parameters()]

        out = model(*protein_inputs())
        loss = loss_fn(out, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        optimizer.step()

        for i, param in enumerate(model.parameters()):
            self.assertFalse(torch.allclose(
                param.data,
                old_params[i]
            ))


class TestGCNLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GCNLayer(10, 4)

    def test_forward(self):
        graph = inchi_to_graph("InChI=1S/CO2/c2-1-3", max_atomic_number=10)
        out = self.model(graph)
        self.assertTrue(torch.allclose(
            out.ndata['x'],
            torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
                          [0.0104, 0.0000, 0.3398, 0.0000],
                          [0.0104, 0.0000, 0.3398, 0.0000]]),
            atol=1e-4
        ))


class TestSiameseBiLSTMFingerprints(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SiameseBiLSTMFingerprints(num_kmers=7, num_fingerprints=4, embedding_dim=10, hidden_size=5,
                                               mlp_sizes=(9,))

    def test_forward(self):
        protein_input, protein_lengths = protein_inputs()
        compound_input = sparse_input()
        out = self.model(protein_input, compound_input, protein_lengths)
        self.assertTrue(torch.allclose(
            out,
            torch.tensor([[-0.0211],
                          [-0.0208]]),
            atol=1e-4
        ))

    def test_backward(self):
        optimizer = optim.SGD(self.model.parameters(), lr=1)
        loss_fn = MSELoss()
        target = torch.FloatTensor([[1e-5],
                                    [3.6e-4]])

        old_params = [p.clone().detach() for p in self.model.parameters()]

        protein_input, protein_lengths = protein_inputs()
        compound_input = sparse_input()
        out = self.model(protein_input, compound_input, protein_lengths)
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


class TestGraphCompoundEmbedder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = GraphCompoundEmbedder(in_dim=10, hidden_dim=5, mlp_sizes=(6,), nb_graph_layers=2)

    def test_forward(self):
        out = self.model(compound_graph_inputs())
        self.assertTrue(torch.allclose(
            out,
            torch.tensor([[0.2319, 0.0000, 0.1928, 0.1516, 0.0309, 0.1693],
                          [0.2048, 0.0000, 0.2241, 0.0896, 0.0127, 0.1472]]),
            atol=1e-4
        ))

    def test_backward(self):
        model = GraphCompoundEmbedder(in_dim=10, hidden_dim=5, mlp_sizes=(6, 1), nb_graph_layers=2)

        optimizer = optim.SGD(model.parameters(), lr=1)
        loss_fn = MSELoss()
        target = torch.FloatTensor([[1e-5],
                                    [3.6e-4]])

        old_params = [p.clone().detach() for p in model.parameters()]

        out = model(compound_graph_inputs())
        loss = loss_fn(out, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad)

        optimizer.step()

        for i, param in enumerate(model.parameters()):
            self.assertFalse(torch.allclose(
                param.data,
                old_params[i]
            ))


class TestGraphBiLSTM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GraphBiLSTM(graph_in_dim=10,
                                 graph_hidden_dim=5,
                                 num_kmers=7,
                                 embedding_dim=10,
                                 lstm_hidden_size=5,
                                 mlp_sizes=(9,))

    def test_forward(self):
        protein_input, protein_lengths = protein_inputs()
        out = self.model(protein_input=protein_input,
                         compound_input=compound_graph_inputs(),
                         protein_lengths=protein_lengths)
        self.assertTrue(torch.allclose(
            out,
            torch.tensor([[0.2124],
                          [0.2179]]),
            atol=1e-4
        ))

    def test_backward(self):
        optimizer = optim.SGD(self.model.parameters(), lr=1)
        loss_fn = MSELoss()
        target = torch.FloatTensor([[1e-5],
                                    [3.6e-4]])

        old_params = [p.clone().detach() for p in self.model.parameters()]

        protein_input, protein_lengths = protein_inputs()
        out = self.model(protein_input, compound_graph_inputs(), protein_lengths=protein_lengths)
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
