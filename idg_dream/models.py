import torch
import math
import torch.nn as nn


### Baseline neural net model ###


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.sparse.mm(input, self.weight) + self.bias


class Baseline(nn.Module):
    def __init__(self, num_kmers, num_fingerprints, embedding_dim=10):
        """
        This model takes fix-sized inputs, it is used as a baseline.
        :param num_kmers:
        :param num_fingerprints:
        :param embedding_dim:
        """
        super().__init__()
        self.num_kmers = num_kmers
        self.num_fingerprints = num_fingerprints
        self.embedding_dim = embedding_dim
        self.protein_branch = nn.Sequential(SparseLinear(num_kmers, self.embedding_dim), nn.ReLU())
        self.compound_branch = nn.Sequential(SparseLinear(num_fingerprints, self.embedding_dim), nn.ReLU())
        self.output_branch = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

    def forward(self, protein_input, compound_input):
        protein_embedding = self.protein_branch(protein_input)
        compound_embedding = self.compound_branch(compound_input)
        joined = torch.cat((protein_embedding, compound_embedding), dim=1)
        return self.output_branch(joined)


class SiameseLSTMFingerprint(nn.Module):
    def __init__(self, num_kmers, num_fingerprints, embedding_dim, hidden_size):
        super().__init__()
        self.num_kmers = num_kmers
        self.num_fingerprints = num_fingerprints
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        # Protein branch layers
        self.protein_embedding = nn.Embedding(num_kmers, embedding_dim)
        self.lstm_states = (None, None)
        self.protein_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        # For now, the mlp will only have one layer
        self.protein_mlp = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim), nn.ReLU())
        # Compound branch layers
        self.compound_branch = nn.Sequential(SparseLinear(num_fingerprints, self.embedding_dim), nn.ReLU())
        # Out layers
        self.output_branch = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

    def init_lstm_state(self, batch_size):
        return torch.randn(2, batch_size, self.hidden_size), torch.randn(2, batch_size, self.hidden_size)

    def protein_branch(self, protein_input, protein_lengths):
        # First let's embed the input
        protein_embedding = self.protein_embedding(protein_input)
        # In order to use mini-batch computations we will use pytorch pack_apdded_sequence
        packed_proteins = nn.utils.rnn.pack_padded_sequence(protein_embedding, protein_lengths, batch_first=True)
        lstm_state = self.init_lstm_state(batch_size=protein_lengths.shape[0])
        # Let's go through LSTM layer
        proteins_features, lstm_state = self.protein_lstm(packed_proteins, lstm_state)
        # Undo the packing
        proteins_features, _ = nn.utils.rnn.pad_packed_sequence(proteins_features, batch_first=True)
        # The hidden state is then fed into a MLP before joining with the compound's features
        return self.protein_mlp(proteins_features[:, -1, :])

    def forward(self, protein_input, compound_input, protein_lengths):
        compound_embedding = self.compound_branch(compound_input)
        protein_embedding = self.protein_branch(protein_input, protein_lengths)
        joined = torch.cat((protein_embedding, compound_embedding), dim=1)
        return self.output_branch(joined)


### Graph neural net model ###
# copied from DGL

def gcn_message(edges):
    """
    This computes a batch of messages called 'msg' using the source node's feature stored in 'x' key
    :param edges: A batch of edges
    :return:
    """
    return {'msg': edges.src['x']}


def gcn_reduce(nodes):
    """
    This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    :param nodes: A batch of nodes
    :return:
    """
    return {'x': torch.sum(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g):
        """
        :param g: a dgl.DGLGraph with associated features
        :return:
        """
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('x')
        # perform linear transformation
        return self.linear(h)
