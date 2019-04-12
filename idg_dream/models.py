import dgl
import torch
import math
import torch.nn as nn


### Layers used by subsequent models


def get_mlp_from_sizes(sizes, activation_last=True):
    """
    Creates sequential layers corresponding to a simple Multi Layer Perceptron
    :param sizes ([int]): The sequence of sizes of the layers in the MLP branch
    :param activation_last (bool): Should the last layer be followed by a non linear activation
    :return: nn.Sequential
    """
    # nn.Module needs to be instantiated before being passed to Sequential,
    # that's why a list is used rather than a generator
    branch = []
    for input_size, out_put_size in zip(sizes, sizes[1:]):
        branch.append(nn.Linear(input_size, out_put_size))
        branch.append(nn.ReLU())
    if branch and not activation_last:
        branch.pop(-1)
    return nn.Sequential(*branch)


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        A sparse linear layer that accepts backward
        :param in_features (int): Input feature size
        :param out_features (int): Output feature size
        """
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


class BiLSTMProteinEmbedder(nn.Module):
    def __init__(self, num_kmers, embedding_dim, hidden_size, mlp_sizes, dropout=0):
        """
        This is a basic architecture that consists in 3 sequential layers :
            1) An Embedding layer is applied to encode kmers
            2) A Bidirectionnal LSTM is applied
            3) A final MLP provides the final proteins embeddings
        :param num_kmers: The total number of kmers (used by the embedding layer)
        :param embedding_dim: The Embedding dimension (used by the embedding layer)
        :param hidden_size: The size of the LSTM hidden unit
        :param mlp_sizes: A tuple of sizes to use for the final sequence of Linear/ReLU units
        """
        super().__init__()
        self.num_kmers = num_kmers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mlp_sizes = [2 * self.hidden_size] + list(mlp_sizes)

        self.protein_embedding = nn.Embedding(num_kmers, embedding_dim)
        self.protein_lstm = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=dropout)
        self.protein_mlp = get_mlp_from_sizes(self.mlp_sizes)

    def forward(self, proteins_inputs, proteins_lengths):
        """
        Applies the forward pass of the class BiLSTMProteinEmbedder.
        Protein sequences are assumed to have been padded already.
        :param proteins_inputs: torch.tensor of shape [batch_size, sequence_len]
        :param proteins_lengths: torch.tensor of shape [batch_size]
        :return: torch.tensor of shape [batch_size, mlp_sizes[-1]]
        """
        batch_size = proteins_lengths.shape[0]
        sequence_len = proteins_lengths[0].item()
        # First let's embed the input
        protein_embedding = self.protein_embedding(proteins_inputs)
        # In order to use mini-batch computations we will use pytorch pack_apdded_sequence
        packed_proteins = nn.utils.rnn.pack_padded_sequence(protein_embedding, proteins_lengths, batch_first=True)
        # Let's go through LSTM layer
        packed_proteins_features, lstm_state = self.protein_lstm(packed_proteins)
        # Undo the packing
        proteins_features, _ = nn.utils.rnn.pad_packed_sequence(packed_proteins_features, batch_first=True)
        # Rearrange lstm output
        proteins_features = proteins_features.view(batch_size, sequence_len, 2, self.hidden_size)
        # The final features are the backward activation of the first unit and
        # the forward activation of the last unit
        proteins_features = torch.cat((proteins_features[:, 0, 1, :], proteins_features[:, -1, 0, :]), dim=1)
        # The hidden state is then fed into a MLP before joining with the compound's features
        return self.protein_mlp(proteins_features)


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
        self.relu = nn.ReLU()

    def forward(self, g):
        """
        :param g: a dgl.DGLGraph with associated features
        :return:
        """
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # perform linear transformation
        g.ndata['x'] = self.linear(g.ndata['x'])
        # Returns the graph
        g.ndata['x'] = self.relu(g.ndata['x'])
        return g


class GraphCompoundEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, mlp_sizes, nb_graph_layers=1):
        super().__init__()
        assert nb_graph_layers >= 1
        layers = [GCNLayer(in_dim, hidden_dim)]
        for i in range(1, nb_graph_layers):
            layers.extend([GCNLayer(hidden_dim, hidden_dim)])
        self.graph_layers = nn.Sequential(*layers)
        self.mlp = get_mlp_from_sizes([hidden_dim] + list(mlp_sizes), activation_last=True)

    def forward(self, graph):
        """
        Applies multiple GCN Layers followed by relu units
        :param graph: DGLGraph
        :return: torch.Tensor
        """
        # Goes through graph units
        graph = self.graph_layers(graph)
        # Graph readout
        hg = dgl.mean_nodes(graph, 'x')
        # mlp final layers
        return self.mlp(hg)


### Models ###


class SiameseNetwork(nn.Module):
    def protein_branch(self, protein_input, **protein_kwargs):
        raise NotImplementedError

    def compound_branch(self, compound_input):
        raise NotImplementedError

    def join(self, protein_features, compound_features):
        raise NotImplementedError

    def output_branch(self, joined):
        raise NotImplementedError

    def forward(self, protein_input, compound_input, **protein_kwargs):
        protein_features = self.protein_branch(protein_input, **protein_kwargs)
        compound_features = self.compound_branch(compound_input)
        joined = self.join(protein_features, compound_features)
        out = self.output_branch(joined)
        return out


class Baseline(SiameseNetwork):
    def __init__(self, num_kmers, num_fingerprints, embedding_dim=10, dropout=0):
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
        self.dropout=dropout
        self._protein_branch = nn.Sequential(SparseLinear(num_kmers, self.embedding_dim),
                                            nn.ReLU(),
                                            nn.Dropout(dropout))
        self._compound_branch = nn.Sequential(SparseLinear(num_fingerprints, self.embedding_dim),
                                             nn.ReLU(),
                                             nn.Dropout(dropout))
        self.bilinear = nn.Bilinear(self.embedding_dim, self.embedding_dim, self.embedding_dim)
        self._output_branch = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

    def protein_branch(self, protein_input, **kwargs):
        return self._protein_branch(protein_input)

    def compound_branch(self, compound_input):
        return self._compound_branch(compound_input)

    def output_branch(self, joined):
        return self._output_branch(joined)

    def join(self, protein_features, compound_features):
        return self.bilinear(protein_features, compound_features)


class SiameseBiLSTMFingerprints(nn.Module):
    def __init__(self, num_kmers, num_fingerprints, embedding_dim, hidden_size, mlp_sizes, lstm_dropout=0):
        """
        This model is a siamese model that uses :
            1) The fingerprint and a mlp to represent the compound
            2) A BiLSTMProteinEmbedder to represent the protein
            3) Joins the outputs of those two branches and passes them to a mlp to output the pKd
        :param num_kmers: The number of kmers used by the protein branch
        :param num_fingerprints: The fingerprint space used by the first embedding layer of the compound branch
        :param embedding_dim: The initial embedding dimension used both by the protein and compound branch
        :param hidden_size: The hidden unit size of the lstm in the protein branch
        :param mlp_sizes: The MLP sizes used by all : compound branch, protein branch, output branch
        """
        super().__init__()
        self.num_kmers = num_kmers
        self.num_fingerprints = num_fingerprints
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.mlp_sizes = mlp_sizes
        self.lstm_dropout = lstm_dropout
        # Protein branch layers
        self.protein_branch = BiLSTMProteinEmbedder(num_kmers,
                                                    embedding_dim,
                                                    hidden_size,
                                                    mlp_sizes,
                                                    dropout=lstm_dropout)
        # Compound branch layers
        self.compound_branch = nn.Sequential(
            SparseLinear(num_fingerprints, self.embedding_dim),
            nn.ReLU(),
            get_mlp_from_sizes([self.embedding_dim] + list(mlp_sizes))
        )
        # Out layers
        self.output_branch = get_mlp_from_sizes(
            [2 * self.mlp_sizes[-1]] + list(mlp_sizes) + [1],
            activation_last=False
        )

    def forward(self, protein_input, compound_input, protein_lengths):
        compound_embedding = self.compound_branch(compound_input)
        protein_embedding = self.protein_branch(protein_input, protein_lengths)
        joined = torch.cat((protein_embedding, compound_embedding), dim=1)
        out = self.output_branch(joined)
        return out


class GraphBiLSTM(SiameseNetwork):
    def __init__(self,
                 graph_in_dim,
                 graph_hidden_dim,
                 num_kmers,
                 embedding_dim,
                 lstm_hidden_size,
                 mlp_sizes,
                 dropout=0,
                 graph_layers=1):
        super().__init__()
        self.graph_in_dim = graph_in_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.num_kmers = num_kmers
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_sizes = mlp_sizes
        self.dropout = dropout
        self._compound_branch = GraphCompoundEmbedder(
            graph_in_dim,
            graph_hidden_dim,
            mlp_sizes,
            nb_graph_layers=graph_layers
        )
        self._protein_branch = BiLSTMProteinEmbedder(
            num_kmers,
            embedding_dim,
            lstm_hidden_size,
            mlp_sizes,
            dropout=dropout
        )
        self._output_branch = get_mlp_from_sizes(
            [2 * self.mlp_sizes[-1]] + list(mlp_sizes) + [1],
            activation_last=False
        )

    def compound_branch(self, compound_input):
        return self._compound_branch(compound_input)

    def protein_branch(self, protein_input, protein_lengths=None):
        return self._protein_branch(protein_input, protein_lengths)

    def join(self, protein_features, compound_features):
        return torch.cat((protein_features, compound_features), dim=1)

    def output_branch(self, joined):
        return self._output_branch(joined)
