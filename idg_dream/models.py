import torch
import math
import torch.nn as nn


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


class GraphNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass