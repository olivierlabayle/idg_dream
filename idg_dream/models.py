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
        self.relu = nn.ReLU()
        self.protein_linear = SparseLinear(num_kmers, self.embedding_dim)
        self.compound_linear = SparseLinear(num_fingerprints, self.embedding_dim)
        self.joined_linear = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.out_linear = nn.Linear(self.embedding_dim, 1)
        # self.output_branch = nn.Sequential(
        #     nn.Linear(2 * self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embedding_dim, 1)
        # )

    def forward(self, protein_input, compound_input):
        protein_embedding = self.protein_linear(protein_input)
        protein_embedding = self.relu(protein_embedding)
        compound_embedding = self.compound_linear(compound_input)
        compound_embedding = self.relu(compound_embedding)
        joined = torch.cat((protein_embedding, compound_embedding), dim=1)
        joined = self.joined_linear(joined)
        joined = self.relu(joined)
        out = self.out_linear(joined)
        return out
