import torch
import torch.nn as nn


class SparseLinear(nn.Linear):
    def forward(self, input):
        return torch.sparse.mm(input, self.weight.t()) + self.bias


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
            nn.Linear(2 * self.embedding_dim, self.embedding_dim), nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

    def forward(self, protein_input, compound_input):
        protein_embedding = self.protein_branch(protein_input)
        compound_embedding = self.compound_branch(compound_input)
        joined = torch.cat((protein_embedding, compound_embedding), dim=1)
        return self.output_branch(joined)
