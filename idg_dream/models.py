import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_kmers, num_fingerprints, embedding_dim=10):
        super().__init__()
        self.num_kmers = num_kmers
        self.num_fingerprints = num_fingerprints
        self.embedding_dim = embedding_dim
        self.protein_embedding = nn.Embedding(num_embeddings=num_kmers, embedding_dim=embedding_dim)
        self.compound_embedding = nn.Embedding(num_embeddings=num_fingerprints, embedding_dim=embedding_dim)
        self.output_branch = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim), nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )

    def forward(self, protein_input, compound_input):
        protein_embedding = self.protein_embedding(protein_input)
        compound_embedding = self.compound_embedding(compound_input)
        joined = torch.cat((protein_embedding.mean(dim=1), compound_embedding.mean(dim=1)), dim=1)
        return self.output_branch(joined)
