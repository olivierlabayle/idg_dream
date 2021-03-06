import pickle
import pandas as pd
import torch
import numpy as np
import scipy.sparse
import itertools
import dgl
from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from rdkit.Chem import MolFromInchi
from skorch import NeuralNetRegressor
from sqlalchemy import create_engine


NB_AMINO_ACID = len(ExtendedIUPACProtein.letters)


def get_engine(db_port, host='127.0.0.1'):
    return create_engine(f'postgresql+pg8000://idg_dream:idg_dream@{host}:{db_port}/idg_dream', echo=False)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_from_csv(path, y_name='standard_value'):
    data = pd.read_csv(path)
    if y_name:
        y = data[['standard_value']].values
        return data.drop('standard_value', axis=1), y
    else:
        return data, None


def load_from_db(engine):
    data = pd.read_sql("select * from training_set;", con=engine)
    y = data[['standard_value']].values
    X = data.drop('standard_value', axis=1)
    return X, y


def update_sparse_data_from_dict(data, row_indexes, col_indexes, dictionary, index):
    row_indexes.extend([index] * len(dictionary))
    temp_col_indexes, values = zip(*dictionary.items())
    col_indexes.extend(temp_col_indexes)
    data.extend(values)


def update_sparse_data_from_list(data, row_indexes, col_indexes, liste, index):
    n = len(liste)
    row_indexes.extend([index] * n)
    col_indexes.extend(liste)
    data.extend([1] * n)


def to_sparse(X, update_method, dim, return_torch=False):
    data, row_indexes, col_indexes = [], [], []
    n = len(X)
    for index, item in enumerate(X):
        update_method(data, row_indexes, col_indexes, item, index)
    if return_torch:
        return torch.sparse.FloatTensor(torch.LongTensor([row_indexes, col_indexes]),
                                             torch.FloatTensor(data),
                                             torch.Size([n, dim]))
    return scipy.sparse.csr_matrix((data, (row_indexes, col_indexes)), shape=(n, dim))


def collate_to_sparse_tensors(batch, protein_input_size=26 ** 3, compound_input_size=1024, device=torch.device("cpu")):
    proteins_indexes = [[], []]
    proteins_values = []
    compounds_indexes = [[], []]
    compounds_values = []
    y = []
    n_samples = len(batch)

    for i, sample in enumerate(batch):
        temp_proteins_indexes = sample[0]["protein_input"]
        temp_compounds_indexes = sample[0]["compound_input"]
        # Protein extraction
        update_sparse_data_from_dict(proteins_values, proteins_indexes[0], proteins_indexes[1], temp_proteins_indexes,
                                     i)
        # Compound extraction
        update_sparse_data_from_list(compounds_values, compounds_indexes[0], compounds_indexes[1],
                                     temp_compounds_indexes, i)
        # target extraction
        y.append(sample[1].tolist())

    protein_input = torch.sparse.FloatTensor(torch.LongTensor(proteins_indexes),
                                             torch.FloatTensor(proteins_values),
                                             torch.Size([n_samples, protein_input_size])).to(device)

    compound_input = torch.sparse.FloatTensor(torch.LongTensor(compounds_indexes),
                                              torch.FloatTensor(compounds_values),
                                              torch.Size([n_samples, compound_input_size])).to(device)

    return (
        {"protein_input": protein_input, "compound_input": compound_input},
        torch.FloatTensor(y).to(device)
    )


def inchi_to_graph(inchi, max_atomic_number=118, device=torch.device('cpu')):
    """
    Converts an inchi string to a DGL Graph object and associate the one hot encoding features for each node.
    :param inchi: An inchi string
    :param max_atomic_number: The max_atomic_number determines the final size of the nodes feature matrix
    :return: DGL.Graph
    """
    mol = MolFromInchi(inchi)
    num_atoms = mol.GetNumAtoms()
    # DGLGraph creation from rdkit mol object
    graph = dgl.DGLGraph()
    graph.add_nodes(num_atoms)
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dest = bond.GetEndAtomIdx()
        graph.add_edge(src, dest)
        # Edges in DGL are directional, to ensure bidirectionality, add reverse edge
        graph.add_edge(dest, src)

    # One hot encoding for nodes features
    one_hot_indexes = []
    for atom_index in range(num_atoms):
        one_hot_indexes.append([mol.GetAtomWithIdx(atom_index).GetAtomicNum()])
    graph.ndata['x'] = torch.zeros(num_atoms, max_atomic_number) \
        .scatter_(1, torch.tensor(one_hot_indexes), 1).to(device)

    return graph


def get_kmers_mapping(kmer_size):
    return dict((''.join(letters), index) for index, letters in
                enumerate(itertools.product(ExtendedIUPACProtein.letters, repeat=kmer_size)))


def sort_batch(batch):
    protein_inputs, protein_lengths, compound_inputs, targets = [], [], [], []
    for x, y in batch:
        protein_inputs.append(x['protein_input'])
        protein_lengths.append(x['protein_lengths'])
        compound_inputs.append(x['compound_input'])
        targets.append(y)

    order = np.argsort(protein_lengths)[::-1]
    return np.array(compound_inputs)[order], np.array(protein_inputs)[order], \
           np.array(protein_lengths)[order], np.array(targets, dtype=np.float32)[order]


def collate_bilstm_fingerprint(batch, device=torch.device("cpu"), ecfp_dim=2 ** 10):
    """
    need to sort by length
    :return:
    """
    compound_inputs, protein_inputs, protein_lengths, targets = sort_batch(batch)

    return (
        {
            "compound_input": to_sparse(compound_inputs, update_sparse_data_from_list, ecfp_dim,
                                        return_torch=True).to(device),
            "protein_input": torch.from_numpy(protein_inputs).to(device),
            "protein_lengths": torch.from_numpy(protein_lengths).to(device)
        },
        torch.from_numpy(targets).to(device)
    )


def collate_graph_bilstm(batch, device=torch.device("cpu")):
    compound_inputs, protein_inputs, protein_lengths, targets = sort_batch(batch)
    return (
        {
            "compound_input": compound_inputs,
            "protein_input": torch.from_numpy(protein_inputs).to(device),
            "protein_lengths": torch.from_numpy(protein_lengths).to(device)
        },
        torch.from_numpy(targets).to(device)
    )


class DGLHackedNeuralNetRegressor(NeuralNetRegressor):
    def infer(self, x, **fit_params):
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        return self.module_(x, **fit_params)
