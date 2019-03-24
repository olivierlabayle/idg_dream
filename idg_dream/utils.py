import pickle
import pandas as pd
import torch
from sqlalchemy import create_engine


def get_engine(db_port, host='127.0.0.1'):
    return create_engine(f'postgresql+pg8000://idg_dream:idg_dream@{host}:{db_port}/idg_dream', echo=False)


def save_pipeline(pipeline, path):
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_from_csv(path):
    data = pd.read_csv(path)
    y = data[['standard_value']].values
    X = data.drop('standard_value', axis=1)
    return X, y


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
