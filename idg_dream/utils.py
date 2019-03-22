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


def collate_to_sparse_tensors(batch, protein_input_size=26**3, compound_input_size=1024, device=torch.device("cpu")):
    proteins_indexes = [[], []]
    compounds_indexes = [[], []]
    y = []
    n_samples = len(batch)

    for i, sample in enumerate(batch):
        temp_proteins_indexes = sample[0]["protein_input"]
        temp_compounds_indexes = sample[0]["compound_input"]

        proteins_indexes[0].extend([i] * len(temp_proteins_indexes))
        proteins_indexes[1].extend(temp_proteins_indexes)

        compounds_indexes[0].extend([i] * len(temp_compounds_indexes))
        compounds_indexes[1].extend(temp_compounds_indexes)

        y.append(sample[1].tolist())

    proteins_values = [1] * len(proteins_indexes[0])
    compounds_values = [1] * len(compounds_indexes[0])

    protein_input = torch.sparse.FloatTensor(torch.LongTensor(proteins_indexes),
                                             torch.FloatTensor(proteins_values),
                                             torch.Size([n_samples, protein_input_size]),
                                             device=device)

    compound_input = torch.sparse.FloatTensor(torch.LongTensor(compounds_indexes),
                                             torch.FloatTensor(compounds_values),
                                             torch.Size([n_samples, compound_input_size]),
                                              device=device)

    return ({"protein_input": protein_input, "compound_input": compound_input}, torch.FloatTensor(y, device=device))
