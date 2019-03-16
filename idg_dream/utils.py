import pickle
import pandas as pd

from sqlalchemy import create_engine


def get_engine(db_port, host='127.0.0.1'):
    return create_engine(f'postgresql+pg8000://idg_dream@{host}:{db_port}/idg_dream', echo=False)


def save_pipeline(pipeline, path):
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)


def load_pipeline(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_from_csv(path):
    data = pd.read_csv(path)
    X = data[['standard_inchi_key', 'target_id']]
    y = data[['standard_value']]
    return X, y


def load_from_db(engine):
    data = pd.read_sql("select * from training_set;", con=engine)
    X = data[['standard_inchi_key', 'target_id']]
    y = data[['standard_value']]
    return X, y
