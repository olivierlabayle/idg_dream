#!/usr/bin/env python
"""
We will use the standard_inchi_key and the uniprot_id to extract features for each samples.
Samples not containing these information are discarded.

Usage:
  create_training_set.py DATA_PATH [--db-port=P] [--chunk-size=C]

Options:
  -h --help        Show this screen.
  --db-port=P      Port on which idg-dream database is listening [default: 5432].
  --chunk-size=C   The dataset is quite big and would not fit on a small memory [default: 1e5]
"""

import docopt
import pandas as pd
from sqlalchemy import create_engine


def drop_training_set(engine):
    engine.execute("DROP TABLE IF EXISTS training_set;")


def drop_dtc_table(engine):
    engine.execute("DROP TABLE IF EXISTS dtc_table;")


def load_training_data(path, chunk_size):
    for chunk in pd.read_csv(path, sep=",", header=0, chunksize=chunk_size):
        yield chunk


def process_chunk(chunk):
    chunk.dropna(how="all", subset=["standard_inchi_key"], inplace=True)
    chunk.dropna(how="all", subset=["target_id"], inplace=True)
    return chunk[["standard_inchi_key", "target_id"]]


def create_training_set(db_port, data_path, chunk_size):
    engine = create_engine(f'postgresql+pg8000://idg_dream@127.0.0.1:{db_port}/idg_dream', echo=False)
    drop_training_set(engine)
    for chunk in load_training_data(data_path, chunk_size=chunk_size):
        processed_chunk = process_chunk(chunk)
        n = len(processed_chunk)
        if n > 0:
            print(f"Processing new chunk of size :{n}.")
            processed_chunk.to_sql("training_set", con=engine, index=False, if_exists='append')


def create_dtc_table(db_port, data_path, chunk_size):
    engine = create_engine(f'postgresql+pg8000://idg_dream@127.0.0.1:{db_port}/idg_dream', echo=False)
    connection = engine.connect()
    drop_dtc_table(engine)
    for chunk in load_training_data(data_path, chunk_size=chunk_size):
        print(f"Inserting new chunk of size :{len(chunk)}.")
        chunk.to_sql("dtc_table", con=connection, index=False, if_exists='append')
    connection.close()


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    create_dtc_table(args['--db-port'], args['DATA_PATH'], int(float(args["--chunk-size"])))
