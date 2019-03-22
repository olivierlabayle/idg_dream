#!/usr/bin/env python
"""
We will use the standard_inchi_key and the uniprot_id to extract features for each samples.
Samples not containing these information are discarded.

Usage:
  create_training_set.py DATA_PATH [--db-port=P] [--db-host=H]

Options:
  -h --help        Show this screen.
  --db-port=P      Port on which idg-dream database is listening [default: 5432]
  --db-host=H      The host Ip address [default: 127.0.0.1]
"""

import docopt
import pandas as pd

from idg_dream.transformers import SequenceLoader, InchiLoader
from idg_dream.utils import get_engine

USED_COLS = ['standard_type', 'standard_units', 'standard_inchi_key', 'target_id', 'standard_value',
             'standard_relation']

KD_TYPES = ['-LOG KD', 'KD', 'Kd', 'LOG 1/KD', 'LOG KD', 'LOGKD', 'PKD']

KD_UNITS = ["NM", "-LOG(10) M"]


def load_training_data(path):
    return pd.read_csv(path, sep=",", header=0, usecols=USED_COLS)


def process_data(data, engine):
    """
    For more information about why I decided to apply those filters, please have a look at data_analysis.ipynb
    :param chunk: pd.DataFrame
    :return: pd.DataFrame
    """

    def convert(x):
        unit = x['standard_units']
        value = x['standard_value']
        if unit == "NM":
            return value * 1e-9
        elif unit == "-LOG(10) M":
            return 10 ** (-value)
        else:
            raise RuntimeError

    # Filter Na
    data.dropna(how="any", subset=USED_COLS, inplace=True)
    # Only keep measurements that are KD related
    data = data[data.standard_type.isin(KD_TYPES)]
    # Only keep measurements with some defined units
    data = data[data.standard_units.isin(KD_UNITS)]
    # Convert to M valued units
    data['standard_value'] = data.apply(convert, axis=1)
    # Keep only equal relation measurements
    data = data[data.standard_relation == '=']
    # Remove multiple targets measurements
    data = data[~data.target_id.str.contains(',')]
    # Remove (target,compound) pairs with more than one measurement
    key = ['standard_inchi_key', 'target_id']
    grouped = data.groupby(key).size()
    join_condition = grouped[grouped == 1].reset_index()[key]
    data = data.merge(join_condition, on=key, how='inner')
    # Remove outliers measurements
    data = data[(data.standard_value <= 1.7e-3) & (data.standard_value >= 1.e-10)]
    # Check the number of lines corresponds to the data analysis
    assert len(data) == 19269
    # Remove samples for which the protein_id has no sequence
    sequence_loader = SequenceLoader(engine=engine)
    data = sequence_loader.transform(data).dropna(how="any")
    # Remove samples for which the compound_id has no inchi
    inchi_loader = InchiLoader(engine=engine)
    data = inchi_loader.transform(data).dropna(how='any')
    # We will only use the following columns
    return data[["standard_inchi_key", "target_id", "standard_inchi", "sequence", "standard_value"]]


def create_training_set(db_port, db_host,data_path):
    engine = get_engine(db_port)
    data = load_training_data(data_path)
    print(f"Processing data.")
    processed_data = process_data(data, engine)
    print(f"Creating training_set table.")
    processed_data.to_sql("training_set", con=engine, index=False, if_exists='replace')


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    create_training_set(args['--db-port'], args['--db-host'], args['DATA_PATH'])
