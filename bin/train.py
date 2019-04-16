#!/usr/bin/env python

"""
Trains the given pipeline against the training_data

Usage:
  train.py PIPELINE_NAME PATH_OUT [--training-sample=T] [--db-port=P] [--db-host=H] [--config-path=C] [--random-state=R]

Options:
  -h --help              Show this screen.
  --db-port=P            Port on which idg-dream database is listening [default: 5432]
  --db-host=H            IP Adress of the idg_dream database [default: 127.0.0.1]
  --training-sample=T    Path to a training sample, if provided, the pipeline will be trained against that sample.
  --config-path=C        Json file containing the pipeline's configuration
  --random-state=R       The integer number used to fix the random seed [default: 0]
"""

import os
import sys
import docopt
import torch
import numpy as np

import idg_dream.pipelines as idg_dream_pipelines
from importlib import import_module
from idg_dream.utils import get_engine, save_pickle, load_from_csv, load_from_db


def main(factory_name, path_out, db_port, db_host, config_path, training_sample_path, random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    engine = None
    if not training_sample_path:
        engine = get_engine(db_port, db_host)

    config_dict = {"engine": engine}
    if config_path:
        module_path, module_name = os.path.split(config_path)
        sys.path.append(module_path)
        config_module = import_module(os.path.splitext(module_name)[0])
        config_dict.update(config_module.CONFIG[factory_name])

    pipeline = getattr(idg_dream_pipelines, factory_name)()(
        **config_dict
    )

    if training_sample_path:
        X, y = load_from_csv(training_sample_path)
    else:
        X, y = load_from_db(engine)

    print(f"Dataset size : {len(X)}")

    pipeline.fit(X, y)

    save_pickle(pipeline, path_out)

    print(f"Model saved at : {path_out}")


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(
        args["PIPELINE_NAME"],
        args["PATH_OUT"],
        db_port=args["--db-port"],
        db_host=args['--db-host'],
        config_path=args["--config-path"],
        training_sample_path=args["--training-sample"],
        random_state=int(args["--random-state"])
    )
