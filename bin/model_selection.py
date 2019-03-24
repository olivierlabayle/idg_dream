#!/usr/bin/env python

"""
Evaluates two pipelines pipelines on the same cross validation set with the given parameters grid

Usage:
  train.py PIPELINE_NAME_1 [--training-sample=T] [--db-port=P] [--config-path=C]

Options:
  -h --help              Show this screen.
  --db-port=P            Port on which idg-dream database is listening [default: 5432]
  --training-sample=T    Path to a training sample, if provided, the pipeline will be trained against that sample.
  --config-path=C        Json file containing the pipeline's configuration
"""

import os
import sys
import docopt
import idg_dream.pipelines as idg_dream_pipelines
from importlib import import_module
from idg_dream.utils import get_engine, save_pipeline, load_from_csv, load_from_db
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error


def main(pipeline_name, db_port=5432, config_path=None, training_sample_path=None):
    engine = None
    if not training_sample_path:
        engine = get_engine(db_port)


    pipeline = getattr(idg_dream_pipelines, pipeline_name)(**config_dict)

    cv = KFold(shuffle=True, random_state=0)

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=mean_squared_error, cv=cv)

    if training_sample_path:
        X, y = load_from_csv(training_sample_path)
    else:
        X, y = load_from_db(engine)

    grid_search.fit(X, y)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args['PIPELINE_NAME_1'],
         db_port=args['--db-port'],
         config_path=args['--config-path'],
         training_sample_path=args['--training-sample-path'])
