#!/usr/bin/env python

"""
Evaluates two pipelines pipelines on the same cross validation set with the given parameters grid

Usage:
  model_selection.py PIPELINE_NAME CONFIG_PATH SAVE_PATH [--random-state=R] [--training-sample=T] [--db-port=P]

Options:
  -h --help              Show this screen.
  --random-state=R       The random state associated with the evaluation [default: 0]
  --db-port=P            Port on which idg-dream database is listening [default: 5432]
  --training-sample=T    Path to a training sample, if provided, the pipeline will be trained against that sample.
"""

import os
import sys
import docopt
import idg_dream.pipelines as idg_dream_pipelines
from importlib import import_module
from idg_dream.utils import get_engine, save_pickle, load_from_csv, load_from_db
from sklearn.model_selection import KFold, GridSearchCV


def main(pipeline_name, config_path, save_path, random_state=0, db_port=5432, training_sample_path=None):
    engine = None
    if not training_sample_path:
        engine = get_engine(db_port)

    pipeline = getattr(idg_dream_pipelines, pipeline_name)(engine=engine)

    cv = KFold(shuffle=True, random_state=random_state)

    module_path, module_name = os.path.split(config_path)
    sys.path.append(module_path)
    config_module = import_module(os.path.splitext(module_name)[0])
    param_grid = config_module.GRIDS[pipeline_name]

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=cv)

    if training_sample_path:
        X, y = load_from_csv(training_sample_path)
    else:
        X, y = load_from_db(engine)

    grid_search.fit(X, y)

    save_pickle(grid_search, save_path)
    print("End of cross validation.")


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args['PIPELINE_NAME'], args["CONFIG_PATH"], args['SAVE_PATH'],
         random_state=int(args['--random-state']),
         db_port=args['--db-port'],
         training_sample_path=args['--training-sample'])