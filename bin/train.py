#!/usr/bin/env python

"""
Trains the given pipeline against the training_data

Usage:
  train.py PIPELINE_NAME PATH_OUT [--training-sample=T] [--db-port=P] [--config-path=C]

Options:
  -h --help              Show this screen.
  --db-port=P            Port on which idg-dream database is listening [default: 5432].
  --training-sample=T    Path to a training sample, if provided, the pipeline will be trained against that sample.
  --config-path=C        Json file containing the pipeline's configuration
"""

import json
import docopt
import idg_dream.pipelines as idg_dream_pipelines

from idg_dream.utils import get_engine, save_pipeline, load_from_csv, load_from_db


def main(pipeline_name, path_out, db_port, config_path, training_sample_path):
    engine = get_engine(db_port)

    config_dict = {}
    if config_path:
        config_dict = json.load(config_path)

    pipeline = getattr(idg_dream_pipelines, pipeline_name)(
        engine,
        **config_dict

    )

    if training_sample_path:
        X, y = load_from_csv(training_sample_path)
    else:
        X, y = load_from_db(training_sample_path)

    pipeline.fit(X, y)

    save_pipeline(pipeline, path_out)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(
        args["PIPELINE_NAME"],
        args["PATH_OUT"],
        db_port=args["--db-port"],
        config_path=args["--config-path"],
        training_sample_path=args["--training-sample"]
    )