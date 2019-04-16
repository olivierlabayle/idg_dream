#!/usr/bin/env python
"""
Executable used to output precictions given a template file

Usage:
  predict.py TEMPLATE_PATH MODEL_PATH OUTPUT_PATH [--db-port=P] [--db-host=H]

Options:
  -h --help        Show this screen.
  --db-port=P      Port on which idg-dream database is listening [default: 5432]
  --db-host=H      The host Ip address [default: 127.0.0.1]
"""

import docopt
from idg_dream.utils import load_from_csv, load_pickle


def main(model_path, template_path, output_path, db_host='127.0.0.1', db_port='5432'):
    model = load_pickle(model_path)
    X, _ = load_from_csv(template_path)
    X['pKd_[M]_pred'] = model.predict(X)
    X.to_csv(output_path)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(
        args['MODEL_PATH'],
        args["TEMPLATE_PATH"],
        args['OUTPUT_PATH'],
        db_host=args['--db-host'],
        db_port=args['--db-port']
    )
