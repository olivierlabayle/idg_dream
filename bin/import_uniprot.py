#!/usr/bin/env python
"""
Import uniprot to the idg_dream database.

Usage:
  naval_fate FASTA_PATH [--db-port=P]

Options:
  -h --help     Show this screen.
  --db-port=P   Port on which idg-dream database is listening [default: 5432].
"""

import docopt
import pandas as pd
from Bio import SeqIO
from sqlalchemy import create_engine

from idg_dream.utils import get_engine


def upload_uniprot(fasta_path, db_port=5432):
    engine = get_engine(db_port)
    uniprot_df = pd.DataFrame(
        [[record.id.split('|')[1], str(record.seq)] for record in SeqIO.parse(fasta_path, "fasta")],
        columns=["uniprot_id", "sequence"])
    uniprot_df.to_sql("uniprot_proteins", con=engine, if_exists='replace')



if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    upload_uniprot(args["FASTA_PATH"], db_port=args["--db-port"])
