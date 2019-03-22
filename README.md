# idg_dream
IDG Dream challenge

## Working environment

You can install the conda environment by running :

`conda env create -n idg_dream -f environment.yml`

Then activate it :

`conda activate idg_dream`

Depending if you have cuda enabled or not :

`pip install -r cpu_requirements.txt`

or

`pip install -r cuda_9_requirements.txt`

if you have another cuda version, check the pytorch documentation.


## Additional data

In order to use machine learning algorithms we will need additional information about the compounds 
and proteins.
* For compound information I decided to use CHEMBL as it is the provided compound_id in the dataset.
* For protein information I used Uniprot for the same reason

For both sources of information I decided to download the databases and to restore them to a local postgres 
database dedicated to the project.

To start the postgres container, run :

`IDG_DREAM_DB_PORT=5432 docker-compose -f chembl_db/docker-compose.yml up `

Note that IDG_DREAM_DB_PORT may be any port you like. 

Then to restore the data you will need to run through the following procedure in order.

### Restore CHEMBL


You can download the postgres dump from :

[download chembl database](ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)


Extract the files and upload the content of the previous download to the database by running :

`cat PATH_TO_CHEMBL_DUMP | docker exec -i idg-dream-db pg_restore -O --username=idg_dream -d idg_dream`

This takes some time. 
 
 
### Restore UNIPROT

For now we will only use the protein sequence thus requiring only the fasta dump. You can 
get it from :

[download uniprot](https://www.uniprot.org/downloads)

Extract the file and run :

`PYTHONPATH=PATH_TO_PROJECT:$PYTHONPATH python bin/import_uniprot.py FASTA_PATH`

Again, the port and host may be specified as options.

This can take some time too.

### Create training set

The data used to create the training set is issued from the DTC website :

[download dtc](https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv)

In order to understand the details of the process of the training set creation you can look at the
ipython notebook data_analysis.ipynb.

To create the table containing the training set, you can use the following script :

`PYTHONPATH=PATH_TO_PROJECT:$PYTHONPATH python bin/create_training_set.py DTC_DATA_PATH`

Again you can provide a port and host in the database is on a remote server.

### Tests

In order to run tests, you will need a postgres image running, you can use the given
docker-compose file :

`docker-compose -f idg_dream_db/docker-compose.test.yml up -d`

and run :

`python -m unittest discover tests/`