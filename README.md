# idg_dream
IDG Dream challenge

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

Extract the file and run `bin/import_uniprot.py FASTA_PATH --db-port=IDG_DREAM_DB_PORT`

Again, the port is the one you chose earlier.

This can take some time too.


### Tests

In order to run tests, you will need a postgres image running, you can use the given
docker-compose file :

`docker-compose -f idg_dream_db/docker-compose.test.yml up -d`