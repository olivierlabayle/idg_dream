# idg_dream
IDG Dream challenge

## Additional data

In order to use machine learning algorithms we will need additional information about the compounds 
and proteins.
* For compound informations I decided to use CHEMBL as it is the provided compound_id in the dataset.
* For protein information I used Uniprot for the same reason

For both sources of information I decided to download the databases and to restore them to a local 
database dedicated to the project.

### Restore CHEMBL

I decided to restore CHEMBL database as a postgres database.

To start the postgres container, run :

`docker-compose -f chembl_db/docker-compose.yml up `

You can download the data from :
[download chembl database](ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)


Extract the files and upload the content of the previous download to the database by running :

`cat chembl_24_postgresql.dmp | docker exec -i chembl_db_chembl_1 pg_restore -U postgres -d chembl_24`


### Restore UNIPROT