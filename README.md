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

`docker-compose -f chembl_db/docker-compose.yml up `

Then to restore the data you will need to run through the following procedure in order.

### Restore CHEMBL


You can download the postgres dump from :
[download chembl database](ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)


Extract the files and upload the content of the previous download to the database by running :

`cat chembl_24_postgresql.dmp | docker exec -i idg-dream-db pg_restore -O --username=postgres -d idg_dream`

This may take a moment. Note that `chembl_24_postgresql.dmp` filename will depend on the time of release.
 
 
### Restore UNIPROT

For now we will only use the protein sequence thus require only the fasta dump. You can \
get it from :
[download uniprot](https://www.uniprot.org/downloads)

Extract the file and run `bin/import_uniprot.py`
