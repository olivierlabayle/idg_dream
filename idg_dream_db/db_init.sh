#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER idg_dream;
    CREATE DATABASE idg_dream;
    GRANT ALL PRIVILEGES ON DATABASE idg_dream TO idg_dream;
EOSQL
