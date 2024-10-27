#!/bin/bash

## Downloads IMDB datasets.
## cd into ./scripts to run this.

set -e

data_dir="../622data_nogit/imdb"
mkdir -p $data_dir

# Download IMDB datasets
wget -O $data_dir/name.basics.tsv.gz https://datasets.imdbws.com/name.basics.tsv.gz
wget -O $data_dir/title.akas.tsv.gz https://datasets.imdbws.com/title.akas.tsv.gz 
wget -O $data_dir/title.basics.tsv.gz https://datasets.imdbws.com/title.basics.tsv.gz
wget -O $data_dir/title.crew.tsv.gz https://datasets.imdbws.com/title.crew.tsv.gz
wget -O $data_dir/title.episode.tsv.gz https://datasets.imdbws.com/title.episode.tsv.gz
wget -O $data_dir/title.principals.tsv.gz https://datasets.imdbws.com/title.principals.tsv.gz
wget -O $data_dir/title.ratings.tsv.gz https://datasets.imdbws.com/title.ratings.tsv.gz

# Extract the downloaded files
cd $data_dir
gunzip -f ./*.gz

# Go back to the scripts directory
cd ../../

echo "IMDB data download and extraction complete."
