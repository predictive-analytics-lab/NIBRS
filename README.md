# NIBRS PROJECT REPO

The project document can be found here: https://www.overleaf.com/read/mgnmfzvkyqby

There are two main sections of this repository:

- The SQL queries which operate on the NIBRS database.
- The Python code which utilizes this extracted data.

## Installation

### NIBRS Postgres DB

To setup the NIBRS database on your machine, you must:

1. Run `download_and_extract.py` - this will download the required NIBRS database files
2. Run the command `python add_to_db.py nibrs` to create the bash script "create_nibrs.sh"
3. Run the created bash script with `./create_nibrs.sh`

All of this presumes you already have postgres and python setup on your machine.

If you do not, please refer to:

- Postgres: https://www.postgresql.org/download/
- Miniconda (Python): https://docs.conda.io/en/latest/miniconda.html

### Data

To download the existing datasets, reports, and maps, you must install DVC:

`pip install dvc[gs]`

Followed by:

`dvc pull`

This should automatically download all the required datasets.


### Python

To install the required python libraries, please install a fresh python environment (using anaconda or such like), then run the command:

`pip install -r requirements.txt`

pointing to the `requirements.txt` file in the NIBRS directory.

## Running

### NIBRS Postgres DB Scripts

It is recommended you use some form of PSQL GUI, it makes running the scripts and investigating the output easier.

Please refer to: https://retool.com/blog/best-postgresql-guis-in-2020/ for a brief list of options.

Alternatively, if you REALLY want to use the command line, run something like:

`psql -d nibrs -a -f sql_scripts/cannabis_20210806.sql`

### Python Scripts

The current main output from the python data processing is the selection ratio.

To create this data for yourself, simply run, for example:

`python scripts/python/data_processing/selection_bias.py --year 2015-2019 --resolution county`

The year range currently available is 2015-2019.
The geographic resolutions available are: agency, county, region, state.

The file will be output in the `/data/output/` folder with the file name: `selection_ratio_{years}.csv`.

Enter:

`python scripts/python/data_processing/selection_bias.py -h`

on the command line for further help.

Additionally, the other data_processing scripts can be run with a similar syntax.

E.g:

`python scripts/python/data_processing/process_nsduh_data.py --year 2015-2019`

giving you the NSDUH data for the years 2015-2019.

The scripts themselves outline their intended usage, please investigate for yourself and email: b.butcher@sussex.ac.uk if you have any questions.
