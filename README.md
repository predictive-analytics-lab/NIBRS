## NIBRS PROJECT REPO

The project document can be found here: https://www.overleaf.com/read/mgnmfzvkyqby


### Installation

#### NIBRS Postgres DB

There are two main sections of this repository:

    - The SQL queries which operate on the NIBRS database 
    - The Python code which utilizies this extracted data.

To setup the NIBRS database on your machine, you must:

1. Run `download_and_extract.py` - this will download the required NIBRS database files
2. Run the command `python add_to_db.py nibrs` to create the bash script "create_nibrs.sh"
3. Run the created bash script with `./create_nibrs.sh`

All of this presumes you already have postgres and python setup on your machine.

#### Data

To download the existing datasets, reports, and maps, you must install DVC:

`pip install dvc[gdrive]`

As the DVC remote is stored on Bradley's google drive, the only way to gain access is via Bradley. Please contact him for authentication.


#### Python Scripts

There is no intensive setup required for the python scripts, standard libraries are used. Simply pip install them as required.

A list of these with a frozen anaconda environment / requirements.txt will be made available shortly.

