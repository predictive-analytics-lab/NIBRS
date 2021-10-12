# Racial Disparities in the Enforcement ofMarijuana Violations in the US

## Installation

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

## Reproducing the Results

### Enforcement Ratios

In order to produce the enforcement ratios you must either download the pre-requisite datasets from DVC, using `dvc pull`, or you may produce all results yourself using the following steps:

First, download the appropriate 

Second, produce the appropriate NIBRS dataset, using the query_nibrs.py python script. The script has a number of arguments that can be explored with: `python query_nibrs.py -h`.

