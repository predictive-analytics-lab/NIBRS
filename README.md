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

1. Download the appropriate raw NIBRS files for each state. This can be achieved by running the python script: `python data_downloading/download_and_extract.py`. This currently defaults to downloading years 2010-2019 and over all states. This can be changed from within the script.

2. Produce the appropriate NIBRS dataset. Using the query_nibrs.py python script `python scripts/python/data_processing/query_nibrs.py`. The script has a number of arguments that can be explored with: `python query_nibrs.py -h`.

3. Produce the enforcement ratios. Run the selection_ratio.py script `scripts/python/data_processing/selection_bias.py` with appropriate arguments. Run `scripts/python/data_processing/selection_bias.py -h` for help, or consult the image below:

![Selection Bias Help](docs/sb_help.svg)

### Paper Figures

There are many scripts used to create the figures and tables in the paper, please consult the table below to find which script corresponds to which figure:

#### Figures

| Figure | Script |
|:---:|:---:|
| 1 | R/generate_plots_4paper.R |
| 2 | python/enforcement_ratio_model_plots.py |
| 3 | R/generate_plots_4paper.R |
| S1 | python/nsduh_usage_plot.py |
| S2 | R/generate_plots_4paper.R |
| S3 | python/enforcement_ratio_location_plot.py |
| S4 | python/legalized_states_agency_reporting_plot.py |
| S5 | NEEDS ADDING |
| S6 | NEEDS ADDING |
| S7 | R/generate_plots_4paper.R |
| S8 | python/enforcement_ratio_model_plots.py |
| S9 | R/generate_plots_4paper.R |
| S10 | python/enforcement_ratio_model_plots.py |
| S11 | R/generate_plots_4paper.R |
| S12 | python/enforcement_ratio_model_plots.py |
| S13 | R/generate_plots_4paper.R |
| S14 | python/enforcement_ratio_model_plots.py |
| S15 | R/generate_plots_4paper.R |
| S16 | Doesn't exist in paper? Bug in latex maybe? |
| S17 | python/enforcement_ratio_model_plots.py |
| S18 | R/generate_plots_4paper.R |
| S19 | python/time_distribution_plot.py + python/enforcement_ratio_model_plots.py |
| S20 | R/generate_plots_4paper.R |

#### Tables

| Table | Script |
|:---:|:---:|
| 1 | R/generate_plots_4paper.R |
| 2 | python/enforcement_ratio_model_plots.py |
| 3 | R/generate_plots_4paper.R |
| S1 | python/nsduh_usage_plot.py |
| S2 | R/generate_plots_4paper.R |
| S3 | python/enforcement_ratio_location_plot.py |
| S4 | python/legalized_states_agency_reporting_plot.py |
| S5 | Need to add |
| S6 | Need to add |
| S7 | R/generate_plots_4paper.R |
| S8 | python/enforcement_ratio_model_plots.py |
| S9 | R/generate_plots_4paper.R |
| S10 | python/enforcement_ratio_model_plots.py |
| S11 | R/generate_plots_4paper.R |
| S12 | python/enforcement_ratio_model_plots.py |
| S13 | R/generate_plots_4paper.R |
| S14 | python/enforcement_ratio_model_plots.py |
| S15 | R/generate_plots_4paper.R |
| S16 | Doesn't exist in paper? Bug in latex maybe? |
| S17 | python/enforcement_ratio_model_plots.py |
| S18 | R/generate_plots_4paper.R |
| S19 | python/time_distribution_plot.py + python/enforcement_ratio_model_plots.py |
| S20 | R/generate_plots_4paper.R |