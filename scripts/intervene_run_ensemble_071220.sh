#!/bin/bash

intervene real --data ../data/NIBRS_full_202012071235.csv --output ../output/ensemble_run_071220 --constraints config_files/full_constraints.yml --interventions config_files/full_intervention.yml --ensemble --bootstrap_prop 0.5
