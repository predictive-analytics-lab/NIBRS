#!/bin/bash

intervene real --data ../../data/NIBRS_drug_20210324.csv --output ../../output/drug_ensemble --constraints ../config_files/constraints/drugs_20210319.yml --interventions ../config_files/interventions/drugs_20210324.yml --ensemble --bootstrap_prop 0.5 --timeout 180000
