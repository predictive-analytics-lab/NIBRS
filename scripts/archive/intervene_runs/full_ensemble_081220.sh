#!/bin/bash

intervene real --data ../../data/NIBRS_full_202012081725.csv --output ../../output/full_ensemble --constraints ../config_files/constraints/full_20201209.yml --interventions ../config_files/interventions/full_20201208.yml --ensemble --bootstrap_prop 0.5
