#!/bin/bash

intervene real --data ../../data/20210112_all2019_5levels.csv --output ../../output/all_ensemble --constraints ../config_files/constraints/full_20210114.yml --interventions ../config_files/interventions/full_20210114.yml --ensemble --bootstrap_prop 0.5
