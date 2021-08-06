#!/bin/bash

intervene real --data ../../data/NIBRS_full_all2019_202012091928.csv --output ../../output/arrest_type_ensemble --constraints ../config_files/constraints/full_20201209.yml --interventions ../config_files/interventions/arrest_type_101220.yml --ensemble --bootstrap_prop 0.8
