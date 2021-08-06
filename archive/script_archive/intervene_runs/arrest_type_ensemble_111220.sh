#!/bin/bash

intervene real --data ../../data/NIBRS_full_202012081725.csv --output ../../output/arrest_type --constraints ../config_files/constraints/full_20201211.yml --interventions ../config_files/interventions/arrest_type_101220.yml --ensemble --bootstrap_prop 0.8
