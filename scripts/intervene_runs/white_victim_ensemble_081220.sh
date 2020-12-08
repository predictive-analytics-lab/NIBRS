#!/bin/bash

intervene real --data ../../data/NIBRS_full_202012071235.csv --output ../../output/white_victim --constraints ../config_files/constraints/full.yml --interventions ../config_files/interventions/full_white_victim.yml --ensemble --bootstrap_prop 0.5
