#!/bin/bash

intervene real --data ../../data/NIBRS_full_202012071235.csv --output ../../output/stranger_firearm --constraints ../config_files/constraints/full.yml --interventions ../config_files/interventions/full_stranger_firearm.yml --ensemble --bootstrap_prop 0.5
