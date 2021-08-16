#!/usr/bin/env bash

for i in `seq $2 $3`
do
    psql -d nibrs_$i -t -A -F"," -f $1 > nibrs_${i}.csv
done
