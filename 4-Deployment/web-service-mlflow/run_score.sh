#!/usr/bin/bash
# coding: utf-8

YEAR=2021
MONTH=2
TAXI_TYPE="green"
RUN_ID="10f4197008104ad183466cdb19e26c4e"

python3 score.py \
    --year ${YEAR} \
    --month ${MONTH} \
    --taxi_type ${TAXI_TYPE} \
    --run_id ${RUN_ID} 
 