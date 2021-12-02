#!/usr/bin/bash

for instance in DATA/*.tsp
do
    for alg in BnB
    do
        city="$(basename -- $instance .tsp)"
        mkdir -p "batchout/$city$alg"
        mydir="batchout/$city$alg"
        python3 -m tsp.tsp_batch -batch-size 1 -alg $alg -inst $instance -odir $mydir -time 600 -appendtottime
    done
done
