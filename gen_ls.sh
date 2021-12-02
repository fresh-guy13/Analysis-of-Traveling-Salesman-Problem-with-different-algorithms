#!/usr/bin/bash

for instance in DATA/*.tsp
do
    for alg in LS1 LS2
    do
        city="$(basename -- $instance .tsp)"
        mkdir -p "lsbatchout/$city$alg"
        mydir="lsbatchout/$city$alg"
        python3 -m tsp.tsp_batch -batch-size 10 -alg $alg -inst $instance -odir $mydir -time 100
    done
done
