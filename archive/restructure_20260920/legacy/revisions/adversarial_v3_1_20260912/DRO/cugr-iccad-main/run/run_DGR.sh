#!/bin/bash

for i in {1..10}
do
    test_name="test$i"
    ./route \
    -via_cost 20 \
    -lef ../benchmark/ispd19_${test_name}/ispd19_${test_name}.input.lef \
    -def ../benchmark/ispd19_${test_name}/ispd19_${test_name}.input.def \
    -sort 1 \
    -output ../benchmark/ispd19_${test_name}/ispd19_${test_name}.DGR.guide \
    -dgr ../../CUGR2_guide/CUgr_ispd19_${test_name}_DGR.txt
done
