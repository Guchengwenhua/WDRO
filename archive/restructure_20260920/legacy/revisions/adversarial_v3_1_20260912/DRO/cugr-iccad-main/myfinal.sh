data="ispd18_sample"
this_path="/remote-home/Chenlin/chip/routing/Differentiable-Global-Router-main" # # DGR path
benchmark_path="/remote-home/Chenlin/chip/routing/Differentiable-Global-Router-main/cu-gr-2-main/benchmark"
cugr2="/remote-home/Chenlin/chip/routing/Differentiable-Global-Router-main/cu-gr-2-main"
method="DGR"

./route -via_cost 20 -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -sort 1 -output $benchmark_path/$data/$data.$method.guide -dgr $this_path/CUGR2_guide/CUgr_$data\_$method.txt > $cugr2/GR_log/$data\_$method.log