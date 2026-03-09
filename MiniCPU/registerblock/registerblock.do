onerror {exit -code 1}
vlib work
vlog -work work registerblock.vo
vlog -work work registerblock.vwf.vt
vsim -novopt -c -t 1ps -L cycloneiiils_ver -L altera_ver -L altera_mf_ver -L 220model_ver -L sgate work.registerblock_vlg_vec_tst -voptargs="+acc"
vcd file -direction registerblock.msim.vcd
vcd add -internal registerblock_vlg_vec_tst/*
vcd add -internal registerblock_vlg_vec_tst/i1/*
run -all
quit -f
