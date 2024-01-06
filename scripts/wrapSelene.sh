#!/bin/bash

# Time
metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"

# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum"

ncu --metrics $metrics --profile-from-start off --target-processes all --csv /host_pwd/dft-fe/Buildfolder/build_multiVectorFEOperators/release/real/dftfe parameterFile_"$2"_poly"$1"_N"$OMPI_COMM_WORLD_SIZE".prm > benchmarks/HN_nvec1024_"$2"_poly"$1"_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt
