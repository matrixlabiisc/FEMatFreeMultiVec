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

FeOrder=$1
Vec=$2
Approach=$3
Nodes=$4
Mesh=$5
TaskperNode=$6
Tasks=$((Nodes*TaskperNode))

echo $OMPI_COMM_WORLD_RANK

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
   ncu --metrics $metrics --profile-from-start off --target-processes all --csv /home/gourab/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm > benchmarks/blocksizeFlops_"$Mesh"_FP64Comm_"$Approach"_poly"$FeOrder"_vec"$Vec"_gpu"$Tasks".txt
else
    /home/gourab/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm
fi
