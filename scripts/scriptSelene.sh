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


for FeOrder in 6 7 8
do
mkdir -p benchmarks
/bin/cp -f parameterFile_a.prm parameterFile_a_poly"$FeOrder"_n1.prm

sed -i "49s/.*/  set POLYNOMIAL ORDER ELECTROSTATICS = "$FeOrder"/" parameterFile_a_poly"$FeOrder"_n1.prm
sed -i "7s/.*/  set DOMAIN VECTORS FILE = domainVectors_"$FeOrder".inp/" parameterFile_a_poly"$FeOrder"_n1.prm
sleep 1

ncu --metrics $metrics --profile-from-start off --target-processes all --csv /host_pwd/dft-fe/Buildfolder/build_multiVectorFEOperators/release/real/dftfe parameterFile_a_poly"$FeOrder"_n1.prm > benchmarks/poly"$FeOrder"_MF_nVec1024_gpu1_rank0.txt
# poly"$1"_MF_nVec1024_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt
# poly"$1"_CM_nVec1024_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt
# poly"$1"_MF_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt
# poly"$1"_CM_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt
# poly"$1"_dealii_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt


# poly"$1"_cells343_gpu"$OMPI_COMM_WORLD_SIZE".txt
# poly"$1"_vecShared8_cells343_gpu"$OMPI_COMM_WORLD_SIZE".txt
# poly"$1"_nVec1000_Domain9_gpu"$OMPI_COMM_WORLD_SIZE".txt
# poly"$1"_MFcuBLAS_nVec1000_Domain10_gpu"$OMPI_COMM_WORLD_SIZE".txt
done
