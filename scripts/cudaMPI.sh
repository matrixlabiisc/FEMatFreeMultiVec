#!/bin/bash
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export CUDA_VISIBLE_DEVICES=0
#export UCX_NET_DEVICES=mlx5_0:1
#export UCX_RNDV_SCHEME=get_zcopy

/host_pwd/dft-fe/hpcx/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.12-x86_64/ompi/tests/osu-micro-benchmarks-5.8-cuda/osu_latency D H > benchmarks.txt
