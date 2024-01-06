#!/bin/bash
#SBATCH -A MAT187_crusher
#SBATCH -J hipTest
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpu-bind=closest

export OMP_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=NUMA

module load PrgEnv-gnu
module load craype-accel-amd-gfx90a
module load rocm
module load cmake
module unload cray-libsci
export MPICH_GPU_SUPPORT_ENABLED=1
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

srun -n "$SLURM_NTASKS" /ccs/home/gourabp/matrixfreeHIP/build/release/real/dftfe parameterFile_a.prm > test_GPU_n"$SLURM_NTASKS".txt
