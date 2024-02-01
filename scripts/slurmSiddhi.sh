#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --job-name=MatrixFree
#SBATCH --exclude=scn48-10g,scn50-10g,scn53-10g
#SBATCH -o slurm.out
##SBATCH -p testp

cd /nlsasfs/home/numerical/gourabp/matrixFree/run

. /nlsasfs/home/numerical/nikhilk/spack/spack/share/spack/setup-env.sh
spack load gcc cuda openmpi%gcc libxc alglib spglib numdiff p4est elpa dealii ninja nccl@2.19.3 cmake

export PMIX_MCA_psec=native
export OMP_NUM_THREADS=1
# export DFTFE_NUM_THREADS=4
export DEAL_II_NUM_THREADS=1
export OMP_SCHEDULE="nonmonotonic:dynamic"
export UCX_TLS=rc_x,sm,self,cuda_copy,cuda_ipc
export UCX_IB_GPU_DIRECT_RDMA=0

FeOrder=7
Quad=$((FeOrder+1))
Vec=1024
Approach=MF
Mesh=Uniform
Nodes=2
TaskperNode=8
Tasks=$((Nodes*TaskperNode))

srun -n $SLURM_NTASKS ./bindinghcoll.sh /nlsasfs/home/numerical/gourabp/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm > benchmarks/HX_multivector_mixPrec_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_gpu"$Tasks".out

# srun -n $SLURM_NTASKS ./bindinghcoll.sh /nlsasfs/home/numerical/gourabp/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm > benchmarks/HX_multivector_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_gpu"$Tasks".out

#srun -n $SLURM_NTASKS ./bindinghcoll.sh ncu -f -o profile --profile-from-start off --target-processes all --set full /nlsasfs/home/numerical/gourabp/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm > warp.out

#. /nlsasfs/home/numerical/gourabp/spack/share/spack/setup-env.sh
# nvidia-smi topo -m > topo.txt
