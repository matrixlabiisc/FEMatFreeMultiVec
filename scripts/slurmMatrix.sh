#!/bin/sh
#SBATCH --job-name=Eigen           # Job name
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
##SBATCH --nodelist=node1
#SBATCH --time=06:00:00                # Time limit hrs:min:sec
#SBATCH -o slurm.out
#SBATCH --gres=gpu:1
##SBATCH --exclusive
##SBATCH --partition=debug

echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NPROCS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load spack
. /shared/spack/share/spack/setup-env.sh
spack load gcc@11.3.0 cuda boost@1.76.0 intel-oneapi-mkl cmake gmake ninja openmpi

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll_hcoll_enable=0
export UCX_NET_DEVICES=mlx5_0:1

cd /home/gourab/matrixFree/build/

FeOrder=6
Quad=$((FeOrder+3))
Vec=32
Approach=MF
Mesh=Uniform
Nodes=1
TaskperNode=1
Tasks=$((Nodes*TaskperNode))


# mpirun -n "$SLURM_NPROCS" --mca btl ^openib /home/gourab/matrixFree/build/release/real/dftfe parameterFile_"$Mesh"_"$Approach"_poly"$FeOrder"_vec"$Vec"_N"$Nodes".prm > benchmarks/HX_q"$Quad"_"$Mesh"_FP64Comm_"$Approach"_poly"$FeOrder"_vec"$Vec"_gpu"$Tasks".txt

mpirun -n "$SLURM_NPROCS" --mca btl ^openib "./wrapMatrix.sh" "$FeOrder" "$Vec" "$Approach" "$Nodes" "$Mesh" "$TaskperNode"
