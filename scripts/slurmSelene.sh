#!/bin/bash
#SBATCH -A sa                   # account
#SBATCH -p luna                 # partition
#SBATCH -t 01:00:00             # wall time limit, hr:min:sec
#SBATCH -N 5
#SBATCH --ntasks-per-node=8
#SBATCH -J sa-dftfe:l1           # job name

cd $SLURM_SUBMIT_DIR
export MELLANOX_VISIBLE_DEVICES=all
export NVIDIA_VISIBLE_DEVICES=all
#export UCX_TLS=cma,cuda,cuda_copy,cuda_ipc,dc,dc_mlx5,dc_x,ib,mm,posix,rc,rc_mlx5,rc_v,rc_verbs,rc_x,self,shm,sm,sysv,tcp,ud,ud_mlx5,ud_v,ud_verbs,ud_x

CONT="gitlab-master.nvidia.com/prtiwari/my-dftfe-project/dftfe_final:v1.0"
echo "Running on hosts: $(echo $(scontrol show hostname))"
echo ${pwd}

FeOrder=8
Approach=CM

srun -n $SLURM_NTASKS --export=ALL,UCX_TLS --container-image="${CONT}" --container-mounts=/lustre/fsw/sa/prtiwari/dftfemount:/host_pwd,$SLURM_SUBMIT_DIR:/mnt1 --container-workdir /mnt1 /host_pwd/dft-fe/Buildfolder/build_multiVectorFEOperators/release/real/dftfe parameterFile_"$Approach"_poly"$FeOrder"_N"$SLURM_JOB_NUM_NODES".prm > benchmarks/vecMPI_"$Approach"_poly"$FeOrder"_gpu"$SLURM_JOB_NUM_NODES".txt
