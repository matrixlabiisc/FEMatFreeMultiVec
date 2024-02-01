#!/bin/bash

cd /nlsasfs/home/numerical/gourabp/matrixFree/run

TaskperNode=8

sed -i "3s/.*/#SBATCH --ntasks-per-node=$TaskperNode/" slurmSiddhi.sh
sed -i "6s/.*/#SBATCH --gres=gpu:A100-SXM4:$TaskperNode/" slurmSiddhi.sh
sed -i "31s/.*/TaskperNode=$TaskperNode/" slurmSiddhi.sh

for i in 1 2 #4 8 16                                              # No of Nodes
do
sed -i "2s/.*/#SBATCH --nodes=$i/" slurmSiddhi.sh

for j in 7 #6 7 8                                                   # FeOrder
do

for k in CM MF                                                  # Approach
do

for m in Uniform #Hanging                           # Mesh
do

for vec in 1024 #4 8 16 32 64 128 #256 512 1024
do

mkdir -p benchmarks
/bin/cp -f parameterFile_"$m".prm parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "48s/.*/  set POLYNOMIAL ORDER = "$j"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "49s/.*/  set POLYNOMIAL ORDER ELECTROSTATICS = "$j"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "7s/.*/  set DOMAIN VECTORS FILE = domainVectors_Uniform_"$j".inp/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "34s/.*/    set CHEBY WFC BLOCK SIZE                                 = "$vec"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "35s/.*/    set WFC BLOCK SIZE                                       = "$vec"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
if [ "$k" == "MF" ];
then
    sed -i "22s/.*/    set D3 ATM                     = true       # matrixfree/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
elif [ "$k" == "CM" ];
then
    sed -i "23s/.*/    set D4 MBD                     = true       # cellFlagGPU/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
fi

sed -i "25s/.*/FeOrder=$j/" slurmSiddhi.sh
sed -i "27s/.*/Vec=$vec/" slurmSiddhi.sh
sed -i "28s/.*/Approach=$k/" slurmSiddhi.sh
sed -i "29s/.*/Mesh=$m/" slurmSiddhi.sh
sed -i "30s/.*/Nodes=$i/" slurmSiddhi.sh

sleep 1

sbatch slurmSiddhi.sh
sleep 2
done
done
done
done
done
