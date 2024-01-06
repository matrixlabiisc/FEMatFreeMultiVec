#!/bin/sh

cd /home/gourab/matrixFree/build/

Nodes=1

for i in 1                                                     # Tasks per Node
do
sed -i "3s/.*/#SBATCH --ntasks-per-node=$i/" slurmMatrix.sh
sed -i "4s/.*/#SBATCH --nodes=$Nodes/" slurmMatrix.sh
sed -i "8s/.*/#SBATCH --gres=gpu:$i/" slurmMatrix.sh

tasks=$((Nodes*i))

for j in 6 7 8                                                    # FeOrder
do

for k in MF CM                                                  # Approach
do

for m in Uniform #Hanging                           # Mesh
do

for vec in 8 16 32 64 128 256 512 1024
do

mkdir -p benchmarks
/bin/cp -f parameterFile_"$m".prm parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm

sed -i "48s/.*/  set POLYNOMIAL ORDER = "$j"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "34s/.*/    set CHEBY WFC BLOCK SIZE                                 = "$vec"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "35s/.*/    set WFC BLOCK SIZE                                       = "$vec"/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
sed -i "7s/.*/   set DOMAIN VECTORS FILE     = domainVectors_"$m"_"$j".inp/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm

if [ "$k" == "MF" ];
then
    sed -i "22s/.*/    set D3 ATM                     = true       # matrixfree/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
elif [ "$k" == "CM" ];
then
    sed -i "23s/.*/    set D4 MBD                     = true       # cellFlagGPU/" parameterFile_"$m"_"$k"_poly"$j"_vec"$vec"_N"$i".prm
fi

sed -i "29s/.*/FeOrder=$j/" slurmMatrix.sh
sed -i "31s/.*/Vec=$vec/" slurmMatrix.sh
sed -i "32s/.*/Approach=$k/" slurmMatrix.sh
sed -i "33s/.*/Mesh=$m/" slurmMatrix.sh

sleep 1

sbatch slurmMatrix.sh

sleep 2
done
done
done
done
done
