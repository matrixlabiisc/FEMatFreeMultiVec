#!/bin/bash

for i in 1 2 3 4 5                                              # No of Nodes
do
sed -i "5s/.*/#SBATCH -N $i/" slurmSelene.sh

for j in 6 7 8                                                  # FeOrder
do

for k in MF CM                                                  # Approach
do

mkdir -p benchmarks
/bin/cp -f parameterFile_a.prm parameterFile_"$k"_poly"$j"_N"$i".prm
sed -i "49s/.*/  set POLYNOMIAL ORDER ELECTROSTATICS = "$j"/" parameterFile_"$k"_poly"$j"_N"$i".prm
sed -i "7s/.*/  set DOMAIN VECTORS FILE = domainVectors_"$j".inp/" parameterFile_"$k"_poly"$j"_N"$i".prm

if [ "$k" == "MF" ];
then
    sed -i "30s/.*/    set D3 ATM                     = true       # matrixfree/" parameterFile_"$k"_poly"$j"_N"$i".prm
elif [ "$k" == "CM" ];
then
    sed -i "31s/.*/    set D4 MBD                     = true       # cellFlagGPU/" parameterFile_"$k"_poly"$j"_N"$i".prm
fi

sed -i "18s/.*/FeOrder=$j/" slurmSelene.sh
sed -i "19s/.*/Approach=$k/" slurmSelene.sh

sleep 1

sbatch slurmSelene.sh
sleep 2
done
done
done
