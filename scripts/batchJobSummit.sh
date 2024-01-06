#!/bin/bash

sed -i "21s/.*/TaskperNode=1/" spectrumSummit.lsf

for i in 1 # 1 2 4 8 16                                      # No of Nodes
do
sed -i "5s/.*/#BSUB -nnodes $i/" spectrumSummit.lsf
sed -i "20s/.*/Nodes=$i/" spectrumSummit.lsf

for j in 6 7 8                                              # FeOrder
do

for k in MF CM                                               # Approach
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

sed -i "15s/.*/FeOrder=$j/" spectrumSummit.lsf
sed -i "17s/.*/Vec=$vec/" spectrumSummit.lsf
sed -i "18s/.*/Approach=$k/" spectrumSummit.lsf
sed -i "19s/.*/Mesh=$m/" spectrumSummit.lsf

sleep 1

bsub spectrumSummit.lsf
sleep 2

done
done
done
done
done
