# Instructions to run benchmarks by multiVectorFEOperators branch
1. Clone the multiVectorFEOperators branch
2. compile the branch (usual DFT-FE installation)
3. Copy contents of dftfe/scripts to dftfe/demo/ex1
4. Edit line 21 to path of dftfe real
5. Run batchJobSummit.sh from the same folder it is in. Submits both MatrixFree and CellMatrix runs as separate jobs

# Details of the scripts
The folder "dftfe/demo/ex1" is to be benchmarked, specifically parameterFile_a
"subsection Dispersion Correction" in parameterFile_a has all the parameters needed for benchmarking that need to be changed.

set D3 ATM = true -> sets if Matrixfree GPU is run
set D4 MBD = true -> sets if CellMatrix GPU is run

Only one true at a time is preferable to run as CellMatrix may run into memory issues. And easier to distinguish what the Nsight Compute Profiles if we know which function is called.

After linking with dealii9.4 and using cuda-aware-mpi, the branch multiVectorFEOperators is compiled and only real exe is tested for now.

The job is submitted using the batchJobSummit.sh script which in turn calls spectrumSummit.lsf to submit jobs.

batchJobSummit.sh has loop for submitting on number of nodes. So 1 2 3 submits on 1, 2 and 3 nodes. It edits the jobscript spectrumSummit.lsf. And another loop for FeOrders and another for both MatrixFree and CellMatrix

spectrumSummit.lsf runs from ex1 folder with all the parameters. The "dftfe/scripts" folder has all the required scripts. For Summit all the scripts required end with Summit. They need to be copied to ex1 folder and batchJobSummit.sh needs to run.

batchJobSummit.sh creates a "benchmark" folder to put all the ouput and then creates a parameter_a file according to each job. Now these parameter files can be edited without overlap by each job which the next for loop does. So, only the original parameter file needs to be edited for all FeOrders and tasks.

The loop edits parameter files and chooses FeOrders and correspondingly the domainVectors is chosen. The jsrun calls a "wrap.sh" script that profiles each mpi process and profiles only the region marked. The output is for each process determined by its rank. The path to dftfe exe needs to edited and
for MatrixFree the output is "poly"$1"_MF_nVec1024_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt"
for CellMatrix the output is "poly"$1"_CM_nVec1024_gpu"$OMPI_COMM_WORLD_SIZE"_rank"$OMPI_COMM_WORLD_RANK".txt"
