#!/bin/bash
# script to setup and build DFT-FE.

set -e
set -o pipefail

if [ -s CMakeLists.txt ]; then
    echo "This script must be run from the build directory!"
    exit 1
fi

# Path to project source
SRC=`dirname $0` # location of source directory

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for required external libraries
dealiiDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/dealii/dealii/installspack"
alglibDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/alglib-3.20.0-deiq5fszrixaq3g43duxmeq4qiuhsixm"
libxcDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/libxc-6.1.0-h47gfrzoedvo7b6jsmvhmnfqmfixmxcm"
spglibDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/spglib-2.0.2-p4o7u6vcdz63ny6paw6jbx4libss434s"
xmlIncludeDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/libxml2-2.10.3-4bjhnxbs7qcx7vvntcml7z3mordvxpi6/include/libxml2"
xmlLibDir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/libxml2-2.10.3-4bjhnxbs7qcx7vvntcml7z3mordvxpi6/lib"
ELPA_PATH="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/elpa/installspack"
# ELPA_PATH="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/elpa-2022.11.001-2n3nzupbcv5ejmt65fckk4dv3suteobl"
# dftdpath="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/dftd/install"
# numdiffdir="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/numdiff-5.9.0-frza6cdkjn2atzvz5rbl3uufmjnzmj32"

#Paths for optional external libraries
# path for NCCL/RCCL libraries
DCCL_PATH="/lustre/fsw/sa/prtiwari/iisc/cds/phanim/spackinstall/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-11.3.0/nccl-2.16.2-1-mczsoftguy7ijuzatxaigcy5afmi6a5d"
mdiPath=""

#Toggle GPU compilation
withGPU=ON
gpuLang="cuda"     # Use "cuda"/"hip"
gpuVendor="nvidia" # Use "nvidia/amd"
withGPUAwareMPI=ON #Please use this option with care
                   #Only use if the machine supports
                   #device aware MPI and is profiled
                   #to be fast

#Option to link to DCCL library (Only for GPU compilation)
withDCCL=ON
withMDI=OFF
# withTorch=OFF
# withCustomizedDealii=ON

#Compiler options and flags
cxx_compiler=mpicxx  #sets DCMAKE_CXX_COMPILER
cxx_flags="-fPIC" #sets DCMAKE_CXX_FLAGS
cxx_flagsRelease="-O3 -march=native" #sets DCMAKE_CXX_FLAGS_RELEASE
device_flags="-arch=sm_80 -O3 -ccbin=$cxx_compiler" # set DCMAKE_CXX_CUDA/HIP_FLAGS
                           #(only applicable for withGPU=ON)
device_architectures="80" # set DCMAKE_CXX_CUDA/HIP_ARCHITECTURES
                           #(only applicable for withGPU=ON)


#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF

# build type: "Release" or "Debug"
build_type=Release

testing=OFF
minimal_compile=ON
###########################################################################
#Usually, no changes are needed below this line
#

#if [[ x"$build_type" == x"Release" ]]; then
#  c_flags="$c_flagsRelease"
#  cxx_flags="$c_flagsRelease"
#else
#fi
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`

function cmake_configure() {
  if [ "$gpuLang" = "cuda" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir \
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH" \
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures" \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  elif [ "$gpuLang" = "hip" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir \
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH" \
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES="$device_architectures" \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  else
    cmake -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler \
    -DCMAKE_CXX_FLAGS="$cxx_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir \
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH" \
    -DWITH_COMPLEX=$withComplex \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile \
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  fi
}

RCol='\e[0m'
Blu='\e[0;34m';
if [ -d "$out" ]; then # build directory exists
    echo -e "${Blu}$out directory already present${RCol}"
else
    rm -rf "$out"
    echo -e "${Blu}Creating $out ${RCol}"
    mkdir -p "$out"
fi

cd $out

withComplex=OFF
echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
mkdir -p real && cd real
cmake_configure "$SRC" && ninja
cd ..

withComplex=ON
echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
mkdir -p complex && cd complex
cmake_configure "$SRC" && ninja
cd ..

echo -e "${Blu}Build complete.${RCol}"
