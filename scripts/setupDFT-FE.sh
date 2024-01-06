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
#Paths for external libraries

mdiPath=""
dealiiDir="/shared/hpcx_dftfesoftwares/dealii94/installGPUV100"       # For V100
# dealiiDir="/shared/hpcx_dftfesoftwares/dealii94/installGPUA100"     # For A100
# dealiiDir="/shared/hpcx_dftfesoftwares/dealii94/installCPU"         # For cpu only, with vectorization
# dealiiDir="/shared/hpcx_dftfesoftwares/dealii94/installspack"
alglibDir="/shared/hpcx_dftfesoftwares/alglib/src"
libxcDir="/shared/hpcx_dftfesoftwares/libxc/install"
spglibDir="/shared/hpcx_dftfesoftwares/spglib/install"
xmlIncludeDir="/shared/spack/opt/spack/linux-centos7-cascadelake/gcc-11.3.0/libxml2-2.9.13-ldx2pcvgcrqkorjuokqu2jmsdtatbhd2/include/libxml2"
xmlLibDir="/shared/spack/opt/spack/linux-centos7-cascadelake/gcc-11.3.0/libxml2-2.9.13-ldx2pcvgcrqkorjuokqu2jmsdtatbhd2/lib/"
ELPA_PATH="/shared/hpcx_dftfesoftwares/elpa/install2022GPUV100"      # For V100
# ELPA_PATH="/shared/hpcx_dftfesoftwares/elpa/install2022GPUA100"      # For A100
# ELPA_PATH="/shared/hpcx_dftfesoftwares/elpa/install2022CPU"          # For cpu only, with vectorization
# ELPA_PATH="/shared/hpcx_dftfesoftwares/elpa/install2022spackmkl2022"
NCCL_PATH=""
numdiffdir="/shared/dftfesoftwares2021/numdiff/build"
dftdpath="/shared/hpcx_dftfesoftwares/dftd/install"

#Toggle GPU compilation
withGPU=ON
gpuLang="cuda"     # Use "cuda"/"hip"
gpuVendor="nvidia" # Use "nvidia/amd"
withGPUAwareMPI=OFF #Please use this option with care
                   #Only use if the machine supports
                   #device aware MPI and is profiled
                   #to be fast
#Option to link to NCCL library (Only for GPU compilation)
withNCCL=OFF
withMDI=OFF
#Compiler options and flags
cxx_compiler=mpicxx  #sets DCMAKE_CXX_COMPILER
cxxstd=17
cxx_flags="-g -fPIC -Wno-deprecated-declarations -Wno-non-template-friend" #sets DCMAKE_CXX_FLAGS
cxx_flagsRelease="-funroll-loops -ftree-vectorize" #sets DCMAKE_CXX_FLAGS_RELEASE
device_flags="-g -arch=sm_70 -forward-unknown-to-host-compiler -Wno-deprecated-declarations -Wno-non-template-friend -funroll-loops -ftree-vectorize -ccbin=$cxx_compiler"                                     # set DCMAKE_CXX_CUDA/HIP_FLAGS
                                                          #(only applicable for withGPU=ON)
device_architectures="70"                                 # set DCMAKE_CXX_CUDA/HIP_ARCHITECTURES
                                                          #(only applicable for withGPU=ON)
#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF
# build type: "Release" or "Debug"
build_type=Release
testing=OFF
minimal_compile=OFF
###########################################################################
#Usually, no changes are needed below this line
#
#if [[ x"$build_type" == x"Release" ]]; then
#  c_flags="$c_flagsRelease"
#  cxx_flags="$c_flagsRelease"
#else
#fi
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`
function cmake_real() {
  mkdir -p real && cd real
  if [ "$gpuLang" = "cuda" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  elif [ "$gpuLang" = "hip" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=OFF -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  else
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=OFF\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  fi
}
function cmake_cplx() {
  mkdir -p complex && cd complex
  if [ "$gpuLang" = "cuda" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  elif [ "$gpuLang" = "hip" ]; then
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=ON -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  else
    cmake -G Ninja -DCMAKE_CXX_STANDARD=$cxxstd -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath \
    -DWITH_NCCL=$withNCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$NCCL_PATH;$numdiffdir;$dftdpath"\
    -DWITH_COMPLEX=ON \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
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
echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
cmake_real "$SRC" && ninja
cd ..
# echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
# cmake_cplx "$SRC" && ninja -v
# cd ..
echo -e "${Blu}Build complete.${RCol}"
