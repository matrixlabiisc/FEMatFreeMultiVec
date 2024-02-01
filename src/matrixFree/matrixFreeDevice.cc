#include <matrixFreeDevice.h>
#include <constants.h>
#include <vectorUtilities.h>
// #include <immintrin.h>

namespace dftfe
{
  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::matrixFree(
    const MPI_Comm &   mpi_comm,
    const unsigned int blocksize)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_ndofsPerCell(d_ndofsPerDim * d_ndofsPerDim * d_ndofsPerDim)
    , d_nQuadPointsPerCell(d_nQuadPointsPerDim * d_nQuadPointsPerDim *
                           d_nQuadPointsPerDim)
    , d_nBatch(ceil((double)blocksize / (double)d_batchsize))
    , d_blocksize(blocksize)
  {}


  __device__ inline unsigned int
  getMultiVectorIdx(const unsigned int node,
                    const unsigned int batch,
                    const unsigned int nLocalDofs,
                    const unsigned int nGhostDofs,
                    const unsigned int *__restrict__ ghostMap)
  {
    return (node < nLocalDofs ?
              (node + batch * nLocalDofs) :
              (ghostMap[node - nLocalDofs + batch * nGhostDofs]));
  }


  template <typename Type, unsigned int vecShared, unsigned int p>
  __global__ void
  distributeKernel(Type *__restrict__ x,
                   const unsigned int *__restrict__ masterNodeBuckets,
                   const unsigned int *__restrict__ masterNodeOffset,
                   const unsigned int *__restrict__ slaveNodeBuckets,
                   const unsigned int *__restrict__ slaveNodeOffset,
                   const Type *__restrict__ weightMatrixList,
                   const unsigned int *__restrict__ weightMatrixOffset,
                   const Type *__restrict__ inhomogenityList,
                   const unsigned int *__restrict__ ghostMap,
                   const unsigned int nLocalDofs,
                   const unsigned int nGhostDofs)
  {
    __shared__ Type sharedMaster[vecShared * p * p];

    unsigned int masterBucketStart = masterNodeOffset[blockIdx.x];
    unsigned int masterBucketSize =
      masterNodeOffset[blockIdx.x + 1] - masterNodeOffset[blockIdx.x];

    for (unsigned int k = threadIdx.y; k < masterBucketSize; k += blockDim.y)
      {
        unsigned int idx =
          getMultiVectorIdx(masterNodeBuckets[k + masterBucketStart],
                            blockIdx.y,
                            nLocalDofs,
                            nGhostDofs,
                            ghostMap);

        sharedMaster[threadIdx.x + k * vecShared] =
          x[threadIdx.x + idx * vecShared];
      }

    __syncthreads();

    unsigned int slaveBucketStart = slaveNodeOffset[blockIdx.x];
    unsigned int slaveBucketSize =
      slaveNodeOffset[blockIdx.x + 1] - slaveNodeOffset[blockIdx.x];
    unsigned int weightMatrixStart = weightMatrixOffset[blockIdx.x];

    Type inhomogenity = inhomogenityList[blockIdx.x];

    for (unsigned int j = threadIdx.y; j < slaveBucketSize; j += blockDim.y)
      {
        Type tmp = inhomogenity;

        for (unsigned int k = 0; k < masterBucketSize; k++)
          tmp +=
            weightMatrixList[k + j * masterBucketSize + weightMatrixStart] *
            sharedMaster[threadIdx.x + k * vecShared];

        unsigned int idx =
          getMultiVectorIdx(slaveNodeBuckets[j + slaveBucketStart],
                            blockIdx.y,
                            nLocalDofs,
                            nGhostDofs,
                            ghostMap);

        x[threadIdx.x + idx * vecShared] = tmp;
      }
  }


  template <typename Type, unsigned int vecShared, unsigned int p>
  __global__ void
  distributeSlaveToMasterKernel(
    Type *__restrict__ Ax,
    Type *__restrict__ x,
    const unsigned int *__restrict__ masterNodeBuckets,
    const unsigned int *__restrict__ masterNodeOffset,
    const unsigned int *__restrict__ slaveNodeBuckets,
    const unsigned int *__restrict__ slaveNodeOffset,
    const Type *__restrict__ weightMatrixList,
    const unsigned int *__restrict__ weightMatrixOffset,
    const unsigned int *__restrict__ ghostMap,
    const unsigned int nLocalDofs,
    const unsigned int nGhostDofs)
  {
    __shared__ Type sharedSlave[vecShared * 4 * p * p];

    unsigned int masterBucketStart = masterNodeOffset[blockIdx.x];
    unsigned int masterBucketSize =
      masterNodeOffset[blockIdx.x + 1] - masterNodeOffset[blockIdx.x];

    unsigned int slaveBucketStart = slaveNodeOffset[blockIdx.x];
    unsigned int slaveBucketSize =
      slaveNodeOffset[blockIdx.x + 1] - slaveNodeOffset[blockIdx.x];

    if (masterBucketSize > 0)
      {
        for (unsigned int k = threadIdx.y; k < slaveBucketSize; k += blockDim.y)
          {
            unsigned int idx =
              getMultiVectorIdx(slaveNodeBuckets[k + slaveBucketStart],
                                blockIdx.y,
                                nLocalDofs,
                                nGhostDofs,
                                ghostMap);

            sharedSlave[threadIdx.x + k * vecShared] =
              Ax[threadIdx.x + idx * vecShared];

            Ax[threadIdx.x + idx * vecShared] = 0.;
            // x[threadIdx.x + idx * vecShared]  = 0.;
          }

        __syncthreads();

        unsigned int weightMatrixStart = weightMatrixOffset[blockIdx.x];

        for (unsigned int j = threadIdx.y; j < masterBucketSize;
             j += blockDim.y)
          {
            Type tmp = 0.;

            for (unsigned int k = 0; k < slaveBucketSize; k++)
              tmp +=
                weightMatrixList[j + k * masterBucketSize + weightMatrixStart] *
                sharedSlave[threadIdx.x + k * vecShared];

            unsigned int idx =
              getMultiVectorIdx(masterNodeBuckets[j + masterBucketStart],
                                blockIdx.y,
                                nLocalDofs,
                                nGhostDofs,
                                ghostMap);

            atomicAdd(&Ax[threadIdx.x + idx * vecShared], tmp);
          }
      }
    else
      {
        for (unsigned int k = threadIdx.y; k < slaveBucketSize; k += blockDim.y)
          {
            unsigned int idx =
              getMultiVectorIdx(slaveNodeBuckets[k + slaveBucketStart],
                                blockIdx.y,
                                nLocalDofs,
                                nGhostDofs,
                                ghostMap);

            Ax[threadIdx.x + idx * vecShared] = 0.;
            // x[threadIdx.x + idx * vecShared]  = 0.;
          }
      }
  }


 constexpr int maxPConstMem = 10;

  __constant__ double constShape[maxPConstMem * maxPConstMem * 4];

  template <typename Type, int p, int q, int dim, int batchSizeMF>
  __global__ void
  computeAXKernel(Type *__restrict__ V,
                  const Type *__restrict__ U,
                  const Type *__restrict__ Veff,
                  const Type *__restrict__ J,
                  const unsigned int *__restrict__ map,
                  const unsigned int *__restrict__ ghostMap,
                  const unsigned int nLocalDofs,
                  const unsigned int nGhostDofs)
  {
    // V = AU
    // gridDim.x = cells;
    // gridDim.y = batch;
    // nVec = batchSizeMF * batch;
    // batchSizeMF -> No of vectors in shared memory
    // First index is the fastest (Order -> x, y, z)
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    constexpr int pad = 0;

    Type *__restrict__ sharedX = SMem;
    Type *__restrict__ sharedY = &sharedX[batchSizeMF * q * q * q + pad];

    Type *__restrict__ constP  = &constShape[0];
    Type *__restrict__ constD  = &constP[q * p];
    Type *__restrict__ constPT = &constD[q * q];
    Type *__restrict__ constDT = &constPT[p * q];

    Type regX[q], regY[q], regZ[q];

    // New Layout - New Map
    // const unsigned int mapOffset = blockIdx.x * p * p * p;

    // New Layout - Base Map
    const unsigned int mapOffset =
      (blockIdx.x + blockIdx.y * gridDim.x) * p * p * p;

    //////////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // X -> Uxyz.Pz.Py.Px

    // 1st GEMM of P
    // Z Direction
    for (int i = threadIdx.y; i < p * p; i += blockDim.y)
      {
        memset(regX, 0, q * sizeof(Type));

        for (int k = 0; k < p; k++)
          {
            // New Layout - New Map
            // unsigned int dof = __ldg(&map[i + k * p*p + mapOffset]);

            // unsigned int idx = getMultiVectorIdx(
            //   dof, blockIdx.y, nLocalDofs, nGhostDofs, ghostMap);

            // u[k] = U[threadIdx.x + idx * batchSizeMF];

            // New Layout - Base Map
            unsigned int dof = __ldg(&map[i + k * p * p + mapOffset]);
            regY[k]          = U[threadIdx.x + dof];

            // Old Layout
            // unsigned int dof = __ldg(&map[i + k * p*p + blockIdx.x * p*p *
            // K]); u[k] = U[threadIdx.x + blockIdx.y * batchSizeMF + dof *
            // gridDim.y];

#pragma unroll
            for (int j = 0; j < q; j++)
              regX[j] += constP[j + k * q] * regY[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedX[threadIdx.x + i * batchSizeMF + j * batchSizeMF * p * p] =
            regX[j];
      }

    __syncthreads();

    // 2nd GEMM of P
    // Y Direction
    for (int i = threadIdx.y; i < p * q; i += blockDim.y)
      {
        int a = i % p;
        int b = i / p;

        memset(regY, 0, q * sizeof(Type));

        for (int k = 0; k < p; k++)
          {
            regX[k] = sharedX[threadIdx.x + a * batchSizeMF +
                              k * batchSizeMF * p + b * batchSizeMF * p * p];

#pragma unroll
            for (int j = 0; j < q; j++)
              regY[j] += constP[j + k * q] * regX[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedY[threadIdx.x + a * batchSizeMF + j * batchSizeMF * p +
                  b * batchSizeMF * p * q] = regY[j];
      }

    __syncthreads();

    // 3rd GEMM of P
    // X Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        memset(regX, 0, q * sizeof(Type));

        for (int k = 0; k < p; k++)
          {
            regY[k] =
              sharedY[threadIdx.x + k * batchSizeMF + i * batchSizeMF * p];

#pragma unroll
            for (int j = 0; j < q; j++)
              regX[j] += constP[j + k * q] * regY[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedX[threadIdx.x + j * batchSizeMF + i * batchSizeMF * q] =
            regX[j];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Grad operation in each direction
    // regZ    -> X.Dz
    // sharedY -> X.Dy
    // sharedX -> X.Dx

    // 1st GEMM of D
    // Z Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        memset(regZ, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regY[k] =
              sharedX[threadIdx.x + i * batchSizeMF + k * batchSizeMF * q * q];

#pragma unroll
            for (int j = 0; j < q; j++)
              regZ[j] += constD[j + k * q] * regY[k];
          }
      }

    // 2nd GEMM of D
    // Y Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type regT[q];

        int a = i % q;
        int b = i / q;

        memset(regT, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            Type tempX = sharedX[threadIdx.x + a * batchSizeMF +
                                 k * batchSizeMF * q + b * batchSizeMF * q * q];

#pragma unroll
            for (int j = 0; j < q; j++)
              regT[j] += constD[j + k * q] * tempX;
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedY[threadIdx.x + a * batchSizeMF + j * batchSizeMF * q +
                  b * batchSizeMF * q * q] = regT[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type regT[q];

        memset(regX, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regT[k] =
              sharedX[threadIdx.x + k * batchSizeMF + i * batchSizeMF * q];

#pragma unroll
            for (int j = 0; j < q; j++)
              regX[j] += constD[j + k * q] * regT[k];
          }
      }

    __syncthreads();

    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
#pragma unroll
        for (int j = 0; j < q; j++)
          sharedX[threadIdx.x + j * batchSizeMF + i * batchSizeMF * q] =
            regX[j];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Gemm with Jacobian Action
    // sharedX, sharedY, regZ have the respective gemms of X, Y, Z
    // directions
    // J.[X Y Z]

    Type detJ;

    // #pragma unroll
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type v[3];

        int jOffset = blockIdx.x * dim * dim;

        for (int j = 0; j < q; j++)
          {
            v[0] = 0.5 * sharedX[threadIdx.x + (i + j * q * q) * batchSizeMF];
            v[1] = 0.5 * sharedY[threadIdx.x + (i + j * q * q) * batchSizeMF];
            v[2] = 0.5 * regZ[j];

            regX[j] = J[0 + jOffset] * v[0] + J[1 + jOffset] * v[1] +
                      J[2 + jOffset] * v[2];
            sharedY[threadIdx.x + (i + j * q * q) * batchSizeMF] =
              J[3 + jOffset] * v[0] + J[4 + jOffset] * v[1] +
              J[5 + jOffset] * v[2];
            regZ[j] = J[6 + jOffset] * v[0] + J[7 + jOffset] * v[1] +
                      J[8 + jOffset] * v[2];

            detJ = J[0 + jOffset] * (J[4 + jOffset] * J[8 + jOffset] -
                                     J[5 + jOffset] * J[7 + jOffset]) -
                   J[1 + jOffset] * (J[3 + jOffset] * J[8 + jOffset] -
                                     J[5 + jOffset] * J[6 + jOffset]) +
                   J[2 + jOffset] * (J[3 + jOffset] * J[7 + jOffset] -
                                     J[4 + jOffset] * J[6 + jOffset]);
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Integration
    // X -> detJ.Veff.Uxyz.Pz.Py.Px
    // X -> regZ.DTz + X
    // X -> regY.DTy + X
    // X -> regX.DTx + X

    // 1st GEMM of DT
    // Z Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type regT[q];

        memset(regT, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
#pragma unroll
            for (int j = 0; j < q; j++)
              regT[j] += constDT[j + k * q] * regZ[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          {
            sharedX[threadIdx.x + i * batchSizeMF + j * batchSizeMF * q * q] =
              Veff[i + j * q * q + blockIdx.x * q * q * q] * detJ * regY[j] +
              regT[j];
          }
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type regT[q];

        int a = i % q;
        int b = i / q;

        memset(regT, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regY[k] = sharedY[threadIdx.x + a * batchSizeMF +
                              k * batchSizeMF * q + b * batchSizeMF * q * q];

#pragma unroll
            for (int j = 0; j < q; j++)
              regT[j] += constDT[j + k * q] * regY[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedX[threadIdx.x + a * batchSizeMF + j * batchSizeMF * q +
                  b * batchSizeMF * q * q] += regT[j];
      }

    __syncthreads();

    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
#pragma unroll
        for (int j = 0; j < q; j++)
          sharedY[threadIdx.x + i * batchSizeMF + j * batchSizeMF * q * q] =
            regX[j];
      }

    __syncthreads();

    // 3rd GEMM of DT
    // X Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        Type regT[q];

        memset(regT, 0, q * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regX[k] =
              sharedY[threadIdx.x + k * batchSizeMF + i * batchSizeMF * q];

#pragma unroll
            for (int j = 0; j < q; j++)
              regT[j] += constDT[j + k * q] * regX[k];
          }

#pragma unroll
        for (int j = 0; j < q; j++)
          sharedX[threadIdx.x + j * batchSizeMF + i * batchSizeMF * q] +=
            regT[j];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Integration combined with Assembly
    // X -> X.PTz.PTy.PTx

    // 1st GEMM of PT
    // Z Direction
    for (int i = threadIdx.y; i < q * q; i += blockDim.y)
      {
        memset(regY, 0, p * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regX[k] =
              sharedX[threadIdx.x + i * batchSizeMF + k * batchSizeMF * q * q];

#pragma unroll
            for (int j = 0; j < p; j++)
              regY[j] += constPT[j + k * p] * regX[k];
          }

#pragma unroll
        for (int j = 0; j < p; j++)
          sharedY[threadIdx.x + i * batchSizeMF + j * batchSizeMF * q * q] =
            regY[j];
      }

    __syncthreads();

    // 2nd GEMM of PT
    // Y Direction
    for (int i = threadIdx.y; i < q * p; i += blockDim.y)
      {
        int a = i % q;
        int b = i / q;

        memset(regX, 0, p * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regY[k] = sharedY[threadIdx.x + a * batchSizeMF +
                              k * batchSizeMF * q + b * batchSizeMF * q * q];

#pragma unroll
            for (int j = 0; j < p; j++)
              regX[j] += constPT[j + k * p] * regY[k];
          }

#pragma unroll
        for (int j = 0; j < p; j++)
          sharedX[threadIdx.x + a * batchSizeMF + j * batchSizeMF * q +
                  b * batchSizeMF * q * p] = regX[j];
      }

    __syncthreads();

    // 3rd GEMM of PT
    // X Direction
    for (int i = threadIdx.y; i < p * p; i += blockDim.y)
      {
        memset(regY, 0, p * sizeof(Type));

        for (int k = 0; k < q; k++)
          {
            regX[k] =
              sharedX[threadIdx.x + k * batchSizeMF + i * batchSizeMF * q];

#pragma unroll
            for (int j = 0; j < p; j++)
              regY[j] += constPT[j + k * p] * regX[k];
          }

#pragma unroll
        for (int j = 0; j < p; j++)
          {
            // New Layout - New Map
            // unsigned int dof = __ldg(&map[j + i * p + mapOffset]);

            // unsigned int idx = getMultiVectorIdx(
            //   dof, blockIdx.y, nLocalDofs, nGhostDofs, ghostMap);

            // atomicAdd(&V[threadIdx.x + idx * batchSizeMF], y[j]);

            // New Layout - Base Map
            int dof = __ldg(&map[j + i * p + mapOffset]);
            atomicAdd(&V[threadIdx.x + dof], regY[j]);

            // Old Layout
            // unsigned int dof = __ldg(&map[j + i * p + blockIdx.x * M * K]);
            // atomicAdd(&V[threadIdx.x + blockIdx.y * batchSizeMF + dof *
            // gridDim.y], y[j]);
          }
      } //*/
  }

  template <const unsigned int p, const unsigned int vecShared>
  __device__ void
  SGemm_NT(double *shared_P, double *shared_U, double *shared_V)
  {
#pragma unroll
    for (unsigned int n = threadIdx.x; n < vecShared; n += blockDim.x)
      {
#pragma unroll
        for (unsigned int i = threadIdx.z; i < p; i += blockDim.z)
          {
            double local_P[p];
            for (unsigned int k = 0; k < p; k++)
              {
                local_P[k] = shared_P[k * p + i];
              }
#pragma unroll
            for (unsigned int j = threadIdx.y; j < p * p; j += blockDim.y)
              {
                double temp;
                temp = 0.0;

                for (unsigned int k = 0; k < p; k++)
                  {
                    temp +=
                      (local_P[k] * shared_U[(k * p * p + j) * vecShared + n]);
                  }
                shared_V[(j * p + i) * vecShared + n] = temp;
              }
          }
      }
    __syncthreads();
  }


  template <const unsigned int p, const unsigned int vecShared>
  __device__ void
  SGemm_TT(double *shared_P, double *shared_U, double *shared_V)
  {
#pragma unroll
    for (unsigned int n = threadIdx.x; n < vecShared; n += blockDim.x)
      {
#pragma unroll
        for (unsigned int i = threadIdx.z; i < p; i += blockDim.z)
          {
            double local_P[p];
            for (unsigned int k = 0; k < p; k++)
              {
                local_P[k] = shared_P[i * p + k];
              }
#pragma unroll
            for (unsigned int j = threadIdx.y; j < p * p; j += blockDim.y)
              {
                double temp;
                temp = 0.0;

                for (unsigned int k = 0; k < p; k++)
                  {
                    temp +=
                      (local_P[k] * shared_U[(k * p * p + j) * vecShared + n]);
                  }
                shared_V[(j * p + i) * vecShared + n] = temp;
              }
          }
      }
    __syncthreads();
  }


  template <const unsigned int p, const unsigned int vecShared>
  __global__ void
  sharedFusedKernel(double *           Yloc, /*output*/
                    const double *     Xloc, /*input*/
                    double *           P,    /*N (pxp) matrix*/
                    double *           D,
                    const double *     Veff,
                    unsigned int *     map,
                    double *           J,
                    unsigned int       nVec,
                    const unsigned int cells,
                    const unsigned int blocks_per_cell,
                    const unsigned int nDofs)
  {
    extern __shared__ double SMem[];
    // double *shared_U = SMem; // p =7, p^3 = 343, Nm = 8, No. of bytes =
    // 8*343*8 double *shared_V = &shared_U[vecShared*p*p*p]; //[(Nm) * [p x p
    // *2]] double *shared_P = &shared_V[vecShared*p*p*p]; // size(P) = p*p [p x
    // p] // constant memory Layout

    double *sharedX = SMem;
    double *sharedY = &sharedX[vecShared * p * p * p];
    double *sharedZ = &sharedY[vecShared * p * p * p];
    double *sharedT = &sharedZ[vecShared * p * p * p];
    double *sharedP = &sharedT[vecShared * p * p * p];
    double *sharedD = &sharedP[p * p];
    double *sharedJ = &sharedD[p * p];

    // Block size : 3D: (Nm,8,8) // Grid size: 1D, (cells*BPC, 1, 1)
    for (unsigned int i = threadIdx.x + blockDim.x * threadIdx.y +
                          blockDim.x * blockDim.y * threadIdx.z;
         i < p * p;
         i += blockDim.x * blockDim.y * blockDim.z)
      {
        sharedP[i] = P[i]; // pxp; cell matrix: p^3xp^3
        sharedD[i] = D[i]; // pxp; cell matrix: p^3xp^3
      }

    __syncthreads();

#pragma unroll
    for (unsigned int c = blockIdx.x; c < cells * blocks_per_cell;
         c += gridDim.x)
      {
        unsigned int cell_index = c / blocks_per_cell;
        unsigned int batchID    = c % blocks_per_cell;

        // Loading in Shared Memory
        // EXTRACTION
        // vecShared, nVec are fastest index Shared_U = [vecShared][p^3]
        for (unsigned int i = threadIdx.x + blockDim.x * threadIdx.y +
                              blockDim.x * blockDim.y * threadIdx.z;
             i < vecShared * p * p * p;
             i += blockDim.x * blockDim.y * blockDim.z)
          {
            unsigned int local_index     = i / vecShared;
            unsigned int local_vec_index = i % vecShared;
            unsigned int index_on_U = map[local_index + cell_index * p * p * p];

            // CV Layout
            sharedX[i] =
              Xloc[local_vec_index + batchID * vecShared + index_on_U * nVec];
            //*/

            // BCV Layout
            /*sharedX[i] =
              Xloc[local_vec_index + index_on_U * vecShared + batchID * nDofs];
            //*/
          }

        __syncthreads();

        // GEMMS

        SGemm_NT<p, vecShared>(sharedP, sharedX, sharedY);
        SGemm_NT<p, vecShared>(sharedP, sharedY, sharedX);
        SGemm_NT<p, vecShared>(sharedP, sharedX, sharedY);

        SGemm_NT<p, vecShared>(sharedD, sharedY, sharedZ);
        SGemm_NT<p, vecShared>(sharedD, sharedY, sharedT);
        SGemm_NT<p, vecShared>(sharedD, sharedY, sharedX);

#pragma unroll
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 3 * 3;
             i += blockDim.x * blockDim.y)
          sharedJ[i] = J[i + blockIdx.x / blocks_per_cell * 3 * 3];

        __syncthreads();

        double detJ;

#pragma unroll
        for (int i = threadIdx.y; i < p * p * p; i += blockDim.y)
          {
            double v[3];

            v[2] = sharedX[threadIdx.x + i * vecShared];
            v[1] = sharedZ[threadIdx.x + i * vecShared];
            v[0] = sharedT[threadIdx.x + i * vecShared];

            sharedX[threadIdx.x + i * vecShared] =
              sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
            sharedZ[threadIdx.x + i * vecShared] =
              sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
            sharedT[threadIdx.x + i * vecShared] =
              sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];

            detJ =
              sharedJ[0] * (sharedJ[4] * sharedJ[8] - sharedJ[5] * sharedJ[7]) -
              sharedJ[1] * (sharedJ[3] * sharedJ[8] - sharedJ[5] * sharedJ[6]) +
              sharedJ[2] * (sharedJ[3] * sharedJ[7] - sharedJ[4] * sharedJ[6]);
          }

        __syncthreads();

        for (unsigned int i = threadIdx.x + blockDim.x * threadIdx.y +
                              blockDim.x * blockDim.y * threadIdx.z;
             i < vecShared * p * p * p;
             i += blockDim.x * blockDim.y * blockDim.z)
          {
            sharedY[i] += sharedX[i] + sharedZ[i] + sharedT[i];
          }

        __syncthreads();

        for (unsigned int i = threadIdx.x + blockDim.x * threadIdx.y +
                              blockDim.x * blockDim.y * threadIdx.z;
             i < vecShared * p * p * p;
             i += blockDim.x * blockDim.y * blockDim.z)
          {
            unsigned int local_index = i / vecShared;
            unsigned int ix          = local_index % p;
            unsigned int iy          = ((unsigned int)local_index / p) % p;
            unsigned int iz          = (unsigned int)local_index / (p * p);

            sharedY[i] *= Veff[local_index + cell_index * p * p * p] * detJ *
                          sharedJ[ix] * sharedJ[iy] * sharedJ[iz];
          }

        __syncthreads();

        SGemm_TT<p, vecShared>(sharedD, sharedY, sharedX);
        SGemm_TT<p, vecShared>(sharedD, sharedX, sharedY);
        SGemm_TT<p, vecShared>(sharedD, sharedY, sharedX);

        SGemm_TT<p, vecShared>(sharedP, sharedY, sharedX);
        SGemm_TT<p, vecShared>(sharedP, sharedX, sharedY);
        SGemm_TT<p, vecShared>(sharedP, sharedY, sharedX);

        // ASSEMBLY
        for (unsigned int i = threadIdx.x + blockDim.x * threadIdx.y +
                              blockDim.x * blockDim.y * threadIdx.z;
             i < vecShared * p * p * p;
             i += blockDim.x * blockDim.y * blockDim.z)
          {
            unsigned int local_index     = i / vecShared;
            unsigned int local_vec_index = i % vecShared;
            unsigned int index_on_U = map[local_index + cell_index * p * p * p];

            // CV Layout
            atomicAdd(
              &Yloc[local_vec_index + batchID * vecShared + index_on_U * nVec],
              sharedX[i]); //*/

            // BCV Layout
            /*atomicAdd(
              &Yloc[local_vec_index + index_on_U * vecShared + batchID * nDofs],
              sharedX[i]); //*/
          }
      }
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::reinit(
    const dealii::MatrixFree<3, double> *    matrixFreeDataPtr,
    const dealii::AffineConstraints<double> *constraintMatrixPtr,
    const distributedCPUVec<double> &        sqrtMassVec,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int                       matrixFreeQuadratureComponentAX)
  {
    MPI_Barrier(mpi_communicator);

    d_matrixFreeDataPtr               = matrixFreeDataPtr;
    d_constraintMatrixPtr             = constraintMatrixPtr;
    d_matrixFreeVectorComponent       = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentAX = matrixFreeQuadratureComponentAX;
    d_nLocalCells                     = d_matrixFreeDataPtr->n_physical_cells();
    d_nLocalDofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent)
        ->locally_owned_size();
    d_nGhostDofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent)
        ->n_ghost_indices();
    d_nRelaventDofs = d_nLocalDofs + d_nGhostDofs;

    vectorTools::createDealiiBatchedVector<double>(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      d_batchsize,
      d_blocksize,
      d_cpuTestInputDealiiBatched);

    d_batchedPartitioner = d_cpuTestInputDealiiBatched.get_partitioner();

    constexpr unsigned int p       = d_ndofsPerDim;
    constexpr unsigned int q       = d_nQuadPointsPerDim;
    constexpr unsigned int dim     = 3;
    constexpr unsigned int qPoints = q * q * q;

    auto dofInfo =
      d_matrixFreeDataPtr->get_dof_info(d_matrixFreeVectorComponent);
    auto shapeInfo =
      d_matrixFreeDataPtr->get_shape_info(d_matrixFreeVectorComponent,
                                          d_matrixFreeQuadratureComponentAX);
    auto mappingData = d_matrixFreeDataPtr->get_mapping_info()
                         .cell_data[d_matrixFreeQuadratureComponentAX];
    auto shapeData = shapeInfo.get_shape_data();

    std::vector<double> spVG(2 * q * p + 2 * q * q);

    for (auto iDoF = 0; iDoF < p; iDoF++)
      {
        for (auto iQuad = 0; iQuad < q; iQuad++)
          {
            double value = shapeData.shape_values[iQuad + iDoF * q][0] *
                           std::sqrt(shapeData.quadrature.weight(iQuad));

            spVG[iQuad + iDoF * q]                 = value;
            spVG[iDoF + iQuad * p + q * p + q * q] = value;
          }
      }

    for (auto iQuad1 = 0; iQuad1 < q; iQuad1++)
      {
        for (auto iQuad2 = 0; iQuad2 < q; iQuad2++)
          {
            double grad =
              shapeData.shape_gradients_collocation[iQuad2 + iQuad1 * q][0] *
              std::sqrt(shapeData.quadrature.weight(iQuad2)) /
              std::sqrt(shapeData.quadrature.weight(iQuad1));

            spVG[iQuad2 + iQuad1 * q + q * p]             = grad;
            spVG[iQuad1 + iQuad2 * q + 2 * q * p + q * q] = grad;
          }
      }

    const unsigned int dofs_per_cell = p * p * p;

    // Single Vector Map
    thrust::host_vector<unsigned int> map(d_ndofsPerCell * d_nLocalCells);

    for (auto iCellBatch = 0, iCell = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         iCellBatch++)
      {
        for (auto iCellLocal = 0;
             iCellLocal < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
             iCellLocal++, iCell++)
          {
            std::memcpy(
              map.data() + iCell * d_ndofsPerCell,
              ((dofInfo.row_starts_plain_indices
                  [iCellBatch * dofInfo.vectorization_length + iCellLocal] ==
                dealii::numbers::invalid_unsigned_int)) ?
                dofInfo.dof_indices.data() +
                  dofInfo
                    .row_starts[iCellBatch * dofInfo.vectorization_length +
                                iCellLocal]
                    .first :
                dofInfo.plain_dof_indices.data() +
                  dofInfo.row_starts_plain_indices
                    [iCellBatch * dofInfo.vectorization_length + iCellLocal],
              d_ndofsPerCell * sizeof(unsigned int));
          }
      }

    // Device BCV Map
    thrust::host_vector<unsigned int> map_newlayout(dofs_per_cell *
                                                    d_nLocalCells * d_nBatch);

    auto         taskGhostMap = sqrtMassVec.get_partitioner()->ghost_targets();
    unsigned int n_procs =
      dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    std::vector<unsigned int> taskGhostStartIndices(n_procs, 0);

    for (unsigned int i = 0; i < taskGhostMap.size(); i++)
      taskGhostStartIndices[taskGhostMap[i].first] = taskGhostMap[i].second;

    unsigned int ghostSum = 0;

    for (unsigned int i = 0; i < taskGhostStartIndices.size(); i++)
      {
        unsigned int tmp = ghostSum;
        ghostSum += taskGhostStartIndices[i];
        taskGhostStartIndices[i] = tmp;
      }

    for (unsigned int iCell = 0; iCell < d_nLocalCells; iCell++)
      {
        for (unsigned int ildof = 0; ildof < dofs_per_cell; ildof++)
          {
            unsigned int l2g = map[ildof + dofs_per_cell * iCell];
            if (l2g >= d_nLocalDofs)
              {
                unsigned int ownerId = 0;
                while (taskGhostStartIndices[ownerId] <= l2g - d_nLocalDofs)
                  {
                    ++ownerId;
                    if (ownerId == n_procs)
                      break;
                  }

                --ownerId;
                unsigned int ghostIdFromOwner =
                  l2g - taskGhostStartIndices[ownerId] - d_nLocalDofs;
                unsigned int nGhostsFromOwner =
                  ownerId == n_procs - 1 ?
                    sqrtMassVec.get_partitioner()->n_ghost_indices() -
                      taskGhostStartIndices[ownerId] :
                    taskGhostStartIndices[ownerId + 1] -
                      taskGhostStartIndices[ownerId];

                for (unsigned int ibatch = 0; ibatch < d_nBatch; ibatch++)
                  {
                    map_newlayout[ildof + dofs_per_cell * iCell +
                                  ibatch * dofs_per_cell * d_nLocalCells] =
                      (d_nLocalDofs + taskGhostStartIndices[ownerId]) *
                        d_blocksize +
                      ghostIdFromOwner * d_batchsize +
                      ibatch * nGhostsFromOwner * d_batchsize;
                  }
              }
            else
              {
                for (unsigned int ibatch = 0; ibatch < d_nBatch; ibatch++)
                  map_newlayout[ildof + dofs_per_cell * iCell +
                                ibatch * dofs_per_cell * d_nLocalCells] =
                    l2g * d_batchsize + ibatch * d_nLocalDofs * d_batchsize;
              }
          }
      }


    thrust::host_vector<double> jacobianFactor(dim * dim * d_nLocalCells),
      detJacobian(d_nLocalCells);

    auto cellOffsets = mappingData.data_index_offsets;

    for (auto iCellBatch = 0, cellCount = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         ++iCellBatch)
      {
        for (auto iCell = 0;
             iCell < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
             ++iCell, ++cellCount)
          {
            for (auto d = 0; d < dim; d++)
              {
                for (auto e = 0; e < dim; e++)
                  {
                    for (auto f = 0; f < dim; f++)
                      {
                        jacobianFactor[e + d * dim + cellCount * dim * dim] +=
                          mappingData.jacobians[0][cellOffsets[iCellBatch]][d]
                                               [f][iCell] *
                          mappingData.jacobians[0][cellOffsets[iCellBatch]][e]
                                               [f][iCell] *
                          mappingData
                            .JxW_values[cellOffsets[iCellBatch]][iCell];
                        detJacobian[cellCount] =
                          mappingData
                            .JxW_values[cellOffsets[iCellBatch]][iCell];
                      }
                  }
              }
          }
      }

    thrust::host_vector<unsigned int> ghostMap(d_nGhostDofs * d_nBatch, 0);

    for (unsigned int ilocalDof = 0; ilocalDof < d_nRelaventDofs; ++ilocalDof)
      {
        if (ilocalDof >= d_nLocalDofs)
          {
            unsigned int ownerId = 0;
            while (taskGhostStartIndices[ownerId] <= ilocalDof - d_nLocalDofs)
              {
                ++ownerId;
                if (ownerId == n_procs)
                  break;
              }
            --ownerId;
            unsigned int ghostIdFromOwner =
              ilocalDof - taskGhostStartIndices[ownerId] - d_nLocalDofs;
            unsigned int nGhostsFromOwner =
              ownerId == n_procs - 1 ?
                d_nGhostDofs - taskGhostStartIndices[ownerId] :
                taskGhostStartIndices[ownerId + 1] -
                  taskGhostStartIndices[ownerId];

            for (unsigned int ibatch = 0; ibatch < d_nBatch; ibatch++)
              {
                ghostMap[ilocalDof - d_nLocalDofs + ibatch * d_nGhostDofs] =
                  (d_nLocalDofs + taskGhostStartIndices[ownerId]) * d_nBatch +
                  ghostIdFromOwner + ibatch * nGhostsFromOwner;
              }
          }
      }

    ghostMapDevice = ghostMap;

    // Construct the device vectors
    d_jacobianFactor = jacobianFactor;
    // d_jacobianFactor = detJacobian;

    d_map = map_newlayout;
    // d_map = map;

    // shapeF = spV;
    // shapeG = spG;

    // shapeFunctionValuePtr    = thrust::raw_pointer_cast(shapeF.data());
    // shapeFunctionGradientPtr = thrust::raw_pointer_cast(shapeG.data());

    d_mapPtr          = thrust::raw_pointer_cast(d_map.data());
    jacobianFactorPtr = thrust::raw_pointer_cast(d_jacobianFactor.data());

    const size_t smem = 2 * d_batchsize * d_nQuadPointsPerCell * sizeof(double);

    const size_t smem2 =
      (4 * d_batchsize * d_nQuadPointsPerCell + 2 * p * p + 3 * 3) *
      sizeof(double);

    // cudaFuncSetSharedMemConfig(
    //   computeAXKernel<double, p * p, q, p, dim, d_batchsize>,
    //   cudaSharedMemBankSizeEightByte);

    cudaFuncSetAttribute(computeAXKernel<double, p, q, dim, d_batchsize>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    // cudaFuncSetAttribute(sharedFusedKernel<p, d_batchsize>,
    //                      cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                      smem2);

    cudaMemcpyToSymbol(constShape,
                       spVG.data(),
                       spVG.size() * sizeof(double),
                       0,
                       cudaMemcpyHostToDevice);

    // Setup Constraints
    /*d_constraintsInfo.initialize(d_matrixFreeDataPtr->get_vector_partitioner(
                                   d_matrixFreeVectorComponent),
                                 *d_constraintMatrixPtr);

    d_constraintsInfo.precomputeMaps(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      d_batchsize,
      d_blocksize); //*/

    initializeOptimizedConstraints(sqrtMassVec);
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::
    initializeOptimizedConstraints(const distributedCPUVec<double> &sqrtMassVec)
  {
    // d_invSqrtElementalMassVector.resize(d_nLocalCells * d_ndofsPerCell);
    // sqrtMassVec.update_ghost_values();
    // for (auto iCell = 0; iCell < d_nLocalCells; i++Cell)
    //   {
    //     for (int iDoF = 0; iDoF < d_ndofsPerCell; i++DoF)
    //       {
    //         int l2g =
    //           singleVectorMap[iDoF + d_ndofsPerCell * iCell];
    //         if (d_constraintMatrixPtr->is_constrained(
    //               d_matrixFreeDataPtr
    //                 ->get_vector_partitioner(d_matrixFreeVectorComponent)
    //                 ->local_to_global(l2g)))
    //           {
    //             d_invSqrtElementalMassVector[iDoF + d_ndofsPerCell * iCell] =
    //               1.0;
    //           }
    //         else
    //           {
    //             d_invSqrtElementalMassVector[iDoF + d_ndofsPerCell * iCell] =
    //               1.0 * sqrtMassVec.local_element(l2g);
    //           }
    //         if (!(d_constraintMatrixPtr->is_constrained(
    //               d_matrixFreeDataPtr
    //                 ->get_vector_partitioner(d_matrixFreeVectorComponent)
    //                 ->local_to_global(l2g))))
    //           {
    //             if (std::abs(sqrtMassVec.local_element(l2g)) < 1.0e-15)
    //               std::cout
    //                 << "DEBUG "
    //                 << d_invSqrtElementalMassVector[iDoF +
    //                                                 d_ndofsPerCell * iCell]
    //                 << " " << sqrtMassVec.local_element(l2g) << " "
    //                 << d_constraintMatrixPtr->is_constrained(
    //                      d_matrixFreeDataPtr
    //                        ->get_vector_partitioner(d_matrixFreeVectorComponent)
    //                        ->local_to_global(l2g))
    //                 << " "
    //                 << d_matrixFreeDataPtr
    //                      ->get_vector_partitioner(d_matrixFreeVectorComponent)
    //                      ->local_to_global(l2g)
    //                 << " " << this_mpi_process << std::endl;
    //           }
    //       }
    //   }

    const dealii::IndexSet &locally_owned_dofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent)
        ->locally_owned_range();
    const dealii::IndexSet &ghost_dofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent)
        ->ghost_indices();

    std::vector<std::vector<unsigned int>> masterNodeBuckets, slaveNodeBuckets;
    std::vector<std::vector<double>>       weightMatrixList;
    std::vector<double>                    inhomogenityList;
    // std::vector<bool> isConstrained(d_nRelaventDofs, false);

    for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
         it != locally_owned_dofs.end();
         it++)
      {
        if (d_constraintMatrixPtr->is_constrained(*it))
          {
            const dealii::types::global_dof_index lineDof = *it;
            const std::vector<
              std::pair<dealii::types::global_dof_index, double>> *rowData =
              d_constraintMatrixPtr->get_constraint_entries(lineDof);
            bool isConstraintRhsExpandingOutOfIndexSet = false;

            for (unsigned int j = 0; j < rowData->size(); j++)
              {
                if (!(d_matrixFreeDataPtr
                        ->get_vector_partitioner(d_matrixFreeVectorComponent)
                        ->is_ghost_entry((*rowData)[j].first) ||
                      d_matrixFreeDataPtr
                        ->get_vector_partitioner(d_matrixFreeVectorComponent)
                        ->in_local_range((*rowData)[j].first)))
                  {
                    isConstraintRhsExpandingOutOfIndexSet = true;
                    break;
                  }
              }

            if (isConstraintRhsExpandingOutOfIndexSet)
              continue;

            // isConstrained[d_matrixFreeDataPtr
            //                 ->get_vector_partitioner(
            //                   d_matrixFreeVectorComponent)
            //                 ->global_to_local(lineDof)] = true;

            std::vector<unsigned int> masterData(rowData->size());
            std::vector<double>       weightData(rowData->size());
            double                    inhomogenity =
              d_constraintMatrixPtr->get_inhomogeneity(lineDof);

            for (auto i = 0; i < rowData->size(); i++)
              {
                masterData[i] =
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local((*rowData)[i].first);
                weightData[i] = (*rowData)[i].second;
                // * sqrtMassVec.local_element(masterData[i]);
              }

            bool         constraintExists = false;
            unsigned int constraintIndex  = 0;

            for (auto i = 0; i < masterNodeBuckets.size(); i++)
              {
                if ((masterNodeBuckets[i] == masterData) &&
                    (inhomogenityList[i] == inhomogenity))
                  {
                    constraintIndex  = i;
                    constraintExists = true;
                    break;
                  }
              }

            if (constraintExists)
              {
                slaveNodeBuckets[constraintIndex].push_back(
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local(lineDof));
                weightMatrixList[constraintIndex].insert(
                  weightMatrixList[constraintIndex].end(),
                  weightData.begin(),
                  weightData.end());
              }
            else
              {
                slaveNodeBuckets.push_back(std::vector<unsigned int>(
                  1,
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local(lineDof)));
                weightMatrixList.push_back(weightData);
                masterNodeBuckets.push_back(masterData);
                inhomogenityList.push_back(inhomogenity);
              }
          }
      }

    for (dealii::IndexSet::ElementIterator it = ghost_dofs.begin();
         it != ghost_dofs.end();
         it++)
      {
        if (d_constraintMatrixPtr->is_constrained(*it))
          {
            const dealii::types::global_dof_index lineDof = *it;
            const std::vector<
              std::pair<dealii::types::global_dof_index, double>> *rowData =
              d_constraintMatrixPtr->get_constraint_entries(lineDof);
            bool isConstraintRhsExpandingOutOfIndexSet = false;

            for (unsigned int j = 0; j < rowData->size(); j++)
              {
                if (!(d_matrixFreeDataPtr
                        ->get_vector_partitioner(d_matrixFreeVectorComponent)
                        ->is_ghost_entry((*rowData)[j].first) ||
                      d_matrixFreeDataPtr
                        ->get_vector_partitioner(d_matrixFreeVectorComponent)
                        ->in_local_range((*rowData)[j].first)))
                  {
                    isConstraintRhsExpandingOutOfIndexSet = true;
                    break;
                  }
              }

            if (isConstraintRhsExpandingOutOfIndexSet)
              continue;

            // isConstrained[d_matrixFreeDataPtr
            //                 ->get_vector_partitioner(
            //                   d_matrixFreeVectorComponent)
            //                 ->global_to_local(lineDof)] = true;

            std::vector<unsigned int> masterData(rowData->size());
            std::vector<double>       weightData(rowData->size());
            double                    inhomogenity =
              d_constraintMatrixPtr->get_inhomogeneity(lineDof);

            for (auto i = 0; i < rowData->size(); i++)
              {
                masterData[i] =
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local((*rowData)[i].first);
                weightData[i] = (*rowData)[i].second;
                // * sqrtMassVec.local_element(masterData[i]);
              }

            bool         constraintExists = false;
            unsigned int constraintIndex  = 0;

            for (auto i = 0; i < masterNodeBuckets.size(); i++)
              {
                if ((masterNodeBuckets[i] == masterData) &&
                    (inhomogenityList[i] == inhomogenity))
                  {
                    constraintIndex  = i;
                    constraintExists = true;
                    break;
                  }
              }
            if (constraintExists)
              {
                slaveNodeBuckets[constraintIndex].push_back(
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local(lineDof));
                weightMatrixList[constraintIndex].insert(
                  weightMatrixList[constraintIndex].end(),
                  weightData.begin(),
                  weightData.end());
              }
            else
              {
                slaveNodeBuckets.push_back(std::vector<unsigned int>(
                  1,
                  d_matrixFreeDataPtr
                    ->get_vector_partitioner(d_matrixFreeVectorComponent)
                    ->global_to_local(lineDof)));
                weightMatrixList.push_back(weightData);
                masterNodeBuckets.push_back(masterData);
                inhomogenityList.push_back(inhomogenity);
              }
          }
      }

    // sqrtMassVec.zero_out_ghosts();

    std::vector<unsigned int> masterNodeOffset(masterNodeBuckets.size() + 1),
      slaveNodeOffset(slaveNodeBuckets.size() + 1),
      weightMatrixOffset(weightMatrixList.size() + 1);

    unsigned int k = 0;

    for (unsigned int i = 0; i < masterNodeBuckets.size(); i++)
      {
        masterNodeOffset[i] = k;
        k += masterNodeBuckets[i].size();
      }

    masterNodeOffset[masterNodeBuckets.size()] = k;
    masterNodeBucketsDevice.resize(k);

    for (unsigned int i = 0; i < masterNodeBuckets.size(); i++)
      {
        cudaMemcpy(thrust::raw_pointer_cast(masterNodeBucketsDevice.data()) +
                     masterNodeOffset[i],
                   masterNodeBuckets[i].data(),
                   masterNodeBuckets[i].size() * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
      }

    k = 0;

    for (unsigned int i = 0; i < slaveNodeBuckets.size(); i++)
      {
        slaveNodeOffset[i] = k;
        k += slaveNodeBuckets[i].size();
      }

    slaveNodeOffset[slaveNodeBuckets.size()] = k;
    slaveNodeBucketsDevice.resize(k);

    for (unsigned int i = 0; i < slaveNodeBuckets.size(); i++)
      cudaMemcpy(thrust::raw_pointer_cast(slaveNodeBucketsDevice.data()) +
                   slaveNodeOffset[i],
                 slaveNodeBuckets[i].data(),
                 slaveNodeBuckets[i].size() * sizeof(unsigned int),
                 cudaMemcpyHostToDevice);

    k = 0;

    for (unsigned int i = 0; i < weightMatrixList.size(); i++)
      {
        weightMatrixOffset[i] = k;
        k += weightMatrixList[i].size();
      }

    weightMatrixOffset[weightMatrixList.size()] = k;
    weightMatrixListDevice.resize(k);

    for (unsigned int i = 0; i < weightMatrixList.size(); i++)
      cudaMemcpy(thrust::raw_pointer_cast(weightMatrixListDevice.data()) +
                   weightMatrixOffset[i],
                 weightMatrixList[i].data(),
                 weightMatrixList[i].size() * sizeof(double),
                 cudaMemcpyHostToDevice);

    masterNodeOffsetDevice   = masterNodeOffset;
    slaveNodeOffsetDevice    = slaveNodeOffset;
    weightMatrixOffsetDevice = weightMatrixOffset;
    inhomogenityListDevice   = inhomogenityList;
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::distribute(
    double *   x,
    const int &numberWaveFunctions)
  {
    if (slaveNodeBucketsDevice.size() == 0)
      return;

    constexpr int yThreads = 64;
    const int     batch    = numberWaveFunctions / d_batchsize;

    dim3 blocks(inhomogenityListDevice.size(), batch, 1);
    dim3 threads(d_batchsize, yThreads, 1);

    distributeKernel<double, d_batchsize, d_ndofsPerDim><<<blocks, threads>>>(
      x,
      thrust::raw_pointer_cast(masterNodeBucketsDevice.data()),
      thrust::raw_pointer_cast(masterNodeOffsetDevice.data()),
      thrust::raw_pointer_cast(slaveNodeBucketsDevice.data()),
      thrust::raw_pointer_cast(slaveNodeOffsetDevice.data()),
      thrust::raw_pointer_cast(weightMatrixListDevice.data()),
      thrust::raw_pointer_cast(weightMatrixOffsetDevice.data()),
      thrust::raw_pointer_cast(inhomogenityListDevice.data()),
      thrust::raw_pointer_cast(ghostMapDevice.data()),
      d_nLocalDofs,
      d_nGhostDofs);
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::
    distributeSlaveToMaster(double *   Ax,
                            double *   x,
                            const int &numberWaveFunctions)
  {
    if (slaveNodeBucketsDevice.size() == 0)
      return;

    constexpr int yThreads = 64;
    const int     batch    = numberWaveFunctions / d_batchsize;

    dim3 blocks(inhomogenityListDevice.size(), batch, 1);
    dim3 threads(d_batchsize, yThreads, 1);

    distributeSlaveToMasterKernel<double, d_batchsize, d_ndofsPerDim>
      <<<blocks, threads>>>(
        Ax,
        x,
        thrust::raw_pointer_cast(masterNodeBucketsDevice.data()),
        thrust::raw_pointer_cast(masterNodeOffsetDevice.data()),
        thrust::raw_pointer_cast(slaveNodeBucketsDevice.data()),
        thrust::raw_pointer_cast(slaveNodeOffsetDevice.data()),
        thrust::raw_pointer_cast(weightMatrixListDevice.data()),
        thrust::raw_pointer_cast(weightMatrixOffsetDevice.data()),
        thrust::raw_pointer_cast(ghostMapDevice.data()),
        d_nLocalDofs,
        d_nGhostDofs);
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::getPartitioner()
  {
    return d_batchedPartitioner;
  }

  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  dftUtils::constraintMatrixInfoDevice &
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::
    getConstraintsInfo()
  {
    return d_constraintsInfo;
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::setCoeffs(
    std::vector<double> &coeffs)
  {
    d_Veff  = coeffs;
    VeffPtr = thrust::raw_pointer_cast(d_Veff.data());
  }


  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  void
  matrixFree<d_ndofsPerDim, d_nQuadPointsPerDim, d_batchsize>::computeAXMF(
    double *      Ax,
    const double *x,
    const int &   numberWaveFunctions)
  {
    constexpr int dim = 3;
    constexpr int p   = d_ndofsPerDim;
    constexpr int q   = d_nQuadPointsPerDim;

    constexpr int yThreads = (q != p ? 128 : (p < 9 ? 64 : 128));
    const int     batch    = numberWaveFunctions / d_batchsize;
    const size_t  smem     = 2 * d_batchsize * q * q * q * sizeof(double);

    dim3 blocks(d_nLocalCells, batch, 1);
    dim3 threads(d_batchsize, yThreads, 1);

    computeAXKernel<double, p, q, dim, d_batchsize>
      <<<blocks, threads, smem>>>(Ax,
                                       x,
                                       VeffPtr,
                                       jacobianFactorPtr,
                                       d_mapPtr,
                                       thrust::raw_pointer_cast(
                                         ghostMapDevice.data()),
                                       d_nLocalDofs,
                                       d_nGhostDofs);

    // DEVICE_API_CHECK(cudaPeekAtLastError());

    // <<<blocks, threads, smem>>>(Ax, x, VeffPtr, jacobianActionPtr, d_mapPtr);
    //*/

    // Old MatrixFree Shared Memory Implementation
    /*sharedFusedKernel<p, batchSize>
      <<<d_nLocalCells * batch, threads, smem2>>>(Ax,
                                                  x,
                                                  shapeFunctionValuePtr,
                                                  shapeFunctionGradientPtr,
                                                  VeffPtr,
                                                  d_mapPtr,
                                                  jacobianActionPtr,
                                                  numberWaveFunctions,
                                                  d_nLocalCells,
                                                  numberWaveFunctions /
                                                    batchSize,
                                                  d_nRelaventDofs); //*/
  }

#include "matrixFreeDevice.inst.cc"
} // namespace dftfe
