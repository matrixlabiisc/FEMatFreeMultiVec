// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//

#ifndef matrixFree_H_
#define matrixFree_H_
#include <matrixFreeBase.h>
#include <DeviceAPICalls.h>
#include <headers.h>
#include <constraintMatrixInfoDevice.h>

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>
#endif

#include <deal.II/matrix_free/vector_data_exchange.h>

namespace dftfe
{
  /**
   * @brief matrixFree class template. template parameter FEOrderElectro
   * is the finite element polynomial order. FEOrder template parameter is used
   * in conjunction with FEOrderElectro to determine the order of the Gauss
   * quadrature rule
   *
   * @author Gourab Panigrahi
   */
  template <unsigned int d_ndofsPerDim,
            unsigned int d_nQuadPointsPerDim,
            unsigned int d_batchsize>
  class matrixFree : public matrixFreeBase
  {
  public:
    /// Constructor
    matrixFree(const MPI_Comm &mpi_comm, const unsigned int blocksize);

    /**
     * @brief clears all datamembers and reset to original state.
     *
     *
     */
    void
    clear();

    /**e
     * @brief initialize optimized constraints.
     *
     *
     */
    void
    initializeOptimizedConstraints(
      const distributedCPUVec<double> &sqrtMassVec);
    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(const dealii::MatrixFree<3, double> *    matrixFreeDataPtr,
           const dealii::AffineConstraints<double> *constraintMatrixPtr,
           const distributedCPUVec<double> &        sqrtMassVec,
           const unsigned int                       matrixFreeVectorComponent,
           const unsigned int matrixFreeQuadratureComponentAX);
    void
    setCoeffs(std::vector<double> &coeffs);

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
    getPartitioner();

    dftUtils::constraintMatrixInfoDevice &
    getConstraintsInfo();

    void
    distribute(double *x, const int &numberWaveFunctions);

    void
    distributeSlaveToMaster(double *   Ax,
                            double *   x,
                            const int &numberWaveFunctions);

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAXMF(double *Ax, const double *x, const int &numberWaveFunctions);

    distributedCPUVec<double> d_cpuTestInputDealiiBatched;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      d_batchedPartitioner;

    /// duplicate constraints object with flattened maps for faster access
    dftUtils::constraintMatrixInfoDevice d_constraintsInfo;

  private:
    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    unsigned int       d_matrixFreeVectorComponent;
    unsigned int       d_matrixFreeQuadratureComponentAX;
    const unsigned int d_blocksize;
    unsigned int       d_nLocalCells;
    const unsigned int d_ndofsPerCell;
    const unsigned int d_nQuadPointsPerCell;
    // const unsigned int d_dofEDim, d_dofODim, d_quadEDim, d_quadODim;
    unsigned int       d_nLocalDofs;
    unsigned int       d_nGhostDofs;
    unsigned int       d_nRelaventDofs;
    const unsigned int d_nBatch;


    /// Matrix free data
    thrust::device_vector<unsigned int> masterNodeBucketsDevice,
      slaveNodeBucketsDevice, masterNodeOffsetDevice, slaveNodeOffsetDevice,
      weightMatrixOffsetDevice, singleVectorMapDevice, ghostMapDevice;
    thrust::device_vector<double> weightMatrixListDevice;
    thrust::device_vector<double> inhomogenityListDevice;

    double *nodalShapeFunctionValuesAtQuadPoints,
      *nodalShapeFunctionValuesAtQuadPointsTranspose;
    double *quadShapeFunctionGradientsAtQuadPoints,
      *quadShapeFunctionGradientsAtQuadPointsTranspose;
    double *      jacobianFactorPtr;
    double *      VeffPtr;
    unsigned int *d_mapPtr;

    double *shapeFunctionValuePtr, *shapeFunctionGradientPtr;

    std::vector<double> d_invSqrtElementalMassVector;

    thrust::device_vector<unsigned int> d_map;
    thrust::device_vector<double>       d_jacobianFactor;
    thrust::device_vector<double>       d_Veff;

    thrust::device_vector<double> shapeF, shapeG;

    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    std::vector<MPI_Request>   mpiRequestsGhost;
    std::vector<MPI_Request>   mpiRequestsCompress;
    dealii::ConditionalOStream pcout;
    std::unique_ptr<
      dealii::internal::MatrixFreeFunctions::VectorDataExchange::Full>
                                                 d_dealiiDataExchangePtr;
    std::vector<dealii::ArrayView<const double>> sharedStoreage;
  };

} // namespace dftfe
#endif // matrixFree_H_
