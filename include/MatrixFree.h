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
#include <MatrixFreeBase.h>
#include <mkl.h>

namespace dftfe
{
  /**
   * @brief MatrixFree class template. template parameter ndofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   */
  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  class MatrixFree : public MatrixFreeBase
  {
  public:
    /// Constructor
    MatrixFree(const MPI_Comm &                    mpi_comm,
               std::shared_ptr<dftfe::basis::FEBasisOperations<
                 dataTypes::number,
                 double,
                 dftfe::utils::MemorySpace::HOST>> basisOperationsPtrHost,
               const unsigned int                  blockSize);


    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(const unsigned int densityQuadratureID);


    /**
     * @brief clears all datamembers and reset to original state.
     *
     */
    void
    clear();


    /**
     * @brief initialize optimized constraints.
     *
     */
    void
    initializeOptimizedConstraints();


    /**
     * @brief evaluate tensor contractions
     *
     */
    void
    evaluateTensorContractions(const unsigned int iCell);


    /**
     * @brief evaluate tensor contractions
     *
     */
    void
    evaluateTensorContractionsJIT(const unsigned int iCell);


    /**
     * @brief even-odd matmul.
     *
     */
    void
    matmulEO(const unsigned int jitPointerIndex,
             const unsigned int m,
             const unsigned int n,
             const unsigned int k,
             double *           A,
             double *           B,
             double *           C,
             const unsigned int c);


    /**
     * @brief matmul.
     *
     */
    void
    matmul(const unsigned int jitPointerIndex,
           const unsigned int m,
           const unsigned int n,
           const unsigned int k,
           double *           A,
           double *           B,
           double *           C,
           const unsigned int c);


    /**
     * @brief destroy jit kernels.
     *
     */
    void
    destroyjitkernels();


    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &Ax,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &x);

    void
    setVeffMF(
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &VeffJxW,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
        &VeffExtPotJxW);

  private:
    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    unsigned int d_matrixFreeVectorComponent;

    unsigned int d_matrixFreeQuadratureComponentAX;

    const unsigned int d_blockSize;

    unsigned int d_nOwnedDofs, d_nRelaventDofs, d_nGhostDofs, d_nCells,
      d_nDofsPerCell, d_nQuadsPerCell;

    // const unsigned int d_dofEDim, d_dofODim, d_quadEDim, d_quadODim;

    const unsigned int d_nBatch;

    /// duplicate constraints object with flattened maps for faster access
    dftUtils::constraintMatrixInfo d_constraintsInfomf;
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      d_batchedPartitioner;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxW;

    /// Matrix free data
    distributedCPUVec<double> cpuTestInputdealiibatched;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;

    std::vector<unsigned int> globalToLocalMap, singleVectorGlobalToLocalMap;

    std::vector<std::vector<unsigned int>> slaveNodeBuckets, masterNodeBuckets;
    std::vector<std::vector<double>>       weightMatrixList;
    std::vector<double>                    inhomogenityList;
    double *                               nodalShapeFunctionValuesAtQuadPoints,
      *nodalShapeFunctionValuesAtQuadPointsTranspose;

    static constexpr int d_quadODim = nQuadPointsPerDim / 2;
    static constexpr int d_quadEDim =
      nQuadPointsPerDim % 2 == 1 ? d_quadODim + 1 : d_quadODim;
    static constexpr int d_dofODim = ndofsPerDim / 2;
    static constexpr int d_dofEDim =
      ndofsPerDim % 2 == 1 ? d_dofODim + 1 : d_dofODim;
    std::array<double, d_dofEDim * d_quadEDim + d_dofODim * d_quadODim>
      nodalShapeFunctionValuesAtQuadPointsEO;
    std::array<double, 2 * d_quadODim * d_quadEDim>
            quadShapeFunctionGradientsAtQuadPointsEO;
    double *quadShapeFunctionGradientsAtQuadPoints,
      *quadShapeFunctionGradientsAtQuadPointsTranspose;
    double *            temp10, *temp11, *temp12, *temp20, *temp21, *temp22;
    std::vector<double> jacobianFactor;
    std::vector<double> jacobianDeterminants;

    std::vector<void *>                    jitpointers;
    std::vector<std::pair<void *, void *>> jitpointersEO;
    std::vector<dgemm_jit_kernel_t>        jitGemmKernels;
    std::vector<std::pair<dgemm_jit_kernel_t, dgemm_jit_kernel_t>>
      jitGemmKernelsEO;
    std::array<double,
               2 * 3 * batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
                   nQuadPointsPerDim +
                 2 * ndofsPerDim * nQuadPointsPerDim>
      tempStorage;
    dealii::AlignedVector<dealii::VectorizedArray<double>>
                                     tempStorageVectorized;
    dealii::VectorizedArray<double> *temp10v, *temp11v, *temp12v, *temp20v,
      *temp21v, *temp22v;
    std::vector<double>        tempGhostStorage, tempCompressStorage;
    std::vector<double>        tempConstraintStorage;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    std::vector<MPI_Request>   mpiRequestsGhost;
    std::vector<MPI_Request>   mpiRequestsCompress;
    std::vector<MPI_Datatype>  mpighostTypes;
    std::vector<MPI_Datatype>  mpiownedTypes;
    std::vector<unsigned int>  mpighostoffsets;
    std::vector<unsigned int>  mpighostcounts;
    std::vector<int>           mpiownedtargets;
    std::vector<int>           mpighosttargets;
    dealii::ConditionalOStream pcout;
    std::unique_ptr<
      dealii::internal::MatrixFreeFunctions::VectorDataExchange::Full>
                                                 d_dealiiDataExchangePtr;
    std::vector<dealii::ArrayView<const double>> sharedStoreageGhosts,
      sharedStoreageCompress;
    std::vector<double *> sharedStoreageGhostsPtrs, sharedStoreageCompressPtrs;
    MPI_Win               sharedWinGhosts, sharedWinCompress; // window
    MPI_Comm              shared_comm;
    int                   rank_sm;
    int                   size_sm;
  };

} // namespace dftfe
#endif // matrixFree_H_
