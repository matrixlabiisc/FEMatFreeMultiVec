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
#include <linearAlgebraOperations.h>
#include <constraintMatrixInfo.h>
#include <vectorUtilities.h>


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
               const bool                          isGGA,
               const unsigned int                  blockSize);


    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(const unsigned int matrixFreeQuadratureID);


    /**
     * @brief set Veff and VeffExtPot for matrixFree AX
     *
     */
    void
    setVJxWMF(
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &VeffJxW,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
        &VeffExtPotJxW,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST> &VGGAJxW);


    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &Ax,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::HOST> &x,
      const double scalarHX);


  private:
    /**
     * @brief evaluate tensor contractions for LDA
     *
     */
    void
    evalHXLDA(const unsigned int iCell);


    /**
     * @brief evaluate tensor contractions for GGA
     *
     */
    void
    evalHXGGA(const unsigned int iCell);


    /**
     * @brief initialize optimized constraints.
     *
     */
    void
    initConstraints();

    void
    setupConstraints(const dealii::IndexSet &indexSet);

    inline unsigned int
    getMultiVectorIndex(const unsigned int nodeIdx,
                        const unsigned int batchIdx) const;


    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    const unsigned int d_blockSize, d_nBatch;
    const bool         d_isGGA;

    unsigned int d_nOwnedDofs, d_nRelaventDofs, d_nGhostDofs, d_nCells,
      d_nDofsPerCell, d_nQuadsPerCell;

    /// duplicate constraints object with flattened maps for faster access
    dftUtils::constraintMatrixInfo d_constraintsInfo;
    std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      d_singleVectorPartitioner, d_singleBatchPartitioner;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_VeffJxW, d_VeffExtPotJxW, d_VGGAJxW;

    /// Matrix free data
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsPtrHost;

    std::vector<unsigned int> singleVectorGlobalToLocalMap,
      singleVectorToMultiVectorMap;
    std::vector<double> jacobianFactor, jacobianDeterminants,
      cellInverseMassVector;

    std::vector<std::vector<unsigned int>> slaveNodeBuckets, masterNodeBuckets;
    std::vector<std::vector<double>> weightMatrixList, scaledWeightMatrixList;
    std::vector<double>              inhomogenityList;

    static constexpr int d_quadODim = nQuadPointsPerDim / 2;
    static constexpr int d_quadEDim =
      nQuadPointsPerDim % 2 == 1 ? d_quadODim + 1 : d_quadODim;
    static constexpr int d_dofODim = ndofsPerDim / 2;
    static constexpr int d_dofEDim =
      ndofsPerDim % 2 == 1 ? d_dofODim + 1 : d_dofODim;

    std::array<double, d_quadEDim * d_dofEDim + d_quadODim * d_dofODim>
      nodalShapeFunctionValuesAtQuadPointsEO;
    std::array<double, 2 * d_quadODim * d_quadEDim>
                                          quadShapeFunctionGradientsAtQuadPointsEO;
    std::array<double, nQuadPointsPerDim> quadratureWeights;

    dealii::AlignedVector<dealii::VectorizedArray<double>> alignedVector;
    dealii::VectorizedArray<double> *arrayV, *arrayW, *arrayX, *arrayY, *arrayZ;

    dealii::ConditionalOStream pcout;
    std::vector<double>        tempGhostStorage, tempCompressStorage;
    const MPI_Comm             mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    std::vector<MPI_Request>   mpiRequestsGhost;
    std::vector<MPI_Request>   mpiRequestsCompress;
  };

} // namespace dftfe
#endif // matrixFree_H_
