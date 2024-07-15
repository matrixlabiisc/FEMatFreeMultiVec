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
#include <oncvClass.h>

namespace dftfe
{
  /**
   * @brief MatrixFree class template. template parameter ndofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   */
  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  class MatrixFree : public MatrixFreeBase
  {
  public:
    /// Constructor
    MatrixFree(
      const MPI_Comm &mpi_comm,
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        basisOperationsPtrHost,
      std::shared_ptr<
        AtomicCenteredNonLocalOperator<dataTypes::number,
                                       dftfe::utils::MemorySpace::HOST>>
        ONCVnonLocalOperator,
      std::shared_ptr<
        dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>>
                 oncvClassPtr,
      const bool isGGA,
      const int  kPointIndex,
      const int  blockSize);


    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    void
    reinit(const int matrixFreeQuadratureID);


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


    void
    computeAX(dealii::VectorizedArray<double> *Ax,
              dealii::VectorizedArray<double> *x,
              dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                                dftfe::utils::MemorySpace::HOST>
                &          d_ONCVNonLocalProjectorTimesVectorBlock,
              const double scalarHX,
              const bool   hasNonlocalComponents);


  private:
    /**
     * @brief evaluate tensor contractions for LDA
     *
     */
    void
    evalHXLDA(const int iCell);


    /**
     * @brief evaluate tensor contractions for GGA
     *
     */
    void
    evalHXGGA(const int iCell);


    /**
     * @brief initialize optimized constraints.
     *
     */
    void
    initConstraints();

    void
    setupConstraints(const dealii::IndexSet &indexSet);

    inline int
    getMultiVectorIndex(const int nodeIdx, const int batchIdx) const;


    /// pointer to dealii MatrixFree object
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

    /// pointer to dealii dealii::AffineConstraints<double> object
    const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::number,
                                     dftfe::utils::MemorySpace::HOST>>
      d_ONCVnonLocalOperator;

    std::shared_ptr<
      dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>>
      d_oncvClassPtr;

    std::vector<std::vector<std::vector<dataTypes::number>>>
      d_CMatrixEntriesConjugate, d_CMatrixEntriesTranspose;

    const int  d_blockSize, d_nBatch, d_kPointIndex;
    const bool d_isGGA;

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

    std::vector<int> singleVectorGlobalToLocalMap, singleVectorToMultiVectorMap;
    std::vector<double> jacobianFactor, jacobianDeterminants,
      cellInverseMassVector;

    std::vector<std::vector<int>> d_constrainingNodeBuckets,
      d_constrainedNodeBuckets;
    std::vector<std::vector<double>> d_weightMatrixList,
      d_scaledWeightMatrixList;
    std::vector<double>                                    d_inhomogenityList;
    dealii::VectorizedArray<double>                        d_temp;
    dealii::AlignedVector<dealii::VectorizedArray<double>> d_constrainingData,
      d_constrainedData;

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
    const int                  n_mpi_processes;
    const int                  this_mpi_process;
    std::vector<MPI_Request>   mpiRequestsGhost;
    std::vector<MPI_Request>   mpiRequestsCompress;
  };

} // namespace dftfe
#endif // matrixFree_H_
