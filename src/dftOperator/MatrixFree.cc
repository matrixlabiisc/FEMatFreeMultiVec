// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Gourab Panigrahi
//

#include <MatrixFree.h>

namespace dftfe
{
  template <int m, int n, int k, int batchCount, bool add, bool trans>
  inline void
  matMulShapeEO(const dealii::VectorizedArray<double> *A,
                const double *                         B,
                dealii::VectorizedArray<double> *      C)
  {
    constexpr int ko = k / 2;
    constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
    constexpr int no = n / 2;
    constexpr int ne = n % 2 == 1 ? no + 1 : no;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];

          for (int q = 0; q < ko; q++)
            {
              tempAe[q] =
                A[i + q * m + b * m * k] + A[i + (k - 1 - q) * m + b * m * k];
              tempAo[q] =
                A[i + q * m + b * m * k] - A[i + (k - 1 - q) * m + b * m * k];
            }

          if (k % 2 == 1)
            tempAe[ko] = A[i + ko * m + b * m * k];

          for (int j = 0; j < no; j++)
            {
              dealii::VectorizedArray<double> tempCe, tempCo;

              if (trans)
                tempCe = tempAe[0] * B[j];
              else
                tempCe = tempAe[0] * B[j * ke];

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    tempCe += tempAe[q] * B[j + q * ne];
                  else
                    tempCe += tempAe[q] * B[q + j * ke];
                }

              if (trans)
                tempCo = tempAo[0] * B[j + ke * ne];
              else
                tempCo = tempAo[0] * B[j * ko + ke * ne];

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    tempCo += tempAo[q] * B[j + q * no + ke * ne];
                  else
                    tempCo += tempAo[q] * B[q + j * ko + ke * ne];
                }

              if (add)
                {
                  C[i + j * m + b * m * n] += tempCe + tempCo;
                  C[i + (n - 1 - j) * m + b * m * n] += tempCe - tempCo;
                }
              else
                {
                  C[i + j * m + b * m * n]           = tempCe + tempCo;
                  C[i + (n - 1 - j) * m + b * m * n] = tempCe - tempCo;
                }
            }

          if (n % 2 == 1)
            {
              if (add)
                {
                  if (trans)
                    C[i + no * m + b * m * n] += tempAe[0] * B[no];
                  else
                    C[i + no * m + b * m * n] += tempAe[0] * B[no * ke];
                }
              else
                {
                  if (trans)
                    C[i + no * m + b * m * n] = tempAe[0] * B[no];
                  else
                    C[i + no * m + b * m * n] = tempAe[0] * B[no * ke];
                }

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    C[i + no * m + b * m * n] += tempAe[q] * B[no + q * ne];
                  else
                    C[i + no * m + b * m * n] += tempAe[q] * B[q + no * ke];
                }
            }
        }
  }


  template <int m, int n, int k, int batchCount, bool trans>
  inline void
  matMulShapeEO(const dealii::VectorizedArray<double> *A,
                const double *                         B,
                dealii::VectorizedArray<double> *      C,
                const double *                         coeffs)
  {
    constexpr int ko = k / 2;
    constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
    constexpr int no = n / 2;
    constexpr int ne = n % 2 == 1 ? no + 1 : no;

    dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];
    dealii::VectorizedArray<double> tempCe, tempCo;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          for (int q = 0; q < ko; q++)
            {
              tempAe[q] =
                A[i + q * m + b * m * k] + A[i + (k - 1 - q) * m + b * m * k];
              tempAo[q] =
                A[i + q * m + b * m * k] - A[i + (k - 1 - q) * m + b * m * k];
            }

          if (k % 2 == 1)
            tempAe[ko] = A[i + ko * m + b * m * k];

          for (int j = 0; j < no; j++)
            {
              if (trans)
                tempCe = tempAe[0] * B[j];
              else
                tempCe = tempAe[0] * B[j * ke];

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    tempCe += tempAe[q] * B[j + q * ne];
                  else
                    tempCe += tempAe[q] * B[q + j * ke];
                }

              if (trans)
                tempCo = tempAo[0] * B[j + ke * ne];
              else
                tempCo = tempAo[0] * B[j * ko + ke * ne];

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    tempCo += tempAo[q] * B[j + q * no + ke * ne];
                  else
                    tempCo += tempAo[q] * B[q + j * ko + ke * ne];
                }

              C[i + j * m + b * m * n] =
                C[i + j * m + b * m * n] * coeffs[i + j * m + b * m * n] +
                tempCe + tempCo;

              C[i + (n - 1 - j) * m + b * m * n] =
                C[i + (n - 1 - j) * m + b * m * n] *
                  coeffs[i + (n - 1 - j) * m + b * m * n] +
                tempCe - tempCo;
            }

          if (n % 2 == 1)
            {
              if (trans)
                C[i + no * m + b * m * n] =
                  C[i + no * m + b * m * n] * coeffs[i + no * m + b * m * n] +
                  tempAe[0] * B[no];
              else
                C[i + no * m + b * m * n] =
                  C[i + no * m + b * m * n] * coeffs[i + no * m + b * m * n] +
                  tempAe[0] * B[no * ke];

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    C[i + no * m + b * m * n] += tempAe[q] * B[no + q * ne];
                  else
                    C[i + no * m + b * m * n] += tempAe[q] * B[q + no * ke];
                }
            }
        }
  }


  template <int m, int n, int k, int batchCount, bool add, bool trans>
  inline void
  matMulGradEO(const dealii::VectorizedArray<double> *A,
               const double *                         B,
               dealii::VectorizedArray<double> *      C)
  {
    constexpr int ko = k / 2;
    constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
    constexpr int no = n / 2;
    constexpr int ne = n % 2 == 1 ? no + 1 : no;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];

          for (int q = 0; q < ko; q++)
            {
              tempAe[q] =
                A[i + q * m + b * m * k] + A[i + (k - 1 - q) * m + b * m * k];
              tempAo[q] =
                A[i + q * m + b * m * k] - A[i + (k - 1 - q) * m + b * m * k];
            }

          if (k % 2 == 1)
            tempAe[ko] = A[i + ko * m + b * m * k];

          for (int j = 0; j < no; j++)
            {
              dealii::VectorizedArray<double> tempCe, tempCo;

              if (trans)
                tempCe = tempAe[0] * B[j];
              else
                tempCe = tempAe[0] * B[j * ke + ko * ne];

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    tempCe += tempAe[q] * B[j + q * no];
                  else
                    tempCe += tempAe[q] * B[q + j * ke + ko * ne];
                }

              if (trans)
                tempCo = tempAo[0] * B[j + ke * no];
              else
                tempCo = tempAo[0] * B[j * ko];

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    tempCo += tempAo[q] * B[j + q * ne + ke * no];
                  else
                    tempCo += tempAo[q] * B[q + j * ko];
                }

              if (add)
                {
                  C[i + j * m + b * m * n] += tempCe + tempCo;
                  C[i + (n - 1 - j) * m + b * m * n] += tempCo - tempCe;
                }
              else
                {
                  C[i + j * m + b * m * n]           = tempCe + tempCo;
                  C[i + (n - 1 - j) * m + b * m * n] = tempCo - tempCe;
                }
            }

          if (n % 2 == 1)
            {
              if (add)
                {
                  if (trans)
                    C[i + no * m + b * m * n] += tempAo[0] * B[no + ke * no];
                  else
                    C[i + no * m + b * m * n] += tempAo[0] * B[no * ko];
                }
              else
                {
                  if (trans)
                    C[i + no * m + b * m * n] = tempAo[0] * B[no + ke * no];
                  else
                    C[i + no * m + b * m * n] = tempAo[0] * B[no * ko];
                }

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    C[i + no * m + b * m * n] +=
                      tempAo[q] * B[no + q * ne + ke * no];
                  else
                    C[i + no * m + b * m * n] += tempAo[q] * B[q + no * ko];
                }
            }
        }
  }


  template <int m, int n, int k, int batchCount, bool trans>
  inline void
  matMulGradEO(const dealii::VectorizedArray<double> *A,
               const double *                         B,
               dealii::VectorizedArray<double> *      C,
               const double *                         coeffs)
  {
    constexpr int ko = k / 2;
    constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
    constexpr int no = n / 2;
    constexpr int ne = n % 2 == 1 ? no + 1 : no;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];

          for (int q = 0; q < ko; q++)
            {
              tempAe[q] =
                A[i + q * m + b * m * k] + A[i + (k - 1 - q) * m + b * m * k];
              tempAo[q] =
                A[i + q * m + b * m * k] - A[i + (k - 1 - q) * m + b * m * k];
            }

          if (k % 2 == 1)
            tempAe[ko] = A[i + ko * m + b * m * k];

          for (int j = 0; j < no; j++)
            {
              dealii::VectorizedArray<double> tempCe, tempCo;

              if (trans)
                tempCe = tempAe[0] * B[j];
              else
                tempCe = tempAe[0] * B[j * ke + ko * ne];

              for (int q = 1; q < ke; q++)
                {
                  if (trans)
                    tempCe += tempAe[q] * B[j + q * no];
                  else
                    tempCe += tempAe[q] * B[q + j * ke + ko * ne];
                }

              if (trans)
                tempCo = tempAo[0] * B[j + ke * no];
              else
                tempCo = tempAo[0] * B[j * ko];

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    tempCo += tempAo[q] * B[j + q * ne + ke * no];
                  else
                    tempCo += tempAo[q] * B[q + j * ko];
                }

              C[i + j * m + b * m * n] =
                C[i + j * m + b * m * n] * coeffs[i + j * m + b * m * n] +
                tempCe + tempCo;

              C[i + (n - 1 - j) * m + b * m * n] =
                C[i + (n - 1 - j) * m + b * m * n] *
                  coeffs[i + (n - 1 - j) * m + b * m * n] +
                tempCo - tempCe;
            }

          if (n % 2 == 1)
            {
              if (trans)
                C[i + no * m + b * m * n] =
                  C[i + no * m + b * m * n] * coeffs[i + no * m + b * m * n] +
                  tempAo[0] * B[no + ke * no];
              else
                C[i + no * m + b * m * n] =
                  C[i + no * m + b * m * n] * coeffs[i + no * m + b * m * n] +
                  tempAo[0] * B[no * ko];

              for (int q = 1; q < ko; q++)
                {
                  if (trans)
                    C[i + no * m + b * m * n] +=
                      tempAo[q] * B[no + q * ne + ke * no];
                  else
                    C[i + no * m + b * m * n] += tempAo[q] * B[q + no * ko];
                }
            }
        }
  }


  template <int m, int n, int k, int batchCount, bool add, bool trans>
  inline void
  matMul(const dealii::VectorizedArray<double> *A,
         const double *                         B,
         dealii::VectorizedArray<double> *      C)
  {
    dealii::VectorizedArray<double> tempA[k];
    dealii::VectorizedArray<double> temp;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          for (int q = 0; q < k; q++)
            tempA[q] = A[i + q * m + b * m * k];

          for (int j = 0; j < n; j++)
            {
              temp = trans ? tempA[0] * B[j] : tempA[0] * B[j * k];

              for (int q = 1; q < k; q++)
                {
                  if (trans)
                    temp += tempA[q] * B[j + q * n];
                  else
                    temp += tempA[q] * B[q + j * k];
                }

              if (add)
                C[i + j * m + b * m * n] += temp;
              else
                C[i + j * m + b * m * n] = temp;
            }
        }
  }


  template <int m, int n, int k, int batchCount, bool trans>
  inline void
  matMul(const dealii::VectorizedArray<double> *A,
         const double *                         B,
         dealii::VectorizedArray<double> *      C,
         const double *                         coeffs)
  {
    dealii::VectorizedArray<double> tempA[k];
    dealii::VectorizedArray<double> temp;

    for (int b = 0; b < batchCount; b++)
      for (int i = 0; i < m; i++)
        {
          for (int q = 0; q < k; q++)
            tempA[q] = A[i + q * m + b * m * k];

          for (int j = 0; j < n; j++)
            {
              temp = trans ? tempA[0] * B[j] : tempA[0] * B[j * k];

              for (int q = 1; q < k; q++)
                {
                  if (trans)
                    temp += tempA[q] * B[j + q * n];
                  else
                    temp += tempA[q] * B[q + j * k];
                }

              C[i + j * m + b * m * n] =
                C[i + j * m + b * m * n] * coeffs[i + j * m + b * m * n] + temp;
            }
        }
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::MatrixFree(
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
    const int  blockSize)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_ONCVnonLocalOperator(ONCVnonLocalOperator)
    , d_oncvClassPtr(oncvClassPtr)
    , d_isGGA(isGGA)
    , d_kPointIndex(kPointIndex)
    , d_blockSize(blockSize)
    , d_nBatch(ceil((double)blockSize / (double)batchSize))
  {}


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::reinit(
    const int matrixFreeQuadratureID)
  {
    d_basisOperationsPtrHost->reinit(0, 0, matrixFreeQuadratureID);
    d_matrixFreeDataPtr = &(d_basisOperationsPtrHost->matrixFreeData());

    auto dofInfo = d_matrixFreeDataPtr->get_dof_info(
      d_basisOperationsPtrHost->d_dofHandlerID);
    auto shapeData =
      d_matrixFreeDataPtr
        ->get_shape_info(d_basisOperationsPtrHost->d_dofHandlerID,
                         matrixFreeQuadratureID)
        .get_shape_data();
    auto mappingData =
      d_matrixFreeDataPtr->get_mapping_info().cell_data[matrixFreeQuadratureID];

    d_constraintMatrixPtr =
      (*(d_basisOperationsPtrHost
           ->d_constraintsVector))[d_basisOperationsPtrHost->d_dofHandlerID];

    // Initialize member variables
    d_nOwnedDofs              = d_basisOperationsPtrHost->nOwnedDofs();
    d_nRelaventDofs           = d_basisOperationsPtrHost->nRelaventDofs();
    d_nQuadsPerCell           = d_basisOperationsPtrHost->nQuadsPerCell();
    d_nCells                  = d_basisOperationsPtrHost->nCells();
    d_nDofsPerCell            = d_basisOperationsPtrHost->nDofsPerCell();
    d_nGhostDofs              = d_nRelaventDofs - d_nOwnedDofs;
    d_singleVectorPartitioner = d_matrixFreeDataPtr->get_vector_partitioner(
      d_basisOperationsPtrHost->d_dofHandlerID);

    // Initialize shape and gradient functions
    std::array<double, ndofsPerDim * nQuadPointsPerDim>
      nodalShapeFunctionValuesAtQuadPoints;
    std::array<double, nQuadPointsPerDim * nQuadPointsPerDim>
      quadShapeFunctionGradientsAtQuadPoints;

    int arraySize =
      (batchSize / dofInfo.vectorization_length) * d_nQuadsPerCell;

    if (d_isGGA)
      {
        alignedVector.resize(5 * arraySize);
        arrayV = alignedVector.data();
        arrayW = arrayV + arraySize;
        arrayX = arrayW + arraySize;
        arrayY = arrayX + arraySize;
        arrayZ = arrayY + arraySize;
      }
    else
      {
        alignedVector.resize(4 * arraySize);
        arrayW = alignedVector.data();
        arrayX = arrayW + arraySize;
        arrayY = arrayX + arraySize;
        arrayZ = arrayY + arraySize;
      }

    for (auto iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
      quadratureWeights[iQuad] = shapeData.quadrature.weight(iQuad);

    for (auto iDoF = 0; iDoF < ndofsPerDim; iDoF++)
      for (auto iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
        {
          nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                               iDoF * nQuadPointsPerDim] =
            shapeData.shape_values[iQuad + iDoF * nQuadPointsPerDim][0] *
            (d_isGGA ? 1 : std::sqrt(quadratureWeights[iQuad]));
        }

    for (auto iQuad2 = 0; iQuad2 < nQuadPointsPerDim; iQuad2++)
      for (auto iQuad1 = 0; iQuad1 < nQuadPointsPerDim; iQuad1++)
        {
          quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                 iQuad2 * nQuadPointsPerDim] =
            shapeData
              .shape_gradients_collocation[iQuad1 + iQuad2 * nQuadPointsPerDim]
                                          [0] *
            (d_isGGA ? 1 :
                       std::sqrt(quadratureWeights[iQuad1]) /
                         std::sqrt(quadratureWeights[iQuad2]));
        }

    for (auto iDoF = 0; iDoF < d_dofEDim; iDoF++)
      for (auto iQuad = 0; iQuad < d_quadEDim; iQuad++)
        {
          nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadEDim] =
            (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                  iDoF * nQuadPointsPerDim] +
             nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                  (ndofsPerDim - 1 - iDoF) *
                                                    nQuadPointsPerDim]) *
            0.5;
        }

    for (auto iDoF = 0; iDoF < d_dofODim; iDoF++)
      for (auto iQuad = 0; iQuad < d_quadODim; iQuad++)
        {
          nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadODim +
                                                 d_quadEDim * d_dofEDim] =
            (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                  iDoF * nQuadPointsPerDim] -
             nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                  (ndofsPerDim - 1 - iDoF) *
                                                    nQuadPointsPerDim]) *
            0.5;
        }

    for (auto iQuad2 = 0; iQuad2 < d_quadEDim; iQuad2++)
      for (auto iQuad1 = 0; iQuad1 < d_quadODim; iQuad1++)
        {
          quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                   iQuad2 * d_quadODim] =
            (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                    iQuad2 *
                                                      nQuadPointsPerDim] +
             quadShapeFunctionGradientsAtQuadPoints
               [iQuad1 +
                (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
            0.5;
        }

    for (auto iQuad2 = 0; iQuad2 < d_quadODim; iQuad2++)
      for (auto iQuad1 = 0; iQuad1 < d_quadEDim; iQuad1++)
        {
          quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                   iQuad2 * d_quadEDim +
                                                   d_quadEDim * d_quadODim] =
            (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                    iQuad2 *
                                                      nQuadPointsPerDim] -
             quadShapeFunctionGradientsAtQuadPoints
               [iQuad1 +
                (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
            0.5;
        }

    // Initialize Jacobian matrix
    int dim = 3;
    jacobianFactor.resize(d_nCells * dim * dim, 0.0);
    jacobianDeterminants.resize(d_nCells, 0.0);

    auto cellOffsets = mappingData.data_index_offsets;

    for (auto iCellBatch = 0, cellCount = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         iCellBatch++)
      for (auto iCell = 0;
           iCell < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
           iCell++, cellCount++)
        for (auto d = 0; d < dim; d++)
          for (auto e = 0; e < dim; e++)
            for (auto f = 0; f < dim; f++)
              {
                jacobianFactor[e + d * dim + cellCount * dim * dim] +=
                  0.5 *
                  mappingData.jacobians[0][cellOffsets[iCellBatch]][d][f][0] *
                  mappingData.jacobians[0][cellOffsets[iCellBatch]][e][f][0] *
                  mappingData.JxW_values[cellOffsets[iCellBatch]][0];

                jacobianDeterminants[cellCount] =
                  mappingData.JxW_values[cellOffsets[iCellBatch]][0];
              }

    // Create matrix-free maps
    distributedCPUVec<double> tempBatchedVector;
    vectorTools::createDealiiVector(d_singleVectorPartitioner,
                                    batchSize,
                                    tempBatchedVector);
    d_singleBatchPartitioner = tempBatchedVector.get_partitioner();

    tempGhostStorage.resize(d_singleBatchPartitioner->n_import_indices(), 0.0);
    tempCompressStorage.resize(d_singleBatchPartitioner->n_import_indices(),
                               0.0);
    singleVectorGlobalToLocalMap.resize(d_nCells * d_nDofsPerCell, 0);
    singleVectorToMultiVectorMap.resize(d_nRelaventDofs, 0);
    std::vector<int> singleVectorGlobalToLocalMapTemp(d_nCells * d_nDofsPerCell,
                                                      0);

    // Construct cellIndexToMacroCellSubCellIndexMap
    auto d_nMacroCells = d_matrixFreeDataPtr->n_cell_batches();
    auto cellPtr       = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .begin_active();
    auto endcPtr = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .end();

    std::map<dealii::CellId, size_type> cellIdToCellIndexMap;
    std::vector<int> cellIndexToMacroCellSubCellIndexMap(d_nCells);

    int iCell = 0;
    for (; cellPtr != endcPtr; cellPtr++)
      if (cellPtr->is_locally_owned())
        {
          cellIdToCellIndexMap[cellPtr->id()] = iCell;
          iCell++;
        }

    iCell = 0;
    for (int iMacroCell = 0; iMacroCell < d_nMacroCells; iMacroCell++)
      {
        const int numberSubCells =
          d_matrixFreeDataPtr->n_components_filled(iMacroCell);

        for (int iSubCell = 0; iSubCell < numberSubCells; iSubCell++)
          {
            cellPtr = d_matrixFreeDataPtr->get_cell_iterator(
              iMacroCell, iSubCell, d_basisOperationsPtrHost->d_dofHandlerID);

            size_type cellIndex = cellIdToCellIndexMap[cellPtr->id()];
            cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;

            iCell++;
          }
      }

    // Construct singleVectorGlobalToLocalMapTemp with matrix-free cell ordering
    for (auto iCellBatch = 0, iCell = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         iCellBatch++)
      for (auto iCellLocal = 0;
           iCellLocal < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
           iCellLocal++, iCell++)
        {
          auto checkExpr =
            dofInfo.row_starts_plain_indices[iCellLocal +
                                             iCellBatch *
                                               dofInfo.vectorization_length] ==
            dealii::numbers::invalid_unsigned_int;

          auto trueClause =
            dofInfo.dof_indices.data() +
            dofInfo
              .row_starts[iCellLocal +
                          iCellBatch * dofInfo.vectorization_length]
              .first;

          auto falseClause =
            dofInfo.plain_dof_indices.data() +
            dofInfo.row_starts_plain_indices[iCellLocal +
                                             iCellBatch *
                                               dofInfo.vectorization_length];

          std::memcpy(singleVectorGlobalToLocalMapTemp.data() +
                        iCell * d_nDofsPerCell,
                      checkExpr ? trueClause : falseClause,
                      d_nDofsPerCell * sizeof(int));
        }

    // Reorder cell numbering to cell-matrix order
    for (auto iCell = 0; iCell < d_nCells; iCell++)
      for (auto iDof = 0; iDof < d_nDofsPerCell; iDof++)
        singleVectorGlobalToLocalMap[iDof + iCell * d_nDofsPerCell] =
          singleVectorGlobalToLocalMapTemp
            [iDof +
             cellIndexToMacroCellSubCellIndexMap[iCell] * d_nDofsPerCell];

    auto taskGhostMap =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->ghost_targets();

    std::vector<int> taskGhostStartIndices(n_mpi_processes, 0);

    for (auto i = 0; i < taskGhostMap.size(); i++)
      taskGhostStartIndices[taskGhostMap[i].first] = taskGhostMap[i].second;

    auto ghostSum = 0;
    for (auto i = 0; i < taskGhostStartIndices.size(); i++)
      {
        auto tmp = ghostSum;
        ghostSum += taskGhostStartIndices[i];
        taskGhostStartIndices[i] = tmp;
      }

    for (int iDof = 0; iDof < d_nRelaventDofs; iDof++)
      {
        if (iDof >= d_nOwnedDofs)
          {
            int ownerId = 0;
            while (taskGhostStartIndices[ownerId] <= iDof - d_nOwnedDofs)
              {
                ++ownerId;
                if (ownerId == n_mpi_processes)
                  break;
              }

            --ownerId;

            int ghostIdFromOwner =
              iDof - taskGhostStartIndices[ownerId] - d_nOwnedDofs;

            int nGhostsFromOwner =
              ownerId == n_mpi_processes - 1 ?
                d_nGhostDofs - taskGhostStartIndices[ownerId] :
                taskGhostStartIndices[ownerId + 1] -
                  taskGhostStartIndices[ownerId];

            singleVectorToMultiVectorMap[iDof] =
              (d_nOwnedDofs * (d_blockSize / batchSize) +
               taskGhostStartIndices[ownerId]) +
              ghostIdFromOwner;
          }
        else
          {
            singleVectorToMultiVectorMap[iDof] = iDof;
          }
      }

    // Construct TensorStructuredDoFMap
    auto &lexMap = d_matrixFreeDataPtr
                     ->get_shape_info(d_basisOperationsPtrHost->d_dofHandlerID,
                                      matrixFreeQuadratureID)
                     .lexicographic_numbering;

    // Construct CMatrixEntries and CMatrixEntriesConjugate in Lexicographic
    // Numbering
    d_ONCVnonLocalOperator->constructLexicoMappedCMatrix(
      d_CMatrixEntriesConjugate,
      d_CMatrixEntriesTranspose,
      lexMap,
      d_nCells,
      d_kPointIndex);

    // Initialize constraints
    initConstraints();
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::initConstraints()
  {
    // Initialize cellInverseMassVector
    cellInverseMassVector.resize(d_nDofsPerCell * d_nCells, 0.0);

    for (int iCell = 0; iCell < d_nCells; iCell++)
      for (int iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
        {
          int cellDoFIdx = iDoF + iCell * d_nDofsPerCell;
          int dofIdx     = singleVectorGlobalToLocalMap[cellDoFIdx];

          int globalIdx =
            d_matrixFreeDataPtr
              ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
              ->local_to_global(dofIdx);

          if (d_constraintMatrixPtr->is_constrained(globalIdx))
            cellInverseMassVector[cellDoFIdx] = 1.0;
          else
            cellInverseMassVector[cellDoFIdx] =
              d_basisOperationsPtrHost->inverseMassVectorBasisData()[dofIdx];
        }

    // Initialize constraint data structures
    const dealii::IndexSet &locallyOwnedDofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->locally_owned_range();

    setupConstraints(locallyOwnedDofs);

    const dealii::IndexSet &ghostDofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->ghost_indices();

    setupConstraints(ghostDofs);
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::setupConstraints(
    const dealii::IndexSet &indexSet)
  {
    for (dealii::IndexSet::ElementIterator iter = indexSet.begin();
         iter != indexSet.end();
         iter++)
      if (d_constraintMatrixPtr->is_constrained(*iter))
        {
          bool isConstraintRhsExpandingOutOfIndexSet    = false;
          const dealii::types::global_dof_index lineDof = *iter;
          const std::vector<std::pair<dealii::types::global_dof_index, double>>
            *rowData = d_constraintMatrixPtr->get_constraint_entries(lineDof);

          for (int j = 0; j < rowData->size(); j++)
            {
              if (!(d_matrixFreeDataPtr
                      ->get_vector_partitioner(
                        d_basisOperationsPtrHost->d_dofHandlerID)
                      ->is_ghost_entry((*rowData)[j].first) ||
                    d_matrixFreeDataPtr
                      ->get_vector_partitioner(
                        d_basisOperationsPtrHost->d_dofHandlerID)
                      ->in_local_range((*rowData)[j].first)))
                {
                  isConstraintRhsExpandingOutOfIndexSet = true;
                  break;
                }
            }

          if (isConstraintRhsExpandingOutOfIndexSet)
            continue;

          std::vector<int>    masterData(rowData->size());
          std::vector<double> weightData(rowData->size());
          std::vector<double> scaledWeightData(rowData->size());

          for (auto i = 0; i < rowData->size(); i++)
            {
              masterData[i] = d_matrixFreeDataPtr
                                ->get_vector_partitioner(
                                  d_basisOperationsPtrHost->d_dofHandlerID)
                                ->global_to_local((*rowData)[i].first);

              weightData[i] = (*rowData)[i].second;

              scaledWeightData[i] =
                ((*rowData)[i].second) *
                d_basisOperationsPtrHost
                  ->inverseMassVectorBasisData()[masterData[i]];
            }

          bool   constraintExists = false;
          int    constraintIndex  = 0;
          double inhomogenity =
            d_constraintMatrixPtr->get_inhomogeneity(lineDof);

          for (auto i = 0; i < masterNodeBuckets.size(); i++)
            if ((masterNodeBuckets[i] == masterData) &&
                (inhomogenityList[i] == inhomogenity))
              {
                constraintIndex  = i;
                constraintExists = true;
                break;
              }

          if (constraintExists)
            {
              slaveNodeBuckets[constraintIndex].push_back(
                d_matrixFreeDataPtr
                  ->get_vector_partitioner(
                    d_basisOperationsPtrHost->d_dofHandlerID)
                  ->global_to_local(lineDof));

              weightMatrixList[constraintIndex].insert(
                weightMatrixList[constraintIndex].end(),
                weightData.begin(),
                weightData.end());

              scaledWeightMatrixList[constraintIndex].insert(
                scaledWeightMatrixList[constraintIndex].end(),
                scaledWeightData.begin(),
                scaledWeightData.end());
            }
          else
            {
              slaveNodeBuckets.push_back(
                std::vector<int>(1,
                                 d_matrixFreeDataPtr
                                   ->get_vector_partitioner(
                                     d_basisOperationsPtrHost->d_dofHandlerID)
                                   ->global_to_local(lineDof)));

              weightMatrixList.push_back(weightData);
              scaledWeightMatrixList.push_back(scaledWeightData);
              masterNodeBuckets.push_back(masterData);
              inhomogenityList.push_back(inhomogenity);
            }
        }
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::setVJxWMF(
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &VeffJxW,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &VeffExtPotJxW,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &VGGAJxW)
  {
    d_VeffJxW.resize(VeffJxW.size());
    if (d_isGGA)
      d_VGGAJxW.resize(VGGAJxW.size());

    // Reorder VeffJxW and VeffExtPotJxW for matrix-free
    for (int iCell = 0; iCell < d_nCells; iCell++)
      for (int iQuad3 = 0; iQuad3 < nQuadPointsPerDim; iQuad3++)
        for (int iQuad2 = 0; iQuad2 < nQuadPointsPerDim; iQuad2++)
          for (int iQuad1 = 0; iQuad1 < nQuadPointsPerDim; iQuad1++)
            {
              double temp;

              int quadIdx = iQuad1 + iQuad2 * nQuadPointsPerDim +
                            iQuad3 * nQuadPointsPerDim * nQuadPointsPerDim;

              if (!d_isGGA)
                temp = quadratureWeights[iQuad1] * quadratureWeights[iQuad2] *
                       quadratureWeights[iQuad3];

              d_VeffJxW[quadIdx + iCell * d_nQuadsPerCell] =
                (VeffJxW[quadIdx + iCell * d_nQuadsPerCell] +
                 VeffExtPotJxW[quadIdx + iCell * d_nQuadsPerCell]) /
                (d_isGGA ? 1 : temp);
            }

    constexpr int dim = 3;

    // Reorder VGGAJxW for matrix-free
    if (d_isGGA)
      for (auto iCell = 0; iCell < d_nCells; iCell++)
        for (auto iQuad = 0; iQuad < d_nQuadsPerCell; iQuad++)
          for (auto iDim = 0; iDim < dim; iDim++)
            d_VGGAJxW[iDim + iQuad * dim + iCell * dim * d_nQuadsPerCell] =
              VGGAJxW[iDim + iQuad * dim + iCell * dim * d_nQuadsPerCell];
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  inline void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::evalHXLDA(
    const int iCell)
  {
    matMulShapeEO<(batchSize / 8) * ndofsPerDim * ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  1,
                  false,
                  true>(arrayX,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayW);

    matMulShapeEO<(batchSize / 8) * ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  false,
                  true>(arrayW,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayX);

    matMulShapeEO<(batchSize / 8),
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim * nQuadPointsPerDim,
                  false,
                  true>(arrayX,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayW);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 1,
                 false,
                 true>(arrayW,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayZ);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 false,
                 true>(arrayW,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayY);

    matMulGradEO<(batchSize / 8),
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim * nQuadPointsPerDim,
                 false,
                 true>(arrayW,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayX);

    matMul<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim *
             nQuadPointsPerDim,
           3,
           3,
           1,
           false,
           true>(arrayX, jacobianFactor.data() + iCell * 9, arrayX);

    matMulGradEO<(batchSize / 8),
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim * nQuadPointsPerDim,
                 false>(arrayX,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW,
                        d_VeffJxW.data() + iCell * d_nQuadsPerCell);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 true,
                 false>(arrayY,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 1,
                 true,
                 false>(arrayZ,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW);

    matMulShapeEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  1,
                  false,
                  false>(arrayW,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayX);

    matMulShapeEO<(batchSize / 8) * nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  false,
                  false>(arrayX,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayW);

    matMulShapeEO<(batchSize / 8),
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim * ndofsPerDim,
                  false,
                  false>(arrayW,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayX);
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  inline void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::evalHXGGA(
    const int iCell)
  {
    matMulShapeEO<(batchSize / 8) * ndofsPerDim * ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  1,
                  false,
                  true>(arrayX,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayW);

    matMulShapeEO<(batchSize / 8) * ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  false,
                  true>(arrayW,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayX);

    matMulShapeEO<(batchSize / 8),
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim * nQuadPointsPerDim,
                  false,
                  true>(arrayX,
                        nodalShapeFunctionValuesAtQuadPointsEO.data(),
                        arrayV);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 1,
                 false,
                 true>(arrayV,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayZ);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 false,
                 true>(arrayV,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayY);

    matMulGradEO<(batchSize / 8),
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim * nQuadPointsPerDim,
                 false,
                 true>(arrayV,
                       quadShapeFunctionGradientsAtQuadPointsEO.data(),
                       arrayX);

    for (int iQuad = 0; iQuad < d_nQuadsPerCell; iQuad++)
      {
        int idx       = iQuad * 3 + iCell * 3 * d_nQuadsPerCell;
        arrayW[iQuad] = d_VGGAJxW[idx] * arrayX[iQuad] +
                        d_VGGAJxW[1 + idx] * arrayY[iQuad] +
                        d_VGGAJxW[2 + idx] * arrayZ[iQuad];
      }

    for (int iQuad3 = 0; iQuad3 < nQuadPointsPerDim; iQuad3++)
      for (int iQuad2 = 0; iQuad2 < nQuadPointsPerDim; iQuad2++)
        for (int iQuad1 = 0; iQuad1 < nQuadPointsPerDim; iQuad1++)
          {
            double temp = quadratureWeights[iQuad1] *
                          quadratureWeights[iQuad2] * quadratureWeights[iQuad3];
            int idx = iQuad1 + iQuad2 * nQuadPointsPerDim +
                      iQuad3 * nQuadPointsPerDim * nQuadPointsPerDim;
            arrayX[idx] = temp * arrayX[idx];
            arrayY[idx] = temp * arrayY[idx];
            arrayZ[idx] = temp * arrayZ[idx];
          }

    matMul<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim *
             nQuadPointsPerDim,
           3,
           3,
           1,
           false,
           true>(arrayX, jacobianFactor.data() + iCell * 9, arrayX);

    for (int iQuad = 0; iQuad < d_nQuadsPerCell; iQuad++)
      {
        int idx = iQuad * 3 + iCell * 3 * d_nQuadsPerCell;
        arrayX[iQuad] += d_VGGAJxW[idx] * arrayV[iQuad];
        arrayY[iQuad] += d_VGGAJxW[1 + idx] * arrayV[iQuad];
        arrayZ[iQuad] += d_VGGAJxW[2 + idx] * arrayV[iQuad];
      }

    for (int iQuad = 0; iQuad < d_nQuadsPerCell; iQuad++)
      arrayW[iQuad] +=
        d_VeffJxW[iQuad + iCell * d_nQuadsPerCell] * arrayV[iQuad];

    matMulGradEO<(batchSize / 8),
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim * nQuadPointsPerDim,
                 true,
                 false>(arrayX,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 true,
                 false>(arrayY,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW);

    matMulGradEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 nQuadPointsPerDim,
                 1,
                 true,
                 false>(arrayZ,
                        quadShapeFunctionGradientsAtQuadPointsEO.data(),
                        arrayW);

    matMulShapeEO<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  1,
                  false,
                  false>(arrayW,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayX);

    matMulShapeEO<(batchSize / 8) * nQuadPointsPerDim,
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim,
                  false,
                  false>(arrayX,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayW);

    matMulShapeEO<(batchSize / 8),
                  ndofsPerDim,
                  nQuadPointsPerDim,
                  ndofsPerDim * ndofsPerDim,
                  false,
                  false>(arrayW,
                         nodalShapeFunctionValuesAtQuadPointsEO.data(),
                         arrayX);
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  inline int
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::getMultiVectorIndex(
    const int nodeIdx,
    const int batchIdx) const
  {
    return (nodeIdx < d_nOwnedDofs ? (nodeIdx + batchIdx * d_nOwnedDofs) :
                                     (singleVectorToMultiVectorMap[nodeIdx] +
                                      (batchIdx % 2) * d_nGhostDofs));
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::computeAX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &Ax,
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &x,
    const double scalarHX)
  {
    constexpr unsigned int          oneInt   = 1;
    constexpr unsigned int          eightInt = batchSize;
    dealii::VectorizedArray<double> temp;

    // Start updateGhost for batch 0
    d_singleBatchPartitioner->export_to_ghosted_array_start<double>(
      (unsigned int)5,
      dealii::ArrayView<const double>(x.data(), batchSize * d_nOwnedDofs),
      dealii::ArrayView<double>(tempGhostStorage),
      dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize,
                                batchSize * d_nGhostDofs),
      mpiRequestsGhost);

    // End updateGhost for batch 0
    d_singleBatchPartitioner->export_to_ghosted_array_finish<double>(
      dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize,
                                batchSize * d_nGhostDofs),
      mpiRequestsGhost);

    // Batch Loop
    for (int iBatch = 0; iBatch < d_nBatch; iBatch++)
      {
        // Use GPU overlap 1st tensor contraction and extraction

        // Overlap batches
        if (iBatch < d_nBatch - 1)
          d_singleBatchPartitioner->export_to_ghosted_array_start<double>(
            (unsigned int)5,
            dealii::ArrayView<const double>(x.data() + (iBatch + 1) *
                                                         batchSize *
                                                         d_nOwnedDofs,
                                            batchSize * d_nOwnedDofs),
            dealii::ArrayView<double>(tempGhostStorage),
            dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize +
                                        d_nGhostDofs * batchSize *
                                          ((iBatch + 1) % 2),
                                      batchSize * d_nGhostDofs),
            mpiRequestsGhost);

        // Overlap batches
        if (iBatch > 0)
          d_singleBatchPartitioner->import_from_ghosted_array_start(
            dealii::VectorOperation::add,
            0,
            dealii::ArrayView<double>(Ax.data() + d_nOwnedDofs * d_blockSize +
                                        d_nGhostDofs * batchSize *
                                          ((iBatch - 1) % 2),
                                      batchSize * d_nGhostDofs),
            dealii::ArrayView<double>(tempCompressStorage),
            mpiRequestsCompress);

        // Constraints distribute
        // Optimize masterNodeBuckets[i].size() from GPUs shared size
        for (auto i = 0; i < masterNodeBuckets.size(); i++)
          {
            double inhomogenity = inhomogenityList[i];

            dealii::AlignedVector<dealii::VectorizedArray<double>>
              tempMasterData(masterNodeBuckets[i].size());

            for (auto k = 0; k < masterNodeBuckets[i].size(); k++)
              tempMasterData[k].load(
                x.data() +
                getMultiVectorIndex(masterNodeBuckets[i][k], iBatch) *
                  batchSize);

            for (auto j = 0; j < slaveNodeBuckets[i].size(); j++)
              {
                temp = inhomogenity;

                for (auto k = 0; k < masterNodeBuckets[i].size(); k++)
                  temp +=
                    weightMatrixList[i][k + j * masterNodeBuckets[i].size()] *
                    tempMasterData[k];

                int dofIdx =
                  getMultiVectorIndex(slaveNodeBuckets[i][j], iBatch);

                temp.store(x.data() + dofIdx * batchSize);
              }
          }

        // Cell Loop
        for (int iCell = 0; iCell < d_nCells; iCell++)
          {
            // Extraction
            for (int iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
              {
                int dofIdx =
                  singleVectorGlobalToLocalMap[iDoF + iCell * d_nDofsPerCell];

                // Possibly optimized
                std::memcpy(arrayX + iDoF,
                            x.data() +
                              getMultiVectorIndex(dofIdx, iBatch) * batchSize,
                            batchSize * sizeof(double));
              }

            if (d_isGGA)
              evalHXGGA(iCell);
            else
              evalHXLDA(iCell);

            // Assembly
            for (auto iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
              {
                int dofIdx =
                  singleVectorGlobalToLocalMap[iDoF + iCell * d_nDofsPerCell];

                arrayX[iDoF] =
                  cellInverseMassVector[iDoF + iCell * d_nDofsPerCell] *
                  arrayX[iDoF];

                // Potential bottleneck
                daxpy_(&eightInt,
                       &scalarHX,
                       (double *)(arrayX + iDoF),
                       &oneInt,
                       Ax.data() +
                         getMultiVectorIndex(dofIdx, iBatch) * batchSize,
                       &oneInt);
              }
          }

        // Constraints distribute transpose
        for (auto i = 0; i < slaveNodeBuckets.size(); i++)
          {
            if (masterNodeBuckets[i].size() > 0)
              {
                dealii::AlignedVector<dealii::VectorizedArray<double>>
                  tempSlaveData(slaveNodeBuckets[i].size());

                for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
                  {
                    tempSlaveData[k].load(
                      Ax.data() +
                      getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                        batchSize);

                    std::fill(
                      Ax.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize,
                      Ax.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize +
                        batchSize,
                      0.0);

                    std::fill(
                      x.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize,
                      x.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize +
                        batchSize,
                      0.0);
                  }

                for (auto j = 0; j < masterNodeBuckets[i].size(); j++)
                  {
                    temp.load(
                      Ax.data() +
                      getMultiVectorIndex(masterNodeBuckets[i][j], iBatch) *
                        batchSize);

                    for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
                      temp +=
                        scaledWeightMatrixList[i][j + k * masterNodeBuckets[i]
                                                            .size()] *
                        tempSlaveData[k];

                    temp.store(
                      Ax.data() +
                      getMultiVectorIndex(masterNodeBuckets[i][j], iBatch) *
                        batchSize);
                  }
              }
            else
              {
                for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
                  {
                    std::fill(
                      Ax.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize,
                      Ax.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize +
                        batchSize,
                      0.0);

                    std::fill(
                      x.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize,
                      x.data() +
                        getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch) *
                          batchSize +
                        batchSize,
                      0.0);
                  }
              }
          }

        if (iBatch > 0)
          d_singleBatchPartitioner->import_from_ghosted_array_finish(
            dealii::VectorOperation::add,
            dealii::ArrayView<const double>(tempCompressStorage),
            dealii::ArrayView<double>(Ax.data() +
                                        (iBatch - 1) * batchSize * d_nOwnedDofs,
                                      batchSize * d_nOwnedDofs),
            dealii::ArrayView<double>(Ax.data() + d_nOwnedDofs * d_blockSize +
                                        d_nGhostDofs * batchSize *
                                          ((iBatch - 1) % 2),
                                      batchSize * d_nGhostDofs),
            mpiRequestsCompress);

        if (iBatch < d_nBatch - 1)
          d_singleBatchPartitioner->export_to_ghosted_array_finish<double>(
            dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize +
                                        d_nGhostDofs * batchSize *
                                          ((iBatch + 1) % 2),
                                      batchSize * d_nGhostDofs),
            mpiRequestsGhost);
      }

    d_singleBatchPartitioner->import_from_ghosted_array_start(
      dealii::VectorOperation::add,
      0,
      dealii::ArrayView<double>(Ax.data() + d_nOwnedDofs * d_blockSize +
                                  d_nGhostDofs * batchSize *
                                    ((d_nBatch - 1) % 2),
                                batchSize * d_nGhostDofs),
      dealii::ArrayView<double>(tempCompressStorage),
      mpiRequestsCompress);

    d_singleBatchPartitioner->import_from_ghosted_array_finish(
      dealii::VectorOperation::add,
      dealii::ArrayView<const double>(tempCompressStorage),
      dealii::ArrayView<double>(Ax.data() +
                                  (d_nBatch - 1) * batchSize * d_nOwnedDofs,
                                batchSize * d_nOwnedDofs),
      dealii::ArrayView<double>(Ax.data() + d_nOwnedDofs * d_blockSize +
                                  d_nGhostDofs * batchSize *
                                    ((d_nBatch - 1) % 2),
                                batchSize * d_nGhostDofs),
      mpiRequestsCompress);

    // Compare with memsets
    x.zeroOutGhosts();
    Ax.zeroOutGhosts();
  }


  template <int ndofsPerDim, int nQuadPointsPerDim, int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::computeAX(
    dealii::VectorizedArray<double> *Ax,
    dealii::VectorizedArray<double> *x,
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST>
      &          d_ONCVNonLocalProjectorTimesVectorBlock,
    const double scalarHX,
    const bool   hasNonlocalComponents)
  {
    dealii::VectorizedArray<double> temp;

    // Start updateGhost for batch 0
    // d_singleBatchPartitioner->export_to_ghosted_array_start<double>(
    //   (unsigned int)5,
    //   dealii::ArrayView<const double>(x.data(), batchSize * d_nOwnedDofs),
    //   dealii::ArrayView<double>(tempGhostStorage),
    //   dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize,
    //                             batchSize * d_nGhostDofs),
    //   mpiRequestsGhost);

    // // End updateGhost for batch 0
    // d_singleBatchPartitioner->export_to_ghosted_array_finish<double>(
    //   dealii::ArrayView<double>(x.data() + d_nOwnedDofs * d_blockSize,
    //                             batchSize * d_nGhostDofs),
    //   mpiRequestsGhost);

    // Batch Loop
    for (int iBatch = 0; iBatch < d_nBatch; iBatch++)
      {
        d_singleBatchPartitioner->export_to_ghosted_array_start<double>(
          (unsigned int)5,
          dealii::ArrayView<const double>((double *)x +
                                            iBatch * batchSize * d_nOwnedDofs,
                                          batchSize * d_nOwnedDofs),
          dealii::ArrayView<double>(tempGhostStorage),
          dealii::ArrayView<double>((double *)x +
                                      batchSize * d_nOwnedDofs * d_nBatch +
                                      (iBatch % 2) * batchSize * d_nGhostDofs,
                                    batchSize * d_nGhostDofs),
          mpiRequestsGhost);

        d_singleBatchPartitioner->export_to_ghosted_array_finish<double>(
          dealii::ArrayView<double>((double *)x +
                                      batchSize * d_nOwnedDofs * d_nBatch +
                                      d_nGhostDofs * batchSize * (iBatch % 2),
                                    batchSize * d_nGhostDofs),
          mpiRequestsGhost);

        // Constraints distribute
        for (auto i = 0; i < masterNodeBuckets.size(); i++)
          {
            double inhomogenity = inhomogenityList[i];

            dealii::AlignedVector<dealii::VectorizedArray<double>>
              tempMasterData(masterNodeBuckets[i].size());

            for (auto k = 0; k < masterNodeBuckets[i].size(); k++)
              tempMasterData[k] =
                x[getMultiVectorIndex(masterNodeBuckets[i][k], iBatch)];

            for (auto j = 0; j < slaveNodeBuckets[i].size(); j++)
              {
                temp = inhomogenity;

                for (auto k = 0; k < masterNodeBuckets[i].size(); k++)
                  temp +=
                    weightMatrixList[i][k + j * masterNodeBuckets[i].size()] *
                    tempMasterData[k];

                int dofIdx =
                  getMultiVectorIndex(slaveNodeBuckets[i][j], iBatch);

                x[dofIdx] = temp;
              }
          }

        if (hasNonlocalComponents)
          {
            d_ONCVnonLocalOperator->initialiseOperatorActionOnXMF(
              d_kPointIndex);

            // Cell Loop
            for (int iCell = 0; iCell < d_nCells; iCell++)
              {
                if (d_ONCVnonLocalOperator->atomSupportInElement(iCell))
                  {
                    // Extraction
                    for (int iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
                      {
                        int dofIdx =
                          singleVectorGlobalToLocalMap[iDoF +
                                                       iCell * d_nDofsPerCell];

                        arrayX[iDoF] = x[getMultiVectorIndex(dofIdx, iBatch)];
                      }

                    d_ONCVnonLocalOperator->applyCconjtransOnXMF(
                      arrayX, d_CMatrixEntriesConjugate, iCell);
                  }
              }

            d_ONCVNonLocalProjectorTimesVectorBlock.setValue(0);

            d_ONCVnonLocalOperator->applyAllReduceOnCconjtransXMF(
              d_ONCVNonLocalProjectorTimesVectorBlock, true);

            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedBegin();
            d_ONCVNonLocalProjectorTimesVectorBlock
              .accumulateAddLocallyOwnedEnd();
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesBegin();
            d_ONCVNonLocalProjectorTimesVectorBlock.updateGhostValuesEnd();

            d_ONCVnonLocalOperator->applyVOnCconjtransXMF(
              CouplingStructure::diagonal,
              d_oncvClassPtr->getCouplingMatrix(),
              d_ONCVNonLocalProjectorTimesVectorBlock,
              true);
          }

        // Cell Loop
        for (int iCell = 0; iCell < d_nCells; iCell++)
          {
            // Extraction
            for (int iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
              {
                int dofIdx =
                  singleVectorGlobalToLocalMap[iDoF + iCell * d_nDofsPerCell];

                arrayX[iDoF] = x[getMultiVectorIndex(dofIdx, iBatch)];
              }

            if (d_isGGA)
              evalHXGGA(iCell);
            else
              evalHXLDA(iCell);

            if (hasNonlocalComponents)
              d_ONCVnonLocalOperator->applyCOnVCconjtransXMF(
                arrayX, d_CMatrixEntriesTranspose, iCell);

            // Assembly
            for (auto iDoF = 0; iDoF < d_nDofsPerCell; iDoF++)
              {
                auto dofIdx =
                  singleVectorGlobalToLocalMap[iDoF + iCell * d_nDofsPerCell];

                Ax[getMultiVectorIndex(dofIdx, iBatch)] +=
                  scalarHX *
                  cellInverseMassVector[iDoF + iCell * d_nDofsPerCell] *
                  arrayX[iDoF];
              }
          }

        // Constraints distribute transpose
        for (auto i = 0; i < slaveNodeBuckets.size(); i++)
          {
            if (masterNodeBuckets[i].size() > 0)
              {
                dealii::AlignedVector<dealii::VectorizedArray<double>>
                  tempSlaveData(slaveNodeBuckets[i].size());

                for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
                  tempSlaveData[k] =
                    Ax[getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch)];

                for (auto j = 0; j < masterNodeBuckets[i].size(); j++)
                  {
                    temp =
                      Ax[getMultiVectorIndex(masterNodeBuckets[i][j], iBatch)];

                    for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
                      temp +=
                        scaledWeightMatrixList[i][j + k * masterNodeBuckets[i]
                                                            .size()] *
                        tempSlaveData[k];

                    Ax[getMultiVectorIndex(masterNodeBuckets[i][j], iBatch)] =
                      temp;
                  }
              }

            for (auto k = 0; k < slaveNodeBuckets[i].size(); k++)
              {
                Ax[getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch)] = 0.0;
                x[getMultiVectorIndex(slaveNodeBuckets[i][k], iBatch)]  = 0.0;
              }
          }

        d_singleBatchPartitioner->import_from_ghosted_array_start(
          dealii::VectorOperation::add,
          0,
          dealii::ArrayView<double>((double *)Ax +
                                      batchSize * d_nOwnedDofs * d_nBatch +
                                      d_nGhostDofs * batchSize * (iBatch % 2),
                                    batchSize * d_nGhostDofs),
          dealii::ArrayView<double>(tempCompressStorage),
          mpiRequestsCompress);

        d_singleBatchPartitioner->import_from_ghosted_array_finish(
          dealii::VectorOperation::add,
          dealii::ArrayView<const double>(tempCompressStorage),
          dealii::ArrayView<double>((double *)Ax +
                                      iBatch * batchSize * d_nOwnedDofs,
                                    batchSize * d_nOwnedDofs),
          dealii::ArrayView<double>((double *)Ax +
                                      batchSize * d_nOwnedDofs * d_nBatch +
                                      d_nGhostDofs * batchSize * (iBatch % 2),
                                    batchSize * d_nGhostDofs),
          mpiRequestsCompress);
      }

    // Compare with memsets
    std::fill(Ax + d_nOwnedDofs * d_nBatch,
              Ax + d_nOwnedDofs * d_nBatch + 2 * d_nGhostDofs,
              0.0);
    std::fill(x + d_nOwnedDofs * d_nBatch,
              x + d_nOwnedDofs * d_nBatch + 2 * d_nGhostDofs,
              0.0);
  }

#include "MatrixFree.inst.cc"
} // namespace dftfe
