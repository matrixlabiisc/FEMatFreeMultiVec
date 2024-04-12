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
  template <int m, int n, int k, int c, bool add, bool trans, int type>
  inline void
  matmulEOdealii(const dealii::VectorizedArray<double> *A,
                 const double *                         B,
                 dealii::VectorizedArray<double> *      C)
  {
    if (type == 2)
      {
        constexpr int ko = k / 2;
        constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
        constexpr int no = n / 2;
        constexpr int ne = n % 2 == 1 ? no + 1 : no;
        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];
                for (auto q = 0; q < ko; ++q)
                  {
                    tempAe[q] = A[i + q * m + m * k * b] +
                                A[i + (k - 1 - q) * m + m * k * b];
                    tempAo[q] = A[i + q * m + m * k * b] -
                                A[i + (k - 1 - q) * m + m * k * b];
                  }
                if (k % 2 == 1)
                  tempAe[ko] = A[i + ko * m + m * k * b];
                for (auto j = 0; j < no; ++j)
                  {
                    dealii::VectorizedArray<double> tempCe, tempCo;
                    if (trans)
                      tempCe = tempAe[0] * B[j];
                    else
                      tempCe = tempAe[0] * B[j * ke + ko * ne];
                    for (auto q = 1; q < ke; ++q)
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
                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          tempCo += tempAo[q] * B[j + q * ne + ke * no];
                        else
                          tempCo += tempAo[q] * B[q + j * ko];
                      }
                    // if (k % 2 == 1)
                    //   {
                    //     if (trans)
                    //       tempCe += tempAe[ko] * B[j + ko * no];
                    //     else
                    //       tempCe += tempAe[ko] * B[ko + j * ke + ko * ne];
                    //   }
                    if (add)
                      {
                        C[i + m * j + m * n * b] += tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] += tempCo - tempCe;
                      }
                    else
                      {
                        C[i + m * j + m * n * b]           = tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] = tempCo - tempCe;
                      }
                  }
                if (n % 2 == 1)
                  {
                    if (add)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] +=
                            tempAo[0] * B[no + ke * no];
                        else
                          C[i + m * no + m * n * b] += tempAo[0] * B[no * ko];
                      }
                    else
                      {
                        if (trans)
                          C[i + m * no + m * n * b] =
                            tempAo[0] * B[no + ke * no];
                        else
                          C[i + m * no + m * n * b] = tempAo[0] * B[no * ko];
                      }

                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] +=
                            tempAo[q] * B[no + q * ne + ke * no];
                        else
                          C[i + m * no + m * n * b] +=
                            tempAo[q] * B[no * ko + q];
                      }
                  }
              }
          }
      }
    else if (type == 1)
      {
        constexpr int ko = k / 2;
        constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
        constexpr int no = n / 2;
        constexpr int ne = n % 2 == 1 ? no + 1 : no;


        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];
                for (auto q = 0; q < ko; ++q)
                  {
                    tempAe[q] = A[i + q * m + m * k * b] +
                                A[i + (k - 1 - q) * m + m * k * b];
                    tempAo[q] = A[i + q * m + m * k * b] -
                                A[i + (k - 1 - q) * m + m * k * b];
                  }
                if (k % 2 == 1)
                  tempAe[ko] = A[i + ko * m + m * k * b];
                for (auto j = 0; j < no; ++j)
                  {
                    dealii::VectorizedArray<double> tempCe, tempCo;
                    if (trans)
                      tempCe = tempAe[0] * B[j];
                    else
                      tempCe = tempAe[0] * B[j * ke];
                    for (auto q = 1; q < ke; ++q)
                      {
                        if (trans)
                          tempCe += tempAe[q] * B[j + q * ne];
                        else
                          tempCe += tempAe[q] * B[q + j * ke];
                      }
                    //   }
                    // for (auto j = 0; j < no; ++j)
                    //   {
                    if (trans)
                      tempCo = tempAo[0] * B[j + ke * ne];
                    else
                      tempCo = tempAo[0] * B[j * ko + ke * ne];
                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          tempCo += tempAo[q] * B[j + q * no + ke * ne];
                        else
                          tempCo += tempAo[q] * B[q + j * ko + ke * ne];
                      }
                    //   }
                    // for (auto j = 0; j < no; ++j)
                    //   {
                    if (add)
                      {
                        C[i + m * j + m * n * b] += tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] += tempCe - tempCo;
                      }
                    else
                      {
                        C[i + m * j + m * n * b]           = tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] = tempCe - tempCo;
                      }
                  }
                if (n % 2 == 1)
                  {
                    if (add)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] += tempAe[0] * B[no];
                        else
                          C[i + m * no + m * n * b] += tempAe[0] * B[no * ke];
                      }
                    else
                      {
                        if (trans)
                          C[i + m * no + m * n * b] = tempAe[0] * B[no];
                        else
                          C[i + m * no + m * n * b] = tempAe[0] * B[no * ke];
                      }

                    for (auto q = 1; q < ke; ++q)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] +=
                            tempAe[q] * B[no + q * ne];
                        else
                          C[i + m * no + m * n * b] +=
                            tempAe[q] * B[no * ke + q];
                      }
                  }
              }
          }
      }
    else if (type == 0)
      {
        dealii::VectorizedArray<double> tempA[k];
        dealii::VectorizedArray<double> temp;
        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                for (auto q = 0; q < k; ++q)
                  {
                    // tempA[q].load(A+i+q*m+m*k*b);
                    tempA[q] = A[i + q * m + m * k * b];
                  }
                for (auto j = 0; j < n; ++j)
                  {
                    // if(add)
                    //   {
                    //     // temp.load(C+i+m*j+m*n*b);
                    //     // temp+= trans ? tempA[0]*B[j]: tempA[0]*B[j*k];
                    //     temp=C[i+m*j+m*n*b] + (trans ? tempA[0]*B[j]:
                    //     tempA[0]*B[j*k]);
                    //   }
                    // else
                    //   temp=trans ? tempA[0]*B[j]: tempA[0]*B[j*k];
                    temp = trans ? tempA[0] * B[j] : tempA[0] * B[j * k];
                    // C[i+m*j]=A[i]*B[j];
                    for (auto q = 1; q < k; ++q)
                      {
                        // tempA.load(A+i+q*m);
                        if (trans)
                          temp += tempA[q] * B[j + q * n];
                        else
                          temp += tempA[q] * B[q + j * k];
                      }
                    // temp.store(C+i+m*j+m*n*b);
                    if (add)
                      C[i + m * j + m * n * b] += temp;
                    else
                      C[i + m * j + m * n * b] = temp;
                    // if(add)
                    // C[i+m*j]+=temp;
                    // else
                    // C[i+m*j]=temp;
                  }
              }
            //  A += m * k;
            //  C += m * n;
          }
      }
    // constexpr int c=dealii::Utilities::pow(k, (2 - direction));
    // constexpr int m=dealii::Utilities::pow(k, direction)*batchSize;
    //   for (auto b = 0; b < c; ++b)
    //   {
    //       for(auto j=0;j<n;++j)
    //       {
    //         for(auto i=0;i<m;++i)
    //         {
    //             C[i+m*j]=A[i]*B[j];
    //         }
    //       }
    //     for(auto q=1;q<k;++q)
    //     {
    //       for(auto j=0;j<n;++j)
    //       {
    //         for(auto i=0;i<m;++i)
    //         {
    //             C[i+m*j]+=A[i+q*m]*B[j+q*n];
    //         }
    //       }
    //     }
    //      A += m * k;
    //      C += m * n;
    //   }
  }

  template <int m, int n, int k, int c, bool constcoeff, bool trans, int type>
  inline void
  matmulEOdealii(const dealii::VectorizedArray<double> *A,
                 const double *                         B,
                 dealii::VectorizedArray<double> *      C,
                 const double *                         coeffs)
  {
    if (type == 2)
      {
        constexpr int ko = k / 2;
        constexpr int ke = k % 2 == 1 ? ko + 1 : ko;
        constexpr int no = n / 2;
        constexpr int ne = n % 2 == 1 ? no + 1 : no;
        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];
                for (auto q = 0; q < ko; ++q)
                  {
                    tempAe[q] = A[i + q * m + m * k * b] +
                                A[i + (k - 1 - q) * m + m * k * b];
                    tempAo[q] = A[i + q * m + m * k * b] -
                                A[i + (k - 1 - q) * m + m * k * b];
                  }
                if (k % 2 == 1)
                  tempAe[ko] = A[i + ko * m + m * k * b];
                for (auto j = 0; j < no; ++j)
                  {
                    dealii::VectorizedArray<double> tempCe, tempCo;
                    if (trans)
                      tempCe = tempAe[0] * B[j];
                    else
                      tempCe = tempAe[0] * B[j * ke + ko * ne];
                    for (auto q = 1; q < ke; ++q)
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
                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          tempCo += tempAo[q] * B[j + q * ne + ke * no];
                        else
                          tempCo += tempAo[q] * B[q + j * ko];
                      }
                    // if (k % 2 == 1)
                    //   {
                    //     if (trans)
                    //       tempCe += tempAe[ko] * B[j + ko * no];
                    //     else
                    //       tempCe += tempAe[ko] * B[ko + j * ke + ko * ne];
                    //   }
                    if (constcoeff)
                      {
                        C[i + m * j + m * n * b] =
                          C[i + m * j + m * n * b] * coeffs[0] + tempCe +
                          tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] =
                          C[i + m * (n - 1 - j) + m * n * b] * coeffs[0] +
                          tempCo - tempCe;
                      }
                    else
                      {
                        C[i + m * j + m * n * b] =
                          C[i + m * j + m * n * b] * coeffs[i + m * j] +
                          tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] =
                          C[i + m * (n - 1 - j) + m * n * b] *
                            coeffs[i + m * (n - 1 - j)] +
                          tempCo - tempCe;
                      }
                  }
                if (n % 2 == 1)
                  {
                    if (constcoeff)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[0] +
                            tempAo[0] * B[no + ke * no];
                        else
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[0] +
                            tempAo[0] * B[no * ko];
                      }
                    else
                      {
                        if (trans)
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[i + m * no] +
                            tempAo[0] * B[no + ke * no];
                        else
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[i + m * no] +
                            tempAo[0] * B[no * ko];
                      }

                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] +=
                            tempAo[q] * B[no + q * ne + ke * no];
                        else
                          C[i + m * no + m * n * b] +=
                            tempAo[q] * B[no * ko + q];
                      }
                  }
              }
          }
      }
    else if (type == 1)
      {
        constexpr int                   ko = k / 2;
        constexpr int                   ke = k % 2 == 1 ? ko + 1 : ko;
        constexpr int                   no = n / 2;
        constexpr int                   ne = n % 2 == 1 ? no + 1 : no;
        dealii::VectorizedArray<double> tempAe[ke], tempAo[ko];
        dealii::VectorizedArray<double> tempCe, tempCo;
        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                for (auto q = 0; q < ko; ++q)
                  {
                    tempAe[q] = A[i + q * m + m * k * b] +
                                A[i + (k - 1 - q) * m + m * k * b];
                    tempAo[q] = A[i + q * m + m * k * b] -
                                A[i + (k - 1 - q) * m + m * k * b];
                  }
                if (k % 2 == 1)
                  tempAe[ko] = A[i + ko * m + m * k * b];
                for (auto j = 0; j < no; ++j)
                  {
                    if (trans)
                      tempCe = tempAe[0] * B[j];
                    else
                      tempCe = tempAe[0] * B[j * ke];
                    for (auto q = 1; q < ke; ++q)
                      {
                        if (trans)
                          tempCe += tempAe[q] * B[j + q * ne];
                        else
                          tempCe += tempAe[q] * B[q + j * ke];
                      }
                    //   }
                    // for (auto j = 0; j < no; ++j)
                    //   {
                    if (trans)
                      tempCo = tempAo[0] * B[j + ke * ne];
                    else
                      tempCo = tempAo[0] * B[j * ko + ke * ne];
                    for (auto q = 1; q < ko; ++q)
                      {
                        if (trans)
                          tempCo += tempAo[q] * B[j + q * no + ke * ne];
                        else
                          tempCo += tempAo[q] * B[q + j * ko + ke * ne];
                      }
                    //   }
                    // for (auto j = 0; j < no; ++j)
                    //   {
                    if (constcoeff)
                      {
                        C[i + m * j + m * n * b] =
                          C[i + m * j + m * n * b] * coeffs[0] + tempCe +
                          tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] =
                          C[i + m * (n - 1 - j) + m * n * b] * coeffs[0] +
                          tempCe - tempCo;
                      }
                    else
                      {
                        C[i + m * j + m * n * b] =
                          C[i + m * j + m * n * b] * coeffs[i + m * j] +
                          tempCe + tempCo;
                        C[i + m * (n - 1 - j) + m * n * b] =
                          C[i + m * (n - 1 - j) + m * n * b] *
                            coeffs[i + m * (n - 1 - j)] +
                          tempCe - tempCo;
                      }
                  }
                if (n % 2 == 1)
                  {
                    if (constcoeff)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[0] +
                            tempAe[0] * B[no];
                        else
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[0] +
                            tempAe[0] * B[no * ko];
                      }
                    else
                      {
                        if (trans)
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[i + m * no] +
                            tempAe[0] * B[no];
                        else
                          C[i + m * no + m * n * b] =
                            C[i + m * no + m * n * b] * coeffs[i + m * no] +
                            tempAe[0] * B[no * ke];
                      }

                    for (auto q = 1; q < ke; ++q)
                      {
                        if (trans)
                          C[i + m * no + m * n * b] +=
                            tempAe[q] * B[no + q * ne];
                        else
                          C[i + m * no + m * n * b] +=
                            tempAe[q] * B[no * ke + q];
                      }
                  }
              }
          }
      }
    else if (type == 0)
      {
        dealii::VectorizedArray<double> tempA[k];
        dealii::VectorizedArray<double> temp;
        for (auto b = 0; b < c; ++b)
          {
            for (auto i = 0; i < m; ++i)
              {
                for (auto q = 0; q < k; ++q)
                  {
                    // tempA[q].load(A+i+q*m+m*k*b);
                    tempA[q] = A[i + q * m + m * k * b];
                  }
                for (auto j = 0; j < n; ++j)
                  {
                    // if(add)
                    //   {
                    //     // temp.load(C+i+m*j+m*n*b);
                    //     // temp+= trans ? tempA[0]*B[j]: tempA[0]*B[j*k];
                    //     temp=C[i+m*j+m*n*b] + (trans ? tempA[0]*B[j]:
                    //     tempA[0]*B[j*k]);
                    //   }
                    // else
                    //   temp=trans ? tempA[0]*B[j]: tempA[0]*B[j*k];
                    temp = trans ? tempA[0] * B[j] : tempA[0] * B[j * k];
                    // C[i+m*j]=A[i]*B[j];
                    for (auto q = 1; q < k; ++q)
                      {
                        // tempA.load(A+i+q*m);
                        if (trans)
                          temp += tempA[q] * B[j + q * n];
                        else
                          temp += tempA[q] * B[q + j * k];
                      }
                    // temp.store(C+i+m*j+m*n*b);
                    if (constcoeff)
                      C[i + m * j + m * n * b] =
                        C[i + m * j + m * n * b] * coeffs[0] + temp;
                    else
                      C[i + m * j + m * n * b] =
                        C[i + m * j + m * n * b] * coeffs[i + m * j] + temp;
                    // if(add)
                    // C[i+m*j]+=temp;
                    // else
                    // C[i+m*j]=temp;
                  }
              }
            //  A += m * k;
            //  C += m * n;
          }
      }
    // constexpr int c=dealii::Utilities::pow(k, (2 - direction));
    // constexpr int m=dealii::Utilities::pow(k, direction)*batchSize;
    //   for (auto b = 0; b < c; ++b)
    //   {
    //       for(auto j=0;j<n;++j)
    //       {
    //         for(auto i=0;i<m;++i)
    //         {
    //             C[i+m*j]=A[i]*B[j];
    //         }
    //       }
    //     for(auto q=1;q<k;++q)
    //     {
    //       for(auto j=0;j<n;++j)
    //       {
    //         for(auto i=0;i<m;++i)
    //         {
    //             C[i+m*j]+=A[i+q*m]*B[j+q*n];
    //         }
    //       }
    //     }
    //      A += m * k;
    //      C += m * n;
    //   }
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::MatrixFree(
    const MPI_Comm &mpi_comm,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
                       basisOperationsPtrHost,
    const unsigned int blockSize)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_nBatch(ceil((double)blockSize / (double)batchSize))
    , d_blockSize(blockSize)
  {}

  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::reinit(
    const unsigned int matrixfreeQuadratureID)
  {
    d_basisOperationsPtrHost->reinit(0, 0, matrixfreeQuadratureID);

    const dealii::MatrixFree<3, double> &matrixFreeData =
      d_basisOperationsPtrHost->matrixFreeData();
    auto dofInfo =
      matrixFreeData.get_dof_info(d_basisOperationsPtrHost->d_dofHandlerID);
    auto shapeInfo =
      matrixFreeData.get_shape_info(d_basisOperationsPtrHost->d_dofHandlerID,
                                    matrixfreeQuadratureID);
    auto mappingData =
      matrixFreeData.get_mapping_info().cell_data[matrixfreeQuadratureID];
    auto shapeData = shapeInfo.get_shape_data();

    d_nOwnedDofs         = d_basisOperationsPtrHost->nOwnedDofs();
    d_nRelaventDofs      = d_basisOperationsPtrHost->nRelaventDofs();
    d_nQuadsPerCell      = d_basisOperationsPtrHost->nQuadsPerCell();
    d_nCells             = d_basisOperationsPtrHost->nCells();
    d_nDofsPerCell       = d_basisOperationsPtrHost->nDofsPerCell();
    d_nGhostDofs         = d_nRelaventDofs - d_nOwnedDofs;
    d_batchedPartitioner = matrixFreeData.get_vector_partitioner(
      d_basisOperationsPtrHost->d_dofHandlerID);

    singleVectorGlobalToLocalMap.resize(d_nCells * d_nDofsPerCell, 0);

    for (auto iCellBatch = 0, iCell = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         ++iCellBatch)
      {
        for (auto iCellLocal = 0;
             iCellLocal < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
             ++iCellLocal, ++iCell)
          {
            std::memcpy(
              singleVectorGlobalToLocalMap.data() + iCell * d_nDofsPerCell,
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
              d_nDofsPerCell * sizeof(unsigned int));
          }
      }

    // singleVectorToMultiVectorMap.resize(d_nRelaventDofs, 0);
    globalToLocalMap.resize(d_nDofsPerCell * d_nCells, 0);

    auto taskGhostMap =
      matrixFreeData
        .get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->ghost_targets();

    std::vector<unsigned int> taskGhostStartIndices(n_mpi_processes, 0);

    for (auto i = 0; i < taskGhostMap.size(); ++i)
      {
        taskGhostStartIndices[taskGhostMap[i].first] = taskGhostMap[i].second;
      }

    auto ghostSum = 0;
    for (auto i = 0; i < taskGhostStartIndices.size(); ++i)
      {
        auto tmp = ghostSum;
        ghostSum += taskGhostStartIndices[i];
        taskGhostStartIndices[i] = tmp;
      }

    for (auto iCell = 0; iCell < d_nCells; ++iCell)
      {
        for (auto iLDoF = 0; iLDoF < d_nDofsPerCell; ++iLDoF)
          {
            auto l2g =
              singleVectorGlobalToLocalMap[iLDoF + d_nDofsPerCell * iCell];
            if (l2g >= d_nOwnedDofs)
              {
                auto ownerId = 0;
                while (taskGhostStartIndices[ownerId] <= l2g - d_nOwnedDofs)
                  {
                    ++ownerId;
                    if (ownerId == n_mpi_processes)
                      break;
                  }

                --ownerId;
                auto ghostIdFromOwner =
                  l2g - taskGhostStartIndices[ownerId] - d_nOwnedDofs;
                auto nGhostsFromOwner =
                  ownerId == n_mpi_processes - 1 ?
                    d_batchedPartitioner->n_ghost_indices() -
                      taskGhostStartIndices[ownerId] :
                    taskGhostStartIndices[ownerId + 1] -
                      taskGhostStartIndices[ownerId];

                globalToLocalMap[iLDoF + d_nDofsPerCell * iCell] =
                  (d_nOwnedDofs * (d_blockSize / batchSize) +
                   taskGhostStartIndices[ownerId]) +
                  ghostIdFromOwner;
              }
            else
              {
                globalToLocalMap[iLDoF + d_nDofsPerCell * iCell] = l2g;
              }
          }
      }

    tempStorageVectorized.resize(
      6 * (batchSize / dofInfo.vectorization_length) * d_nQuadsPerCell);
    temp10v = tempStorageVectorized.data();
    temp11v =
      temp10v + d_nQuadsPerCell * (batchSize / dofInfo.vectorization_length);
    temp12v =
      temp11v + d_nQuadsPerCell * (batchSize / dofInfo.vectorization_length);
    temp20v =
      temp12v + d_nQuadsPerCell * (batchSize / dofInfo.vectorization_length);
    temp21v =
      temp20v + d_nQuadsPerCell * (batchSize / dofInfo.vectorization_length);
    temp22v =
      temp21v + d_nQuadsPerCell * (batchSize / dofInfo.vectorization_length);

    nodalShapeFunctionValuesAtQuadPoints = tempStorage.data();
    temp10 =
      nodalShapeFunctionValuesAtQuadPoints + ndofsPerDim * nQuadPointsPerDim;
    temp11 = temp10 + d_nQuadsPerCell * batchSize;
    temp12 = temp11 + d_nQuadsPerCell * batchSize;
    quadShapeFunctionGradientsAtQuadPoints =
      temp12 + d_nQuadsPerCell * batchSize;
    temp20 = quadShapeFunctionGradientsAtQuadPoints +
             nQuadPointsPerDim * nQuadPointsPerDim;
    temp21 = temp20 + d_nQuadsPerCell * batchSize;
    temp22 = temp21 + d_nQuadsPerCell * batchSize;

    for (auto iDoF = 0; iDoF < ndofsPerDim; ++iDoF)
      {
        for (auto iQuad = 0; iQuad < nQuadPointsPerDim; ++iQuad)
          {
            nodalShapeFunctionValuesAtQuadPoints[iDoF * nQuadPointsPerDim +
                                                 iQuad] =
              shapeData.shape_values[iDoF * nQuadPointsPerDim + iQuad][0] *
              std::sqrt(shapeData.quadrature.weight(iQuad));
          }
      }

    for (auto iQuad1 = 0; iQuad1 < nQuadPointsPerDim; ++iQuad1)
      {
        for (auto iQuad2 = 0; iQuad2 < nQuadPointsPerDim; ++iQuad2)
          {
            quadShapeFunctionGradientsAtQuadPoints[iQuad1 * nQuadPointsPerDim +
                                                   iQuad2] =
              shapeData.shape_gradients_collocation[iQuad1 * nQuadPointsPerDim +
                                                    iQuad2][0] *
              std::sqrt(shapeData.quadrature.weight(iQuad2)) /
              std::sqrt(shapeData.quadrature.weight(iQuad1));
          }
      }

    for (auto iDoF = 0; iDoF < d_dofEDim; ++iDoF)
      {
        for (auto iQuad = 0; iQuad < d_quadEDim; ++iQuad)
          {
            nodalShapeFunctionValuesAtQuadPointsEO[iDoF * d_quadEDim + iQuad] =
              (nodalShapeFunctionValuesAtQuadPoints[iDoF * nQuadPointsPerDim +
                                                    iQuad] +
               nodalShapeFunctionValuesAtQuadPoints[(ndofsPerDim - 1 - iDoF) *
                                                      nQuadPointsPerDim +
                                                    iQuad]) *
              0.5;
          }
      }

    for (auto iDoF = 0; iDoF < d_dofODim; ++iDoF)
      {
        for (auto iQuad = 0; iQuad < d_quadODim; ++iQuad)
          {
            nodalShapeFunctionValuesAtQuadPointsEO[d_dofEDim * d_quadEDim +
                                                   iDoF * d_quadODim + iQuad] =
              (nodalShapeFunctionValuesAtQuadPoints[iDoF * nQuadPointsPerDim +
                                                    iQuad] -
               nodalShapeFunctionValuesAtQuadPoints[(ndofsPerDim - 1 - iDoF) *
                                                      nQuadPointsPerDim +
                                                    iQuad]) *
              0.5;
          }
      }

    for (auto iQuad1 = 0; iQuad1 < d_quadEDim; ++iQuad1)
      {
        for (auto iQuad2 = 0; iQuad2 < d_quadODim; ++iQuad2)
          {
            quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 * d_quadODim +
                                                     iQuad2] =
              (quadShapeFunctionGradientsAtQuadPoints[iQuad1 *
                                                        nQuadPointsPerDim +
                                                      iQuad2] +
               quadShapeFunctionGradientsAtQuadPoints
                 [(nQuadPointsPerDim - 1 - iQuad1) * nQuadPointsPerDim +
                  iQuad2]) *
              0.5;
          }
      }

    for (auto iQuad1 = 0; iQuad1 < d_quadODim; ++iQuad1)
      {
        for (auto iQuad2 = 0; iQuad2 < d_quadEDim; ++iQuad2)
          {
            quadShapeFunctionGradientsAtQuadPointsEO[d_quadEDim * d_quadODim +
                                                     iQuad1 * d_quadEDim +
                                                     iQuad2] =
              (quadShapeFunctionGradientsAtQuadPoints[iQuad1 *
                                                        nQuadPointsPerDim +
                                                      iQuad2] -
               quadShapeFunctionGradientsAtQuadPoints
                 [(nQuadPointsPerDim - 1 - iQuad1) * nQuadPointsPerDim +
                  iQuad2]) *
              0.5;
          }
      }

    jacobianFactor.resize(d_nCells * 9, 0.0);
    jacobianDeterminants.resize(d_nCells, 0.0);

    auto cellOffsets = mappingData.data_index_offsets;

    for (auto iCellBatch = 0, cellCount = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         ++iCellBatch)
      {
        for (auto iCell = 0;
             iCell < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
             ++iCell, ++cellCount)
          {
            for (auto d = 0; d < 3; ++d)
              {
                for (auto e = 0; e < 3; ++e)
                  {
                    for (auto f = 0; f < 3; ++f)
                      {
                        jacobianFactor[cellCount * 9 + d * 3 + e] +=
                          mappingData
                            .jacobians[0][cellOffsets[iCellBatch]][d][f][0] *
                          mappingData
                            .jacobians[0][cellOffsets[iCellBatch]][e][f][0] *
                          mappingData.JxW_values[cellOffsets[iCellBatch]][0] *
                          0.5;
                        jacobianDeterminants[cellCount] =
                          mappingData.JxW_values[cellOffsets[iCellBatch]][0];
                      }
                  }
              }
          }
      }

    // coeffs.resize(d_nQuadsPerCell * d_nCells);
    // for (auto iCell = 0; iCell < d_nCells; ++iCell)
    //   for (auto q = 0; q < d_nQuadsPerCell; ++q)
    //     coeffs[iCell * d_nQuadsPerCell + q] =
    //       2 * M_PI * jacobianDeterminants[iCell];

    /*d_constraintsInfomf.initialize(d_matrixFreeDataPtr->get_vector_partitioner(
                                     d_basisOperationsPtrHost->d_dofHandlerID),
                                   *d_constraintMatrixPtr);

    d_constraintsInfomf.precomputeMaps(
      d_matrixFreeDataPtr->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID),
      cpuTestInputdealiibatched.get_partitioner(),
      d_blockSize,
      batchSize);

    MPI_Barrier(mpi_communicator);
    pcout << "DEBUG setup constraints done" << std::endl;
    tempGhostStorage.resize(
      cpuTestInputdealiibatched.get_partitioner()->n_import_indices(), 0.0);
    tempCompressStorage.resize(
      cpuTestInputdealiibatched.get_partitioner()->n_import_indices(), 0.0);
    tempConstraintStorage.resize(batchSize); //*/

    // initializeOptimizedConstraints();
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::
    initializeOptimizedConstraints()
  {
    unsigned int            ncons = 0;
    const dealii::IndexSet &locally_owned_dofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->locally_owned_range();
    const dealii::IndexSet &ghost_dofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->ghost_indices();

    for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
         it != locally_owned_dofs.end();
         ++it)
      {
        if (d_constraintMatrixPtr->is_constrained(*it))
          {
            ncons++;
            const dealii::types::global_dof_index lineDof = *it;
            const std::vector<
              std::pair<dealii::types::global_dof_index, double>> *rowData =
              d_constraintMatrixPtr->get_constraint_entries(lineDof);

            bool isConstraintRhsExpandingOutOfIndexSet = false;
            for (unsigned int j = 0; j < rowData->size(); ++j)
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

            std::vector<unsigned int> masterData(rowData->size());
            std::vector<double>       weightData(rowData->size());
            double                    inhomogenity =
              d_constraintMatrixPtr->get_inhomogeneity(lineDof);
            for (auto i = 0; i < rowData->size(); ++i)
              {
                masterData[i] = d_matrixFreeDataPtr
                                  ->get_vector_partitioner(
                                    d_basisOperationsPtrHost->d_dofHandlerID)
                                  ->global_to_local((*rowData)[i].first);
                weightData[i] = (*rowData)[i].second;
              }
            bool         constraintExists = false;
            unsigned int constraintIndex  = 0;
            for (auto i = 0; i < masterNodeBuckets.size(); ++i)
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
                    ->get_vector_partitioner(
                      d_basisOperationsPtrHost->d_dofHandlerID)
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
                    ->get_vector_partitioner(
                      d_basisOperationsPtrHost->d_dofHandlerID)
                    ->global_to_local(lineDof)));
                weightMatrixList.push_back(weightData);
                masterNodeBuckets.push_back(masterData);
                inhomogenityList.push_back(inhomogenity);
              }
          }
      }


    for (dealii::IndexSet::ElementIterator it = ghost_dofs.begin();
         it != ghost_dofs.end();
         ++it)
      {
        if (d_constraintMatrixPtr->is_constrained(*it))
          {
            ncons++;
            const dealii::types::global_dof_index lineDof = *it;
            const std::vector<
              std::pair<dealii::types::global_dof_index, double>> *rowData =
              d_constraintMatrixPtr->get_constraint_entries(lineDof);

            bool isConstraintRhsExpandingOutOfIndexSet = false;
            for (unsigned int j = 0; j < rowData->size(); ++j)
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

            std::vector<unsigned int> masterData(rowData->size());
            std::vector<double>       weightData(rowData->size());
            double                    inhomogenity =
              d_constraintMatrixPtr->get_inhomogeneity(lineDof);
            for (auto i = 0; i < rowData->size(); ++i)
              {
                masterData[i] = d_matrixFreeDataPtr
                                  ->get_vector_partitioner(
                                    d_basisOperationsPtrHost->d_dofHandlerID)
                                  ->global_to_local((*rowData)[i].first);
                weightData[i] = (*rowData)[i].second;
              }
            bool         constraintExists = false;
            unsigned int constraintIndex  = 0;
            for (auto i = 0; i < masterNodeBuckets.size(); ++i)
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
                    ->get_vector_partitioner(
                      d_basisOperationsPtrHost->d_dofHandlerID)
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
                    ->get_vector_partitioner(
                      d_basisOperationsPtrHost->d_dofHandlerID)
                    ->global_to_local(lineDof)));
                weightMatrixList.push_back(weightData);
                masterNodeBuckets.push_back(masterData);
                inhomogenityList.push_back(inhomogenity);
              }
          }
      }
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::setVeffMF(
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &VeffJxW,
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST> &VeffExtPotJxW)
  {
    const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr =
      &(d_basisOperationsPtrHost->matrixFreeData());

    auto d_nMacroCells = d_matrixFreeDataPtr->n_cell_batches();
    auto d_nCells      = d_matrixFreeDataPtr->n_physical_cells();
    auto d_nDofsPerCell =
      d_matrixFreeDataPtr
        ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
        .get_fe()
        .dofs_per_cell;

    std::vector<unsigned int> cellIndexToMacroCellSubCellIndexMap(d_nCells);

    auto cellPtr = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .begin_active();
    auto endcPtr = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .end();

    std::map<dealii::CellId, size_type> cellIdToCellIndexMap;
    d_VeffJxW.resize(VeffJxW.size());

    unsigned int iCell = 0;
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          cellIdToCellIndexMap[cellPtr->id()] = iCell;
          pcout << "cellPtr->id(): " << cellPtr->id() << ", iCell: " << iCell << std::endl;
          ++iCell;
        }

    iCell = 0;
    for (unsigned int iMacroCell = 0; iMacroCell < d_nMacroCells; ++iMacroCell)
      {
        const unsigned int numberSubCells =
          d_matrixFreeDataPtr->n_components_filled(iMacroCell);
        for (unsigned int iSubCell = 0; iSubCell < numberSubCells; ++iSubCell)
          {
            cellPtr = d_matrixFreeDataPtr->get_cell_iterator(
              iMacroCell, iSubCell, d_basisOperationsPtrHost->d_dofHandlerID);
            size_type cellIndex = cellIdToCellIndexMap[cellPtr->id()];
            cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;
            std::cout << "iMacroCell: " << iMacroCell
                      << ", iSubCell: " << iSubCell
                      << ", cellIndex: " << cellIndex << ", iCell: " << iCell
                      << std::endl;
            ++iCell;
          }
      }

    /*typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_basisOperationsPtrHost->matrixFreeData()
               .get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
               .begin_active(),
      endc = d_basisOperationsPtrHost->matrixFreeData()
               .get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
               .end();

    std::vector<unsigned int> normalCellIdToMacroCellIdMap(d_nCells);

    const unsigned int numberMacroCells =
      d_basisOperationsPtrHost->matrixFreeData().n_macro_cells();
    unsigned int iElemNormal = 0;

    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            bool         isFound        = false;
            unsigned int iElemMacroCell = 0;
            for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
                 ++iMacroCell)
              {
                const unsigned int n_sub_cells =
                  d_basisOperationsPtrHost->matrixFreeData()
                    .n_components_filled(iMacroCell);

                for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
                  {
                    cellPtr = d_basisOperationsPtrHost->matrixFreeData()
                                .get_cell_iterator(
                                  iMacroCell,
                                  iCell,
                                  d_basisOperationsPtrHost->d_dofHandlerID);

                    if (cell->id() == cellPtr->id())
                      {
                        normalCellIdToMacroCellIdMap[iElemNormal] =
                          iElemMacroCell;
                        isFound = true;
                        break;
                      }

                    iElemMacroCell++;
                  }

                if (isFound)
                  break;
              }

            iElemNormal++;
          }
      } //*/

    pcout << "d_nMacroCells: " << d_nMacroCells << std::endl
          << "d_nCells: " << d_nCells << std::endl
          << "d_nDofsPerCell: " << d_nDofsPerCell << std::endl
          << "d_nQuadsPerCell: " << d_nQuadsPerCell << std::endl;

    pcout << "cellIndexToMacroCellSubCellIndexMap" << std::endl;

    for (int i = 0; i < d_nCells; i++)
      pcout << "i: " << i
            << ", Value: " << cellIndexToMacroCellSubCellIndexMap[i]
            << std::endl;

    for (auto iCell = 0; iCell < d_nCells; ++iCell)
      {
        for (auto iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
          d_VeffJxW[iQuad + cellIndexToMacroCellSubCellIndexMap[iCell] *
                              d_nQuadsPerCell] =
            VeffJxW[iQuad + iCell * d_nQuadsPerCell] *
            jacobianDeterminants[cellIndexToMacroCellSubCellIndexMap[iCell]];
        //     d_VeffJxW[iQuad + iCell * d_nQuadsPerCell] =
        //     jacobianDeterminants[iCell];
      } //*/
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::computeAX(
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &Ax,
    dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &x)
  {
    // x.updateGhostValues();

    for (auto iBatch = 0; iBatch < d_nBatch; ++iBatch)
      {
        std::vector<bool> dofEncountered(d_nRelaventDofs, false);

        /*d_constraintsInfomf.distribute(x,
                                       tempConstraintStorage,
                                       batchSize,
                                       iBatch); //*/

        for (auto iCell = 0; iCell < d_nCells; ++iCell)
          {
            // Extraction
            for (unsigned int iDoF = 0; iDoF < d_nDofsPerCell; ++iDoF)
              {
                std::memcpy(temp10v + iDoF * (batchSize / 8),
                            x.begin() +
                              globalToLocalMap[iDoF + iCell * d_nDofsPerCell] *
                                batchSize +
                              (globalToLocalMap[iDoF + iCell * d_nDofsPerCell] *
                                     batchSize >=
                                   d_nOwnedDofs * d_blockSize ?
                                 d_nGhostDofs :
                                 d_nOwnedDofs) *
                                batchSize * iBatch,
                            batchSize * sizeof(double));

                // temp10v[iDoF * (batchSize / 8) + (b / 8)].load(
                //   x.begin() +
                //   globalToLocalMap[iDoF + iCell * d_nDofsPerCell] *
                //     batchSize +
                //   (globalToLocalMap[iDoF + iCell * d_nDofsPerCell] *
                //          batchSize >=
                //        d_nOwnedDofs * d_blockSize ?
                //      d_nGhostDofs :
                //      d_nOwnedDofs) *
                //     batchSize * iBatch +
                //   b);
                // }
              }

            evaluateTensorContractions(iCell);

            // Assembly
            for (auto i = 0; i < d_nDofsPerCell; ++i)
              {
                if (dofEncountered[singleVectorGlobalToLocalMap
                                     [i + d_nDofsPerCell * iCell]])
                  {
                    for (auto b = 0; b < batchSize; b += 8)
                      {
                        dealii::VectorizedArray<double> temp;
                        temp.load(
                          Ax.begin() +
                          globalToLocalMap[i + iCell * d_nDofsPerCell] *
                            batchSize +
                          (globalToLocalMap[i + iCell * d_nDofsPerCell] *
                                 batchSize >=
                               d_nOwnedDofs * d_blockSize ?
                             d_nGhostDofs :
                             d_nOwnedDofs) *
                            batchSize * iBatch +
                          b);

                        temp += temp10v[i * (batchSize / 8) + (b / 8)];

                        temp.store(
                          Ax.begin() +
                          globalToLocalMap[i + iCell * d_nDofsPerCell] *
                            batchSize +
                          (globalToLocalMap[i + iCell * d_nDofsPerCell] *
                                 batchSize >=
                               d_nOwnedDofs * d_blockSize ?
                             d_nGhostDofs :
                             d_nOwnedDofs) *
                            batchSize * iBatch +
                          b);
                      }
                  }
                else
                  {
                    dofEncountered[singleVectorGlobalToLocalMap
                                     [i + d_nDofsPerCell * iCell]] = true;
                    for (auto b = 0; b < batchSize; b += 8)
                      {
                        temp10v[i * (batchSize / 8) + (b / 8)].store(
                          Ax.begin() +
                          globalToLocalMap[i + iCell * d_nDofsPerCell] *
                            batchSize +
                          (globalToLocalMap[i + iCell * d_nDofsPerCell] *
                                 batchSize >=
                               d_nOwnedDofs * d_blockSize ?
                             d_nGhostDofs :
                             d_nOwnedDofs) *
                            batchSize * iBatch +
                          b);
                      }
                  }
              } //*/
          }

        /*d_constraintsInfomf.distribute_slave_to_master(Ax, batchSize, iBatch);
         * //*/
      }

    // Ax.accumulateAddLocallyOwned();
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::destroyjitkernels()
  {
    for (int i = 0; i < jitpointers.size(); ++i)
      {
        mkl_jit_destroy(jitpointers[i]);
      }
  }


  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::matmul(
    const unsigned int jitPointerIndex,
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    double *           A,
    double *           B,
    double *           C,
    const unsigned int c)
  {
    // for (auto b = 0; b < c; ++b, A += m * k, C += m * n)
    //   {
    //     xmm[jitPointerIndex](A, B, C);
    //   }
    // xmm[jitPointerIndex](A, B, C,LIBXSMM_GEMM_PREFETCH_A(A),
    // LIBXSMM_GEMM_PREFETCH_B(B),
    //       LIBXSMM_GEMM_PREFETCH_C(C));
    for (auto b = 0; b < c; ++b, A += m * k, C += m * n)
      {
        jitGemmKernels[jitPointerIndex](jitpointers[jitPointerIndex], A, B, C);
      }
  }

  template <unsigned int ndofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFree<ndofsPerDim, nQuadPointsPerDim, batchSize>::
    evaluateTensorContractions(const unsigned int iCell)
  {
    matmulEOdealii<(batchSize / 8) * ndofsPerDim * ndofsPerDim,
                   nQuadPointsPerDim,
                   ndofsPerDim,
                   1,
                   false,
                   true,
                   1>(temp10v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp11v);
    matmulEOdealii<(batchSize / 8) * ndofsPerDim,
                   nQuadPointsPerDim,
                   ndofsPerDim,
                   nQuadPointsPerDim,
                   false,
                   true,
                   1>(temp11v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp10v);
    matmulEOdealii<(batchSize / 8),
                   nQuadPointsPerDim,
                   ndofsPerDim,
                   nQuadPointsPerDim * nQuadPointsPerDim,
                   false,
                   true,
                   1>(temp10v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp20v);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   1,
                   false,
                   true,
                   2>(temp20v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp12v);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   false,
                   true,
                   2>(temp20v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp11v);
    matmulEOdealii<(batchSize / 8),
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim * nQuadPointsPerDim,
                   false,
                   true,
                   2>(temp20v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp10v);
    matmulEOdealii<nQuadPointsPerDim * nQuadPointsPerDim *
                     nQuadPointsPerDim *(batchSize / 8),
                   3,
                   3,
                   1,
                   false,
                   true,
                   0>(temp10v, jacobianFactor.data() + iCell * 9, temp10v);
    matmulEOdealii<(batchSize / 8),
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim * nQuadPointsPerDim,
                   false,
                   false,
                   2>(temp10v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp20v,
                      d_VeffJxW.data() + iCell * d_nQuadsPerCell);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   true,
                   false,
                   2>(temp11v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp20v);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   nQuadPointsPerDim,
                   1,
                   true,
                   false,
                   2>(temp12v,
                      quadShapeFunctionGradientsAtQuadPointsEO.data(),
                      temp20v);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim * nQuadPointsPerDim,
                   ndofsPerDim,
                   nQuadPointsPerDim,
                   1,
                   false,
                   false,
                   1>(temp20v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp10v);
    matmulEOdealii<(batchSize / 8) * nQuadPointsPerDim,
                   ndofsPerDim,
                   nQuadPointsPerDim,
                   ndofsPerDim,
                   false,
                   false,
                   1>(temp10v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp20v);
    matmulEOdealii<(batchSize / 8),
                   ndofsPerDim,
                   nQuadPointsPerDim,
                   ndofsPerDim * ndofsPerDim,
                   false,
                   false,
                   1>(temp20v,
                      nodalShapeFunctionValuesAtQuadPointsEO.data(),
                      temp10v);
  }

#include "MatrixFree.inst.cc"
} // namespace dftfe
