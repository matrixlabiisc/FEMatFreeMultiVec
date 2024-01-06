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

#ifndef matrixFreeBase_H_
#define matrixFreeBase_H_
#include <constraintMatrixInfoDevice.h>

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

  class matrixFreeBase
  {
  public:
    virtual ~matrixFreeBase(){};

    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the atoms
     * parameter.
     *
     */
    virtual void
    reinit(const dealii::MatrixFree<3, double> *    matrixFreeDataPtr,
           const dealii::AffineConstraints<double> *constraintMatrixPtr,
           const distributedCPUVec<double> &        sqrtMassVec,
           const unsigned int                       matrixFreeVectorComponent,
           const unsigned int matrixFreeQuadratureComponentAX) = 0;

    virtual void
    setCoeffs(std::vector<double> &coeffs) = 0;

    virtual std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
    getPartitioner() = 0;

    virtual dftUtils::constraintMatrixInfoDevice &
    getConstraintsInfo() = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    computeAXMF(double *      Ax,
                const double *x,
                const int &   numberWaveFunctions) = 0;

    virtual void
    distribute(double *x, const int &numberWaveFunctions) = 0;

    virtual void
    distributeSlaveToMaster(double *   Ax,
                            double *   x,
                            const int &numberWaveFunctions) = 0;
  };

} // namespace dftfe
#endif // matrixFreeBase_H_
