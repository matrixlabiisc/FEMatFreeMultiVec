// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri, Sambit Das
//


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <omp.h>

#include "pseudoGS.cc"

namespace dftfe{

  namespace linearAlgebraOperations
  {

    void callevd(const unsigned int dimensionMatrix,
		 double *matrix,
		 double *eigenValues)
    {

      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &iwork[0],
	      &liwork,
	      &info);

      //
      //free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);

    }


     void callevd(const unsigned int dimensionMatrix,
		 std::complex<double> *matrix,
		 double *eigenValues)
    {
      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      const unsigned int lrwork = 1 + 5*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix;
      std::vector<double> rwork(lrwork);
      std::vector<std::complex<double> > work(lwork);


      zheevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

      //
      //free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<std::complex<double> >().swap(work);
      std::vector<int>().swap(iwork);


    }


    void callevr(const unsigned int dimensionMatrix,
		 std::complex<double> *matrixInput,
		 std::complex<double> *eigenVectorMatrixOutput,
		 double *eigenValues)
    {
      char jobz = 'V', uplo = 'U', range = 'A';
      const double vl=0.0,vu=0.0;
      const unsigned int il=0,iu = 0;
      const double abstol = 1e-08;
      std::vector<unsigned int> isuppz(2*dimensionMatrix);
      const int lwork = 2*dimensionMatrix;
      std::vector<std::complex<double> > work(lwork);
      const int liwork = 10*dimensionMatrix;
      std::vector<int> iwork(liwork);
      const int lrwork = 24*dimensionMatrix;
      std::vector<double> rwork(lrwork);
      int info;

      zheevr_(&jobz,
	      &range,
	      &uplo,
	      &dimensionMatrix,
	      matrixInput,
	      &dimensionMatrix,
	      &vl,
	      &vu,
	      &il,
	      &iu,
	      &abstol,
	      &dimensionMatrix,
	      eigenValues,
	      eigenVectorMatrixOutput,
	      &dimensionMatrix,
	      &isuppz[0],
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);
    }




    void callevr(const unsigned int dimensionMatrix,
		 double *matrixInput,
		 double *eigenVectorMatrixOutput,
		 double *eigenValues)
    {
      char jobz = 'V', uplo = 'U', range = 'A';
      const double vl=0.0,vu = 0.0;
      const unsigned int il=0,iu=0;
      const double abstol = 0.0;
      std::vector<unsigned int> isuppz(2*dimensionMatrix);
      const int lwork = 26*dimensionMatrix;
      std::vector<double> work(lwork);
      const int liwork = 10*dimensionMatrix;
      std::vector<int> iwork(liwork);
      int info;

      dsyevr_(&jobz,
	      &range,
	      &uplo,
	      &dimensionMatrix,
	      matrixInput,
	      &dimensionMatrix,
	      &vl,
	      &vu,
	      &il,
	      &iu,
	      &abstol,
	      &dimensionMatrix,
	      eigenValues,
	      eigenVectorMatrixOutput,
	      &dimensionMatrix,
	      &isuppz[0],
	      &work[0],
	      &lwork,
	      &iwork[0],
	      &liwork,
	      &info);

      AssertThrow(info==0,dealii::ExcMessage("Error in dsyevr"));


    }




    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<double> & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<double> & X,
		  dealii::parallel::distributed::Vector<double> & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const double alpha = 1.0, beta = 0.0;
      dgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }


    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<std::complex<double> > & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<std::complex<double> > & X,
		  dealii::parallel::distributed::Vector<std::complex<double> > & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const std::complex<double> alpha = 1.0, beta = 0.0;
      zgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }



    //
    //chebyshev filtering of given subspace XArray
    //
    template<typename T>
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 dealii::parallel::distributed::Vector<T> & XArray,
			 const unsigned int numberWaveFunctions,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
    {
      if (dftParameters::chebyshevOMPThreads!=0)
	  omp_set_num_threads(dftParameters::chebyshevOMPThreads);

      double e, c, sigma, sigma1, sigma2, gamma;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;

      dealii::parallel::distributed::Vector<T> YArray;//,YNewArray;

      //
      //create YArray
      //
      YArray.reinit(XArray);


      //
      //initialize to zeros.
      //x
      const T zeroValue = 0.0;
      YArray = zeroValue;


      //
      //call HX
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(XArray,
			numberWaveFunctions,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
			scaleFlag,
			scalar,
			YArray);


      T alpha1 = sigma1/e, alpha2 = -c;

      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      YArray.add(alpha2,XArray);
      YArray *= alpha1;

      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  //
	  //multiply XArray with alpha2
	  //
	  XArray *= alpha2;
	  XArray.add(-c*alpha1,YArray);


	  //
	  //call HX
	  //
	  bool scaleFlag = true;
	  operatorMatrix.HX(YArray,
			    numberWaveFunctions,
			    flattenedArrayMacroCellLocalProcIndexIdMap,
			    flattenedArrayCellLocalProcIndexIdMap,
			    scaleFlag,
			    alpha1,
			    XArray);

	  //
	  //XArray = YArray
	  //
	  XArray.swap(YArray);

	  //
	  //YArray = YNewArray
	  //
	  sigma = sigma2;

	}

      //copy back YArray to XArray
      XArray = YArray;

      if (dftParameters::chebyshevOMPThreads!=0)
	  omp_set_num_threads(1);
    }

    template<typename T>
    void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				      const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(X.get_mpi_communicator(),
		   localVectorSize,
		   PETSC_DETERMINE,
		   &templateVec);
      VecSetFromOptions(templateVec);


      //
      //Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(X.get_mpi_communicator(),&columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors,
			templateVec,
			numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      //create list of indices
      //
      std::vector<PetscInt> indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize);

      PetscInt low,high;

      VecGetOwnershipRange(templateVec,
			   &low,
			   &high);


      for(PetscInt index = 0;index < localVectorSize; ++index)
	indices[index] = low+index;

      VecDestroy(&templateVec);

      //
      //Fill in data into BV object
      //
      Vec v;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v);
	  VecSet(v,0.0);
	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    data[iNode] = X.local_element(numberVectors*iNode + iColumn);

	  VecSetValues(v,
		       localVectorSize,
		       &indices[0],
		       &data[0],
		       INSERT_VALUES);

	  VecAssemblyBegin(v);
	  VecAssemblyEnd(v);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v);
	}

      //
      //orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors,NULL);

      //
      //Copy data back into X
      //
      Vec v1;
      PetscScalar * pointerv1;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v1);

	  VecGetArray(v1,
		      &pointerv1);

	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    X.local_element(numberVectors*iNode + iColumn) = pointerv1[iNode];

	  VecRestoreArray(v1,
			  &pointerv1);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v1);
	}

      BVDestroy(&columnSpaceOfVectors);

    }

    template<typename T>
    void rayleighRitz(operatorDFTClass & operatorMatrix,
		      dealii::parallel::distributed::Vector<T> & X,
		      const unsigned int numberWaveFunctions,
		      const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
		      const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
		      std::vector<double> & eigenValues)
    {
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(dftParameters::orthoRROMPThreads);

      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);
      //
      //compute projected Hamiltonian
      //
      std::vector<T> ProjHam;
      const unsigned int numberEigenValues = numberWaveFunctions;
      eigenValues.resize(numberEigenValues);

      computing_timer.enter_section("XtHX");
      operatorMatrix.XtHX(X,
			  numberEigenValues,
			  flattenedArrayMacroCellLocalProcIndexIdMap,
			  flattenedArrayCellLocalProcIndexIdMap,
			  ProjHam);
      computing_timer.exit_section("XtHX");

      //
      //compute eigendecomposition of ProjHam
      //
      computing_timer.enter_section("eigen decomp in RR");
#if(defined WITH_SCALAPACK && !USE_COMPLEX)
      const unsigned rowsBlockSize=std::min((unsigned int)50,numberWaveFunctions);
      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
      internal::createProcessGridSquareMatrix(X.get_mpi_communicator(),
		                              numberWaveFunctions,
		                              processGrid,
				              rowsBlockSize);

      dealii::ScaLAPACKMatrix<T> projHamPar(numberWaveFunctions,
                                            processGrid,
                                            rowsBlockSize);
      computing_timer.enter_section("scalapack copy");
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
           {
             const unsigned int glob_i = projHamPar.global_column(i);
             for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
               {
                 const unsigned int glob_j = projHamPar.global_row(j);
                 projHamPar.local_el(j, i)   = ProjHam[glob_i*numberWaveFunctions+glob_j];
               }
           }
      computing_timer.exit_section("scalapack copy");

      computing_timer.enter_section("scalapack eigen decomp");
      eigenValues=projHamPar.eigenpairs_symmetric_by_index_MRRR(std::make_pair(0,numberWaveFunctions-1),true);
      computing_timer.exit_section("scalapack eigen decomp");

      computing_timer.enter_section("scalapack copy");
      std::fill(ProjHam.begin(),ProjHam.end(),T(0));
      if (processGrid->is_process_active())
         for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
           {
             const unsigned int glob_i = projHamPar.global_column(i);
             for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
               {
                 const unsigned int glob_j = projHamPar.global_row(j);
                 ProjHam[glob_i*numberWaveFunctions+glob_j]=projHamPar.local_el(j, i);
               }
           }
      dealii::Utilities::MPI::sum(ProjHam, X.get_mpi_communicator(), ProjHam);
      computing_timer.exit_section("scalapack copy");

#else
      callevd(numberEigenValues,
	      &ProjHam[0],
	      &eigenValues[0]);

#endif
      computing_timer.exit_section("eigen decomp in RR");


      //
      //rotate the basis in the subspace X = X*Q
      //
      const unsigned int localVectorSize = X.local_size()/numberEigenValues;
      dealii::parallel::distributed::Vector<T> rotatedBasis;
      rotatedBasis.reinit(X);

      computing_timer.enter_section("subspace rotation in RR");
      callgemm(numberEigenValues,
	       localVectorSize,
	       ProjHam,
	       X,
	       rotatedBasis);
      computing_timer.exit_section("subspace rotation in RR");

      X = rotatedBasis;

      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(1);
    }

    template<typename T>
    void computeEigenResidualNorm(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<T> & X,
				  const std::vector<double> & eigenValues,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,

				  std::vector<double> & residualNorm)

    {

      //
      //get the number of eigenVectors
      //
      const unsigned int numberVectors = eigenValues.size();

      //
      //reinit blockSize require for HX later
      //
      operatorMatrix.reinit(numberVectors);

      //
      //create temp Array
      //
      dealii::parallel::distributed::Vector<T> Y;
      Y.reinit(X);

      //
      //initialize to zero
      //
      const T zeroValue = 0.0;
      Y = zeroValue;

      //
      //compute operator times X
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(X,
			numberVectors,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
			scaleFlag,
			scalar,
			Y);

      if(dftParameters::verbosity>=2)
	{
	  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    std::cout<<"L-2 Norm of residue   :"<<std::endl;
	}


      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //compute residual norms
      //
      std::vector<T> residualNormSquare(numberVectors,0.0);
      for(unsigned int iDof = 0; iDof < localVectorSize; ++iDof)
	{
	  for(unsigned int iWave = 0; iWave < numberVectors; iWave++)
	    {
	      T value = Y.local_element(numberVectors*iDof + iWave) - eigenValues[iWave]*X.local_element(numberVectors*iDof + iWave);
	      residualNormSquare[iWave] += std::abs(value)*std::abs(value);
	    }
	}


      dealii::Utilities::MPI::sum(residualNormSquare,X.get_mpi_communicator(),residualNormSquare);


      for(unsigned int iWave = 0; iWave < numberVectors; ++iWave)
	{
#ifdef USE_COMPLEX
	  double value = residualNormSquare[iWave].real();
#else
	  double value = residualNormSquare[iWave];
#endif
	  residualNorm[iWave] = sqrt(value);

	  if(dftParameters::verbosity>=2)
	    {
	      if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		std::cout<<"eigen vector "<< iWave<<": "<<residualNorm[iWave]<<std::endl;
	    }
	}

      if(dftParameters::verbosity>=2)
      {
	if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	  std::cout <<std::endl;
      }

    }





#ifdef USE_COMPLEX
    void lowdenOrthogonalization(dealii::parallel::distributed::Vector<std::complex<double> > & X,
				 const unsigned int numberVectors)
    {
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(dftParameters::orthoRROMPThreads);

      const unsigned int localVectorSize = X.local_size()/numberVectors;
      std::vector<std::complex<double> > overlapMatrix(numberVectors*numberVectors,0.0);

      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;

      //
      //compute overlap matrix S = {(Zc)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Zc)^T is conjugate transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the transpose of overlap matrix i.e S^{T} = X*{(Xc)^T} here
      //
      const char uplo = 'U';
      const char trans = 'N';

      zherk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);


      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);

      //
      //evaluate the conjugate of {S^T} to get actual overlap matrix
      //
      for(unsigned int i = 0; i < overlapMatrix.size(); ++i)
	overlapMatrix[i] = std::conj(overlapMatrix[i]);


      //
      //set lapack eigen decomposition flags and compute eigendecomposition of S = Q*D*Q^{H}
      //
      int info;
      const unsigned int lwork = 1 + 6*numberVectors + 2*numberVectors*numberVectors, liwork = 3 + 5*numberVectors;
      std::vector<int> iwork(liwork,0);
      const char jobz='V';
      const unsigned int lrwork = 1 + 5*numberVectors + 2*numberVectors*numberVectors;
      std::vector<double> rwork(lrwork,0.0);
      std::vector<std::complex<double> > work(lwork);
      std::vector<double> eigenValuesOverlap(numberVectors,0.0);

      zheevd_(&jobz,
	      &uplo,
	      &numberVectors,
	      &overlapMatrix[0],
	      &numberVectors,
	      &eigenValuesOverlap[0],
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

       //
       //free up memory associated with work
       //
       work.clear();
       iwork.clear();
       rwork.clear();
       std::vector<std::complex<double> >().swap(work);
       std::vector<double>().swap(rwork);
       std::vector<int>().swap(iwork);

       //
       //compute D^{-1/4} where S = Q*D*Q^{H}
       //
       std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,0.0);

       for(unsigned i = 0; i < numberEigenValues; ++i)
	 invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "zheevd"
       //
       const unsigned int inc = 1;
       for(unsigned int i = 0; i < numberEigenValues; ++i)
	 {
	   std::complex<double> scalingCoeff = invFourthRootEigenValuesMatrix[i];
	   zscal_(&numberEigenValues,
		  &scalingCoeff,
		  &overlapMatrix[0]+i*numberEigenValues,
                  &inc);
	 }

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{H} = (Q*D^{-1/4})*(Q*D^{-1/4))^{H}
       //
       std::vector<std::complex<double> > invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'C';
       const std::complex<double> alpha1 = 1.0, beta1 = 0.0;


       zgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha1,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta1,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<std::complex<double> >().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Y^T = {S^{-1/2}}^T*{X^T}
       //using the column major format of blas
       //
       const char transA2  = 'T', transB2  = 'N';
       dealii::parallel::distributed::Vector<std::complex<double> > orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       zgemm_(&transA2,
	     &transB2,
	     &numberEigenValues,
             &localVectorSize,
	     &numberEigenValues,
	     &alpha1,
	     &invSqrtOverlapMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta1,
	     orthoNormalizedBasis.begin(),
	     &numberEigenValues);


       X = orthoNormalizedBasis;

       if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(1);
    }
#else
    void lowdenOrthogonalization(dealii::parallel::distributed::Vector<double> & X,
				 const unsigned int numberVectors)
    {
      if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(dftParameters::orthoRROMPThreads);

      const unsigned int localVectorSize = X.local_size()/numberVectors;

      std::vector<double> overlapMatrix(numberVectors*numberVectors,0.0);


      dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      dealii::TimerOutput computing_timer(pcout,
					  dftParameters::reproducible_output ? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					  dealii::TimerOutput::wall_times);




      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char uplo = 'U';
      const char trans = 'N';

      //
      //compute overlap matrix S = {(Z)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Z)^T is transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the overlap matrix as S = S^{T} = X*{X^T} here
      //

      computing_timer.enter_section("local overlap matrix for lowden");
      dsyrk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);
      computing_timer.exit_section("local overlap matrix for lowden");

      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix);

      std::vector<double> eigenValuesOverlap(numberVectors);
      computing_timer.enter_section("eigen decomp. of overlap matrix");
      callevd(numberVectors,
	      &overlapMatrix[0],
	      &eigenValuesOverlap[0]);
      computing_timer.exit_section("eigen decomp. of overlap matrix");

      //
      //compute D^{-1/4} where S = Q*D*Q^{T}
      //
      std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues);
      unsigned int nanFlag = 0;
      for(unsigned i = 0; i < numberEigenValues; ++i)
	{
	  invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);
	  if(std::isnan(invFourthRootEigenValuesMatrix[i]))
	    {
	      nanFlag = 1;
	      std::cout<<"Nan obtained in proc: "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" and switching to more robust dsyevr for eigen decomposition "<<std::endl;
	      break;
	    }
	}

      if(nanFlag == 1)
	{
	  std::vector<double> overlapMatrixEigenVectors(numberVectors*numberVectors,0.0);
	  eigenValuesOverlap.clear();
	  eigenValuesOverlap.resize(numberVectors);
	  invFourthRootEigenValuesMatrix.clear();
	  invFourthRootEigenValuesMatrix.resize(numberVectors);
	  computing_timer.enter_section("eigen decomp. of overlap matrix");
	  callevr(numberVectors,
		  &overlapMatrix[0],
		  &overlapMatrixEigenVectors[0],
		  &eigenValuesOverlap[0]);
	  computing_timer.exit_section("eigen decomp. of overlap matrix");

	  overlapMatrix = overlapMatrixEigenVectors;
	  overlapMatrixEigenVectors.clear();
	  std::vector<double>().swap(overlapMatrixEigenVectors);

	  //
	  //compute D^{-1/4} where S = Q*D*Q^{T}
	  //
	  for(unsigned i = 0; i < numberEigenValues; ++i)
	    {
	      invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],(1.0/4.0));
	      AssertThrow(!std::isnan(invFourthRootEigenValuesMatrix[i]),dealii::ExcMessage("Eigen values of overlap matrix during Lowden Orthonormalization are very small and close to zero or negative"));
	    }
	}

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "dsyevd"
       //
      computing_timer.enter_section("scaling in Lowden");
      const unsigned int inc = 1;
      for(unsigned int i = 0; i < numberEigenValues; ++i)
	{
	  double scalingCoeff = invFourthRootEigenValuesMatrix[i];
	  dscal_(&numberEigenValues,
		 &scalingCoeff,
		 &overlapMatrix[0]+i*numberEigenValues,
		 &inc);
	}
      computing_timer.exit_section("scaling in Lowden");

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{T} = (Q*D^{-1/4})*(Q*D^{-1/4))^{T}
       //
       std::vector<double> invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'T';
       computing_timer.enter_section("inverse sqrt overlap");
       dgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);
       computing_timer.exit_section("inverse sqrt overlap");

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<double>().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Yt = S^{-1/2}*Xt
       //using the column major format of blas
       //
       const char transA2  = 'N', transB2  = 'N';
       dealii::parallel::distributed::Vector<double> orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       computing_timer.enter_section("subspace rotation in lowden");
       dgemm_(&transA2,
	      &transB2,
	      &numberEigenValues,
	      &localVectorSize,
	      &numberEigenValues,
	      &alpha,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues,
	      X.begin(),
	      &numberEigenValues,
	      &beta,
	      orthoNormalizedBasis.begin(),
	      &numberEigenValues);
       computing_timer.exit_section("subspace rotation in lowden");


       X = orthoNormalizedBasis;

       if (dftParameters::orthoRROMPThreads!=0)
	  omp_set_num_threads(1);
    }
#endif



    template void chebyshevFilter(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<dataTypes::number> & ,
				  const unsigned int ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const unsigned int,
				  const double ,
				  const double ,
				  const double );


    template void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> &,
					       const unsigned int);

    template void pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> &,
					             const unsigned int);

    template void pseudoGramSchmidtOrthogonalizationSerial
	                                           (dealii::parallel::distributed::Vector<dataTypes::number> &,
					            const unsigned int);

    template void pseudoGramSchmidtOrthogonalizationParallel
	                                           (dealii::parallel::distributed::Vector<dataTypes::number> &,
					            const unsigned int);

    template void rayleighRitz(operatorDFTClass  & operatorMatrix,
			       dealii::parallel::distributed::Vector<dataTypes::number> &,
			       const unsigned int numberWaveFunctions,
			       const std::vector<std::vector<dealii::types::global_dof_index> > &,
			       const std::vector<std::vector<dealii::types::global_dof_index> > &,
			       std::vector<double>     & eigenValues);

    template void computeEigenResidualNorm(operatorDFTClass        & operatorMatrix,
					   dealii::parallel::distributed::Vector<dataTypes::number> & X,
					   const std::vector<double> & eigenValues,
					   const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
					   const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
					   std::vector<double>     & residualNorm);




  }//end of namespace

}
