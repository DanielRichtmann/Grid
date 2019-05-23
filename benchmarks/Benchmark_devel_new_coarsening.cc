/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_devel_new_coarsening.cc

    Copyright (C) 2015-2018

    Author: Daniel Richtmann <daniel.richtmann@ur.de>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
/*  END LEGAL */

#include <Grid/Grid.h>
#include <Benchmark_helpers.h>

using namespace Grid;
using namespace Grid::QCD;
using namespace Grid::BenchmarkHelpers;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_devel_new_coarsening
#ifndef NBASIS
#define NBASIS 40
#endif

///////////////////////////////////////////////////////////////////////////////
//             Type definitions (needed outside of main function)            //
///////////////////////////////////////////////////////////////////////////////

typedef OriginalCoarseningPolicy<LatticeFermion, vComplex, NBASIS>  OneSpinCoarseningPol;
typedef TwoSpinCoarseningPolicy<LatticeFermion, vComplex, NBASIS/2> TwoSpinCoarseningPol;

typedef CoarsenedMatrix<vSpinColourVector, vTComplex, NBASIS> OriginalCoarsenedMatrix;
typedef CoarsenedMatrixUsingPolicies<OneSpinCoarseningPol> OneSpinCoarsenedMatrix;
typedef CoarsenedMatrixUsingPolicies<TwoSpinCoarseningPol> TwoSpinCoarsenedMatrix;

typedef OriginalCoarsenedMatrix::CoarseVector OriginalCoarseVector;
typedef OneSpinCoarsenedMatrix::FermionField  OneSpinCoarseVector;
typedef TwoSpinCoarsenedMatrix::FermionField  TwoSpinCoarseVector;

typedef OriginalCoarsenedMatrix::CoarseMatrix OriginalCoarseLinkField;
typedef OneSpinCoarsenedMatrix::LinkField  OneSpinCoarseLinkField;
typedef TwoSpinCoarsenedMatrix::LinkField  TwoSpinCoarseLinkField;

typedef Aggregation<vSpinColourVector, vTComplex, NBASIS> OriginalAggregation;
typedef AggregationUsingPolicies<OneSpinCoarseningPol> OneSpinAggregation;
typedef AggregationUsingPolicies<TwoSpinCoarseningPol> TwoSpinAggregation;

///////////////////////////////////////////////////////////////////////////////
//                       Definition of helper functions                      //
///////////////////////////////////////////////////////////////////////////////

template<class InputCoarsenedMatrix>
void convertToOriginalLayout(typename InputCoarsenedMatrix::LinkField const &inputLayoutIn,
                             OriginalCoarseLinkField &                       originalLayoutOut) {
  assert(0);
}
template<>
void convertToOriginalLayout<OneSpinCoarsenedMatrix>(typename OneSpinCoarsenedMatrix::LinkField const &oneSpinLayoutIn,
                                                     OriginalCoarseLinkField &                         originalLayoutOut) {
  originalLayoutOut = oneSpinLayoutIn;
}
template<>
void convertToOriginalLayout<TwoSpinCoarsenedMatrix>(typename TwoSpinCoarsenedMatrix::LinkField const &twoSpinLayoutIn,
                                                     OriginalCoarseLinkField &                         originalLayoutOut) {
  conformable(twoSpinLayoutIn._grid, originalLayoutOut._grid);
  GridBase *grid   = originalLayoutOut._grid;
  int       Nb     = TwoSpinCoarsenedMatrix::Nbasis;
  int       Nbasis = Nb * 2;
  parallel_for(int ss = 0; ss < grid->oSites(); ss++) {
    for(int n1 = 0; n1 < Nbasis; n1++) {
      for(int n2 = 0; n2 < Nbasis; n2++) {
        int spinIndex1                             = n1 / Nb;
        int spinIndex2                             = n2 / Nb;
        int colourIndex1                           = n1 % Nb;
        int colourIndex2                           = n2 % Nb;
        originalLayoutOut._odata[ss](n1, n2)()()() = twoSpinLayoutIn._odata[ss]()(spinIndex1, spinIndex2)(colourIndex1, colourIndex2);
      }
    }
  }
}

template<class InputCoarsenedMatrix>
void compareToResultInOriginalLayout(OriginalCoarseLinkField const &                 originalLayoutReference,
                                     typename InputCoarsenedMatrix::LinkField const &otherLayoutResult) {
  OriginalCoarseLinkField originalLayoutTmp(originalLayoutReference._grid);
  convertToOriginalLayout<InputCoarsenedMatrix>(otherLayoutResult, originalLayoutTmp);
  printDeviationFromReference(originalLayoutReference, originalLayoutTmp);
}


template<class InputCoarsenedMatrix>
void convertToOriginalLayout(typename InputCoarsenedMatrix::FermionField const &inputLayoutIn,
                             OriginalCoarseVector &                             originalLayoutOut) {
  assert(0);
}
template<>
void convertToOriginalLayout<OneSpinCoarsenedMatrix>(typename OneSpinCoarsenedMatrix::FermionField const &oneSpinLayoutIn,
                                                     OriginalCoarseVector &                               originalLayoutOut) {
  originalLayoutOut = oneSpinLayoutIn;
}
template<>
void convertToOriginalLayout<TwoSpinCoarsenedMatrix>(typename TwoSpinCoarsenedMatrix::FermionField const &twoSpinLayoutIn,
                                                     OriginalCoarseVector &                               originalLayoutOut) {
  conformable(twoSpinLayoutIn._grid, originalLayoutOut._grid);
  GridBase *grid   = originalLayoutOut._grid;
  int       Nb     = TwoSpinCoarsenedMatrix::Nbasis;
  int       Nbasis = Nb * 2;
  parallel_for(int ss = 0; ss < grid->oSites(); ss++) {
    for(int n = 0; n < Nbasis; n++) {
      int spinIndex                         = n / Nb;
      int colourIndex                       = n % Nb;
      originalLayoutOut._odata[ss](n)()()() = twoSpinLayoutIn._odata[ss]()(spinIndex)(colourIndex);
    }
  }
}

template<class InputCoarsenedMatrix>
void compareToResultInOriginalLayout(OriginalCoarseVector const &                       originalLayoutReference,
                                     typename InputCoarsenedMatrix::FermionField const &otherLayoutResult) {
  OriginalCoarseVector originalLayoutTmp(originalLayoutReference._grid);
  convertToOriginalLayout<InputCoarsenedMatrix>(otherLayoutResult, originalLayoutTmp);
  printDeviationFromReference(originalLayoutReference, originalLayoutTmp);
}

template<class InputCoarsenedMatrixCoarseVector>
void convertToTwoSpinLayout(InputCoarsenedMatrixCoarseVector const &       inputLayoutIn,
                            typename TwoSpinCoarsenedMatrix::FermionField &twoSpinLayoutOut) {
  conformable(inputLayoutIn._grid, twoSpinLayoutOut._grid);
  GridBase *grid   = inputLayoutIn._grid;
  int       Nb     = TwoSpinCoarsenedMatrix::Nbasis;
  int       Nbasis = Nb * 2;
  parallel_for(int ss = 0; ss < grid->oSites(); ss++) {
    for(int n = 0; n < Nbasis; n++) {
      int spinIndex                                         = n / Nb;
      int colourIndex                                       = n % Nb;
      twoSpinLayoutOut._odata[ss]()(spinIndex)(colourIndex) = inputLayoutIn._odata[ss](n)()()();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                               Main function                               //
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

  Grid_init(&argc, &argv);

  /////////////////////////////////////////////////////////////////////////////
  //                          Read from command line                         //
  /////////////////////////////////////////////////////////////////////////////

  const int nBasis           = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int nB               = nBasis / 2;
  const int nIter            = readFromCommandLineInt(&argc, &argv, "--niter", 10);
  std::vector<int> blockSize = readFromCommandLineIntVec(&argc, &argv, "--blocksize", std::vector<int>({4, 4, 4, 4}));

  /////////////////////////////////////////////////////////////////////////////
  //                              General setup                              //
  /////////////////////////////////////////////////////////////////////////////

  std::vector<int> clatt = calcCoarseLattSize(GridDefaultLatt(), blockSize);

  GridCartesian *        FGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian *        CGrid   = SpaceTimeGrid::makeFourDimGrid(clatt, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);
  GridRedBlackCartesian *CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  FGrid->show_decomposition();
  CGrid->show_decomposition();

  GridParallelRNG FPRNG(FGrid);
  GridParallelRNG CPRNG(CGrid);

  std::vector<int> seeds({1, 2, 3, 4});

  FPRNG.SeedFixedIntegers(seeds);
  CPRNG.SeedFixedIntegers(seeds);

  /////////////////////////////////////////////////////////////////////////////
  //                    Setup of Dirac Matrix and Operator                   //
  /////////////////////////////////////////////////////////////////////////////

  RealD mass=0.5;
  LatticeGaugeField Umu(FGrid); SU3::HotConfiguration(FPRNG,Umu);
  WilsonFermionR Dw(Umu,*FGrid,*FrbGrid,mass);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> LinOp(Dw);

  /////////////////////////////////////////////////////////////////////////////
  //                           Setup of Aggregation                          //
  /////////////////////////////////////////////////////////////////////////////

  OriginalAggregation originalAggregates(CGrid, FGrid, 0);
  OneSpinAggregation  oneSpinAggregates(CGrid, FGrid, 0);
  TwoSpinAggregation  twoSpinAggregates(CGrid, FGrid, 0);

  originalAggregates.CreateSubspaceRandom(FPRNG);
  for(int i=0; i<twoSpinAggregates.Subspace().size(); i++)
    twoSpinAggregates.Subspace()[i] = originalAggregates.Subspace()[i];
  performChiralDoubling(originalAggregates.Subspace());
  for(int i = 0; i < originalAggregates.Subspace().size(); i++)
    oneSpinAggregates.Subspace()[i] = originalAggregates.Subspace()[i];

  /////////////////////////////////////////////////////////////////////////////
  //                         Setup of CoarsenedMatrix                        //
  /////////////////////////////////////////////////////////////////////////////

  OriginalCoarsenedMatrix originalCoarsenedMatrix(*CGrid, *CrbGrid);
  OneSpinCoarsenedMatrix oneSpinCoarsenedMatrix(*CGrid, *CrbGrid);
  TwoSpinCoarsenedMatrix twoSpinCoarsenedMatrix(*CGrid, *CrbGrid);

  /////////////////////////////////////////////////////////////////////////////
  //                     Setup of vectors for Benchmarks                     //
  /////////////////////////////////////////////////////////////////////////////

  // NOTE: All "In" vectors set to random, all "Out" vectors set to zero

  std::vector<OriginalCoarsenedMatrix::CoarseMatrix> originalLinkFieldTmp(originalCoarsenedMatrix.geom.npoint, CGrid); for (auto &elem: originalLinkFieldTmp) elem = zero;
  OriginalCoarseVector originalCoarseVectorIn(CGrid); random(CPRNG, originalCoarseVectorIn);
  OneSpinCoarseVector  oneSpinCoarseVectorIn(CGrid); oneSpinCoarseVectorIn     = originalCoarseVectorIn;
  TwoSpinCoarseVector  twoSpinCoarseVectorIn(CGrid); convertToTwoSpinLayout(originalCoarseVectorIn, twoSpinCoarseVectorIn);

  OriginalCoarseVector originalCoarseVectorOut(CGrid); originalCoarseVectorOut = zero;
  OneSpinCoarseVector  oneSpinCoarseVectorOut(CGrid); oneSpinCoarseVectorOut   = zero;
  TwoSpinCoarseVector  twoSpinCoarseVectorOut(CGrid);  twoSpinCoarseVectorOut  = zero;
  OriginalCoarseVector originalCoarseVectorTmp(CGrid); originalCoarseVectorTmp = zero;

  LatticeFermion originalFineVectorIn(FGrid); random(FPRNG, originalFineVectorIn);
  LatticeFermion oneSpinFineVectorIn(FGrid); oneSpinFineVectorIn     = originalFineVectorIn;
  LatticeFermion twoSpinFineVectorIn(FGrid);  twoSpinFineVectorIn    = originalFineVectorIn;

  LatticeFermion originalFineVectorOut(FGrid); originalFineVectorOut = zero;
  LatticeFermion oneSpinFineVectorOut(FGrid); oneSpinFineVectorOut   = zero;
  LatticeFermion twoSpinFineVectorOut(FGrid); twoSpinFineVectorOut   = zero;

  /////////////////////////////////////////////////////////////////////////////
  //            Calculate performance figures for instrumentation            //
  /////////////////////////////////////////////////////////////////////////////

  double nStencil      = originalCoarsenedMatrix.geom.npoint;
  double nAccum        = nStencil;
  double FSiteElems    = Nc * Ns;
  double originalCSiteElems = nBasis;
  double oneSpinCSiteElems = originalCSiteElems;
  double twoSpinCSiteElems = TwoSpinCoarsenedMatrix::Ncs * TwoSpinCoarsenedMatrix::Nbasis;

  double FVolume = std::accumulate(FGrid->_fdimensions.begin(), FGrid->_fdimensions.end(), 1, std::multiplies<double>());
  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

  // M

  double flopOriginalM = CVolume * ((nStencil * (8 * originalCSiteElems * originalCSiteElems - 2 * originalCSiteElems) + nAccum * 2 * originalCSiteElems) + 8 * originalCSiteElems);
  double byteOriginalM = CVolume * ((nStencil * (originalCSiteElems * originalCSiteElems + originalCSiteElems) + originalCSiteElems) + originalCSiteElems) * sizeof(Complex);

  double flopOneSpinM  = CVolume * ((nStencil * (8 * oneSpinCSiteElems * oneSpinCSiteElems - 2 * oneSpinCSiteElems) + nAccum * 2 * oneSpinCSiteElems) + 8 * oneSpinCSiteElems);
  double byteOneSpinM  = CVolume * ((nStencil * (oneSpinCSiteElems * oneSpinCSiteElems + oneSpinCSiteElems) + oneSpinCSiteElems) + oneSpinCSiteElems) * sizeof(Complex);

  double flopTwoSpinM  = CVolume * ((nStencil * (8 * twoSpinCSiteElems * twoSpinCSiteElems - 2 * twoSpinCSiteElems) + nAccum * 2 * twoSpinCSiteElems) + 8 * twoSpinCSiteElems);
  double byteTwoSpinM  = CVolume * ((nStencil * (twoSpinCSiteElems * twoSpinCSiteElems + twoSpinCSiteElems) + twoSpinCSiteElems) + twoSpinCSiteElems) * sizeof(Complex);

  // Mdag NOTE: These refer to Galerkin coarsening, i.e., Mdag = g5c * M * g5c

  double flopOriginalMdag = flopOriginalM + 2 * CVolume * (3 * originalCSiteElems);
  double byteOriginalMdag = byteOriginalM + 2 * CVolume * (3 * originalCSiteElems) * sizeof(Complex);

  double flopOneSpinMdag  = flopOneSpinM  + 2 * CVolume * (3 * oneSpinCSiteElems);
  double byteOneSpinMdag  = byteOneSpinM  + 2 * CVolume * (3 * oneSpinCSiteElems) * sizeof(Complex);

  double flopTwoSpinMdag  = flopTwoSpinM  + 2 * CVolume * (3 * twoSpinCSiteElems);
  double byteTwoSpinMdag  = byteTwoSpinM  + 2 * CVolume * (3 * twoSpinCSiteElems) * sizeof(Complex);

  // Mdir

  double flopOriginalMdir = CVolume * (8 * originalCSiteElems * originalCSiteElems - 2 * originalCSiteElems);
  double byteOriginalMdir = CVolume * (originalCSiteElems * originalCSiteElems + 2 * originalCSiteElems) * sizeof(Complex);

  double flopOneSpinMdir  = CVolume * (8 * oneSpinCSiteElems * oneSpinCSiteElems - 2 * oneSpinCSiteElems);
  double byteOneSpinMdir  = CVolume * (oneSpinCSiteElems * oneSpinCSiteElems + 2 * oneSpinCSiteElems) * sizeof(Complex);

  double flopTwoSpinMdir  = CVolume * (8 * twoSpinCSiteElems * twoSpinCSiteElems - 2 * twoSpinCSiteElems);
  double byteTwoSpinMdir  = CVolume * (twoSpinCSiteElems * twoSpinCSiteElems + 2 * twoSpinCSiteElems) * sizeof(Complex);

  // Mdiag

  double flopOriginalMdiag = flopOriginalMdir;
  double byteOriginalMdiag = byteOriginalMdir;

  double flopOneSpinMdiag  = flopOneSpinMdir;
  double byteOneSpinMdiag  = byteOneSpinMdir;

  double flopTwoSpinMdiag  = flopTwoSpinMdir;
  double byteTwoSpinMdiag  = byteTwoSpinMdir;

  // blockProject

  double flopOriginalProject = FVolume * (8 * FSiteElems) * nBasis;
  double byteOriginalProject = FVolume * (2 * 1 + 2 * FSiteElems) * nBasis * sizeof(Complex);

  double flopOneSpinProject  = FVolume * (8 * FSiteElems) * nBasis;
  double byteOneSpinProject  = FVolume * (2 * 1 + 2 * FSiteElems) * nBasis * sizeof(Complex);

  double flopTwoSpinProject  = FVolume * (8 * FSiteElems) * nB;
  double byteTwoSpinProject  = FVolume * (2 * 1 + 2 * FSiteElems) * nB * sizeof(Complex);

  // blockPromote

  double flopOriginalPromote = FVolume * (8 * (nBasis - 1) + 6) * FSiteElems;
  double byteOriginalPromote = FVolume * ((1 * 1 + 3 * FSiteElems) * (nBasis - 1) + (1 * 1 + 2 * FSiteElems) * 1) * sizeof(Complex);

  double flopOneSpinPromote  = FVolume * (8 * (nBasis - 1) + 6) * FSiteElems;
  double byteOneSpinPromote  = FVolume * ((1 * 1 + 3 * FSiteElems) * (nBasis - 1) + (1 * 1 + 2 * FSiteElems) * 1) * sizeof(Complex);

  double flopTwoSpinPromote  = FVolume * (8 * (nB - 1) + 6) * FSiteElems;
  double byteTwoSpinPromote  = FVolume * ((1 * 1 + 3 * FSiteElems) * (nB - 1) + (1 * 1 + 2 * FSiteElems) * 1) * sizeof(Complex);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark CoarsenOperator" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  // TODO: Values for flop and byte here
  BenchmarkFunction(originalCoarsenedMatrix.CoarsenOperator, 0, 0, nIter, FGrid, LinOp, originalAggregates);
  BenchmarkFunction(oneSpinCoarsenedMatrix.CoarsenOperator, 0, 0, nIter, FGrid, LinOp, oneSpinAggregates);
  BenchmarkFunction(twoSpinCoarsenedMatrix.CoarsenOperator, 0, 0, nIter, FGrid, LinOp, twoSpinAggregates);

  // NOTE: For these comparisons to work at the current state of development, the call to blockOrthogonalise in OriginalCoarsenedMatrix::CoarsenOperator must be commented!
  std::cout << GridLogMessage << "Deviations of one-spin layout from original layout" << std::endl;
  for (int p = 0; p < originalCoarsenedMatrix.geom.npoint; ++p) {
    compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarsenedMatrix.A[p], oneSpinCoarsenedMatrix._Y[p]);
  }
  std::cout << GridLogMessage << "Deviations of two-spin layout from original layout" << std::endl;
  for(int p = 0; p < originalCoarsenedMatrix.geom.npoint; ++p) {
    compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarsenedMatrix.A[p], twoSpinCoarsenedMatrix._Y[p]);
  }

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark coarse M" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalCoarseVectorOut = zero;
  oneSpinCoarseVectorOut  = zero;
  twoSpinCoarseVectorOut  = zero;

  BenchmarkFunction(originalCoarsenedMatrix.M, flopOriginalM, byteOriginalM, nIter, originalCoarseVectorIn, originalCoarseVectorOut);
  BenchmarkFunction(oneSpinCoarsenedMatrix.M,  flopOneSpinM,  byteOneSpinM,  nIter, oneSpinCoarseVectorIn,  oneSpinCoarseVectorOut);
  BenchmarkFunction(twoSpinCoarsenedMatrix.M,  flopTwoSpinM,  byteTwoSpinM,  nIter, twoSpinCoarseVectorIn,  twoSpinCoarseVectorOut);

  compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarseVectorOut, oneSpinCoarseVectorOut);
  compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarseVectorOut, twoSpinCoarseVectorOut);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark coarse Mdag" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalCoarseVectorOut = zero;
  oneSpinCoarseVectorOut  = zero;
  twoSpinCoarseVectorOut  = zero;

  BenchmarkFunction(originalCoarsenedMatrix.Mdag, flopOriginalMdag, byteOriginalMdag, nIter, originalCoarseVectorIn, originalCoarseVectorOut);
  BenchmarkFunction(oneSpinCoarsenedMatrix.Mdag,  flopOneSpinMdag,  byteOneSpinMdag,  nIter, oneSpinCoarseVectorIn,  oneSpinCoarseVectorOut);
  BenchmarkFunction(twoSpinCoarsenedMatrix.Mdag,  flopTwoSpinMdag,  byteTwoSpinMdag,  nIter, twoSpinCoarseVectorIn,  twoSpinCoarseVectorOut);

  compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarseVectorOut, oneSpinCoarseVectorOut);
  compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarseVectorOut, twoSpinCoarseVectorOut);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark coarse Mdir" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalCoarseVectorOut = zero;
  oneSpinCoarseVectorOut  = zero;
  twoSpinCoarseVectorOut  = zero;

  BenchmarkFunction(originalCoarsenedMatrix.Mdir, flopOriginalMdir, byteOriginalMdir, nIter, originalCoarseVectorIn, originalCoarseVectorOut, 2, 1);
  BenchmarkFunction(oneSpinCoarsenedMatrix.Mdir,  flopOneSpinMdir,  byteOneSpinMdir,  nIter, oneSpinCoarseVectorIn,  oneSpinCoarseVectorOut,  2, 1);
  BenchmarkFunction(twoSpinCoarsenedMatrix.Mdir,  flopTwoSpinMdir,  byteTwoSpinMdir,  nIter, twoSpinCoarseVectorIn,  twoSpinCoarseVectorOut,  2, 1);

  compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarseVectorOut, oneSpinCoarseVectorOut);
  compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarseVectorOut, twoSpinCoarseVectorOut);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark coarse Mdiag" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalCoarseVectorOut = zero;
  oneSpinCoarseVectorOut  = zero;
  twoSpinCoarseVectorOut  = zero;

  BenchmarkFunction(originalCoarsenedMatrix.Mdiag, flopOriginalMdiag, byteOriginalMdiag, nIter, originalCoarseVectorIn, originalCoarseVectorOut);
  BenchmarkFunction(oneSpinCoarsenedMatrix.Mdiag,  flopOneSpinMdiag,  byteOneSpinMdiag,  nIter, oneSpinCoarseVectorIn,  oneSpinCoarseVectorOut);
  BenchmarkFunction(twoSpinCoarsenedMatrix.Mdiag,  flopTwoSpinMdiag,  byteTwoSpinMdiag,  nIter, twoSpinCoarseVectorIn,  twoSpinCoarseVectorOut);

  compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarseVectorOut, oneSpinCoarseVectorOut);
  compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarseVectorOut, twoSpinCoarseVectorOut);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark projectToSubspace" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalCoarseVectorOut = zero;
  oneSpinCoarseVectorOut  = zero;
  twoSpinCoarseVectorOut  = zero;

  BenchmarkFunction(originalAggregates.ProjectToSubspace, flopOriginalProject, byteOriginalProject, nIter, originalCoarseVectorOut, originalFineVectorIn);
  BenchmarkFunction(oneSpinAggregates.ProjectToSubspace,  flopOneSpinProject,  byteOneSpinProject,  nIter, oneSpinCoarseVectorOut,  oneSpinFineVectorIn);
  BenchmarkFunction(twoSpinAggregates.ProjectToSubspace,  flopTwoSpinProject,  byteTwoSpinProject,  nIter, twoSpinCoarseVectorOut,  twoSpinFineVectorIn);

  compareToResultInOriginalLayout<OneSpinCoarsenedMatrix>(originalCoarseVectorOut, oneSpinCoarseVectorOut);
  compareToResultInOriginalLayout<TwoSpinCoarsenedMatrix>(originalCoarseVectorOut, twoSpinCoarseVectorOut);

  std::cout << GridLogMessage << "***************************************************************************" << std::endl;
  std::cout << GridLogMessage << "Performance figures and result comparison for benchmark promoteFromSubspace" << std::endl;
  std::cout << GridLogMessage << "***************************************************************************" << std::endl;

  originalFineVectorOut = zero;
  oneSpinFineVectorOut  = zero;
  twoSpinFineVectorOut  = zero;

  BenchmarkFunction(originalAggregates.PromoteFromSubspace, flopOriginalPromote, byteOriginalPromote, nIter, originalCoarseVectorIn, originalFineVectorOut);
  BenchmarkFunction(oneSpinAggregates.PromoteFromSubspace,  flopOneSpinPromote,  byteOneSpinPromote,  nIter, oneSpinCoarseVectorIn,  oneSpinFineVectorOut);
  BenchmarkFunction(twoSpinAggregates.PromoteFromSubspace,  flopTwoSpinPromote,  byteTwoSpinPromote,  nIter, twoSpinCoarseVectorIn,  twoSpinFineVectorOut);

  printDeviationFromReference(originalFineVectorOut, oneSpinFineVectorOut);
  printDeviationFromReference(originalFineVectorOut, twoSpinFineVectorOut);

  Grid_finalize();
}
