/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./benchmarks/Benchmark_coarse_operator.cc

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
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Benchmark_coarse_operator
#ifndef NBASIS
#define NBASIS 40
#endif

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);

  std::vector<int> seeds({1, 2, 3, 4});

  // clang-format off
  const int nBasis          = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int nb              = nBasis / 2;
  int       nIter           = readFromCommandLineInt(&argc, &argv, "--niter", 10);
  bool      doPerfProfiling = readFromCommandLineToggle(&argc, &argv, "--perfprofiling");
  // NB: blocksize not needed here, since we can use the --grid switch for the coarse grid directly
  // clang-format on

  typedef CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis> CoarseDiracMatrix;
  typedef CoarseDiracMatrix::CoarseVector                       CoarseVector;

  GridCartesian *        CGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *CrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(CGrid);

  CGrid->show_decomposition();

  GridParallelRNG CPRNG(CGrid);
  CPRNG.SeedFixedIntegers(seeds);

  CoarseDiracMatrix CoarseMatrix(*CGrid, *CrbGrid);
  for(auto &elem : CoarseMatrix.A) random(CPRNG, elem);

  CoarseVector CoarseVecIn(CGrid);
  CoarseVector CoarseVecOut(CGrid);

  random(CPRNG, CoarseVecIn);
  random(CPRNG, CoarseVecOut);

  auto nStencil   = CoarseMatrix.geom.npoint;
  auto nAccum     = nStencil;
  auto CSiteElems = getSiteElems<decltype(CoarseVecIn)>();

  double CVolume = std::accumulate(CGrid->_fdimensions.begin(), CGrid->_fdimensions.end(), 1, std::multiplies<double>());

  double flopM      = CVolume * ((nStencil * (8 * CSiteElems * CSiteElems - 2 * CSiteElems) + nAccum * 2 * CSiteElems) + (8 * CSiteElems));
  double byteM      = CVolume * ((nStencil * (CSiteElems * CSiteElems + CSiteElems) + CSiteElems) + CSiteElems) * sizeof(Complex);
  double footprintM = CVolume * (2 * CSiteElems + nStencil * CSiteElems * CSiteElems) * sizeof(Complex);

  double flopMdir      = CVolume * (8 * CSiteElems * CSiteElems - 2 * CSiteElems);
  double byteMdir      = CVolume * (CSiteElems * CSiteElems + 2 * CSiteElems) * sizeof(Complex);
  double footprintMdir = CVolume * (2 * CSiteElems + nStencil * CSiteElems * CSiteElems) * sizeof(Complex);

  BenchmarkFunction(CoarseMatrix.M,     flopM,    byteM,    nIter, CoarseVecIn, CoarseVecOut);
  BenchmarkFunction(CoarseMatrix.Mdag,  flopM,    byteM,    nIter, CoarseVecIn, CoarseVecOut); // TODO: with the current implementation of Mdag, this line is not correct
  BenchmarkFunction(CoarseMatrix.Mdir,  flopMdir, byteMdir, nIter, CoarseVecIn, CoarseVecOut, 2, 1);
  BenchmarkFunction(CoarseMatrix.Mdiag, flopMdir, byteMdir, nIter, CoarseVecIn, CoarseVecOut);

  if (doPerfProfiling) {
    PerfProfileFunction(CoarseMatrix.M,     nIter, CoarseVecIn, CoarseVecOut);
    PerfProfileFunction(CoarseMatrix.Mdag,  nIter, CoarseVecIn, CoarseVecOut); // TODO: with the current implementation of Mdag, this line is not correct
    PerfProfileFunction(CoarseMatrix.Mdir,  nIter, CoarseVecIn, CoarseVecOut, 2, 1);
    PerfProfileFunction(CoarseMatrix.Mdiag, nIter, CoarseVecIn, CoarseVecOut);
  }

  Grid_finalize();
}
