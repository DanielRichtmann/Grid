/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/solver/Test_coarse_red_black.cc

    Copyright (C) 2015-2019

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

using namespace Grid;
using namespace Grid::QCD;

// Enable control of nbasis from the compiler command line
// NOTE to self: Copy the value of CXXFLAGS from the makefile and call make as follows:
//   make CXXFLAGS="-DNBASIS=24 VALUE_OF_CXXFLAGS_IN_MAKEFILE" Test_coarse_red_black
#ifndef NBASIS
#define NBASIS 40
#endif

std::vector<int> readFromCommandLineIntVec(int *argc, char ***argv, const std::string &option, const std::vector<int> &defaultValues) {
  std::string      arg;
  std::vector<int> ret(defaultValues);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionIntVector(arg, ret);
  }
  return ret;
}

std::vector<int> calcCoarseLattSize(const std::vector<int> &fineLattSize, const std::vector<int> &blockSize) {
  std::vector<int> ret(fineLattSize);
  for(int d = 0; d < ret.size(); d++) {
    ret[d] /= blockSize[d];
  }
  return ret;
}

// NOTE: These tests are written in analogy to tests/core/Test_wilson_clover.cc

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  std::cout << std::scientific;

  std::vector<int> seeds({1, 2, 3, 4});

  // clang-format off
  const int                 nBasis = NBASIS; static_assert((nBasis & 0x1) == 0, "");
  const int                 nB     = nBasis/2;
  const std::vector<int> blockSize = readFromCommandLineIntVec(&argc, &argv, "--blocksize", std::vector<int>({2, 2, 2, 2}));
  // clang-format on

  const std::vector<int> lattsize_f = GridDefaultLatt();
  const std::vector<int> lattsize_c = calcCoarseLattSize(lattsize_f, blockSize);

#if defined(USE_TWOSPIN_COARSENING)
  typedef TwoSpinCoarseningPolicy<Lattice<vSpinColourVector>, vComplex, nBasis/2> CoarseningPolicy;
  typedef AggregationUsingPolicies<CoarseningPolicy>                              Aggregates;
  typedef CoarsenedMatrixUsingPolicies<CoarseningPolicy>                          CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField                                CoarseVector;
#elif defined (USE_ONESPIN_COARSENING)
  typedef OriginalCoarseningPolicy<Lattice<vSpinColourVector>, vComplex, nBasis>  CoarseningPolicy;
  typedef AggregationUsingPolicies<CoarseningPolicy>                              Aggregates;
  typedef CoarsenedMatrixUsingPolicies<CoarseningPolicy>                          CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField                                CoarseVector;
#else
  typedef Aggregation<vSpinColourVector, vTComplex, nBasis>                       Aggregates;
  typedef CoarsenedMatrix<vSpinColourVector, vTComplex, nBasis>                   CoarseDiracMatrix;
  typedef CoarseDiracMatrix::CoarseVector                                         CoarseVector;
#endif

  GridCartesian *        Grid_f   = SpaceTimeGrid::makeFourDimGrid(lattsize_f, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridCartesian *        Grid_c   = SpaceTimeGrid::makeFourDimGrid(lattsize_c, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian *RBGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(Grid_f);
  GridRedBlackCartesian *RBGrid_c = SpaceTimeGrid::makeFourDimRedBlackGrid(Grid_c);

  std::cout << GridLogMessage << "Grid_f:" << std::endl; Grid_f->show_decomposition();
  std::cout << GridLogMessage << "Grid_c:" << std::endl; Grid_c->show_decomposition();
  std::cout << GridLogMessage << "RBGrid_f:" << std::endl; RBGrid_f->show_decomposition();
  std::cout << GridLogMessage << "RBGrid_c:" << std::endl; RBGrid_c->show_decomposition();

  GridParallelRNG pRNG_f(Grid_f);
  GridParallelRNG pRNG_c(Grid_c);

  pRNG_f.SeedFixedIntegers(seeds);
  pRNG_c.SeedFixedIntegers(seeds);

  LatticeGaugeField Umu(Grid_f);
  SU3::HotConfiguration(pRNG_f, Umu);

  RealD checkTolerance = 1e-15;

  RealD                                               mass = -0.1;
  RealD                                               csw  = 1.0;
  WilsonFermionR                                      Dw(Umu, *Grid_f, *RBGrid_f, mass);
  WilsonCloverFermionR                                Dwc(Umu, *Grid_f, *RBGrid_f, mass, csw, csw);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDw(Dw);
  MdagMLinearOperator<WilsonFermionR, LatticeFermion> MdagMOpDwc(Dwc);

  MdagMLinearOperator<WilsonFermionR, LatticeFermion>* MdagMOp = nullptr;

  if(GridCmdOptionExists(argv, argv + argc, "--doclover")) {
    MdagMOp = &MdagMOpDwc;
    std::cout << "Running tests for clover fermions" << std::endl;
  } else {
    MdagMOp = &MdagMOpDw;
    std::cout << "Running tests for wilson fermions" << std::endl;
  }

  // // setup with CG like in HDCR
  // Aggregates Aggs(Grid_c, Grid_f, 0);
  // Aggs.CreateSubspace(pRNG_f, MdagMOp, nB);
  // Aggs.DoChiralDoubling();

  // setup with GMRES like in Wilson MG
  Aggregates Aggs(Grid_c, Grid_f, 0);
  Aggs.CreateSubspaceDDalphaAMG(pRNG_f, *MdagMOp, true, nB, 4);
  Aggs.Orthogonalise();
  Aggs.DoChiralDoubling();

  CoarseDiracMatrix Dc(*Grid_c, *RBGrid_c);
  Dc.CoarsenOperator(Grid_f, *MdagMOp, Aggs);

  MdagMLinearOperator<CoarseDiracMatrix, CoarseVector> MdagMOp_Dc(Dc);

  LatticeFermion src_f(Grid_f); random(pRNG_f, src_f);
  CoarseVector src(Grid_c); random(pRNG_c, src);

  /////////////////////////////////////////////////////////////////////////////
  //                              Start of tests                             //
  /////////////////////////////////////////////////////////////////////////////

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Testing that Dhop + Ddiag = Dunprec                         " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector phi(Grid_c);   phi = zero;
    CoarseVector chi(Grid_c);   chi = zero;
    CoarseVector res(Grid_c);   res = zero;
    CoarseVector ref(Grid_c);   ref = zero;
    CoarseVector diff(Grid_c); diff = zero;
    // clang-format on

    // clang-format off
    Dc.Mdiag(src, phi);          std::cout << GridLogMessage << "Applied Mdiag" << std::endl;
    Dc.Dhop(src, chi, DaggerNo); std::cout << GridLogMessage << "Applied Dhop"  << std::endl;
    Dc.M(src, ref);              std::cout << GridLogMessage << "Applied M"     << std::endl;
    // clang-format on

    std::cout << GridLogDebug << "norm phi = " << norm2(phi) << " norm chi = " << norm2(chi) << " norm ref = " << norm2(ref) << std::endl;

    res = phi + chi;

    diff = ref - res;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(DUnprec), norm2(Dhop + Ddiag), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(res) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Testing that Deo + Doe = DhopUnprec                         " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector src_e(RBGrid_c); src_e = zero;
    CoarseVector src_o(RBGrid_c); src_o = zero;
    CoarseVector r_e(RBGrid_c);     r_e = zero;
    CoarseVector r_o(RBGrid_c);     r_o = zero;
    CoarseVector r_eo(Grid_c);     r_eo = zero;
    CoarseVector ref(Grid_c);       ref = zero;
    CoarseVector diff(Grid_c);     diff = zero;
    // clang-format on

    pickCheckerboard(Even, src_e, src);
    pickCheckerboard(Odd, src_o, src);

    // clang-format off
    Dc.Meooe(src_e, r_o);        std::cout << GridLogMessage << "Applied Meo"  << std::endl;
    Dc.Meooe(src_o, r_e);        std::cout << GridLogMessage << "Applied Moe"  << std::endl;
    Dc.Dhop(src, ref, DaggerNo); std::cout << GridLogMessage << "Applied Dhop" << std::endl;
    // clang-format on

    setCheckerboard(r_eo, r_o);
    setCheckerboard(r_eo, r_e);

    diff = ref - r_eo;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(DhopUnprec), norm2(Deo + Doe), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(r_eo) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test |(Im(v^dag D^dag D v)| = 0                             " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector tmp(Grid_c); tmp = zero;
    CoarseVector phi(Grid_c); phi = zero;
    // clang-format on

    // clang-format off
    Dc.M(src, tmp);    std::cout << GridLogMessage << "Applied M"    << std::endl;
    Dc.Mdag(tmp, phi); std::cout << GridLogMessage << "Applied Mdag" << std::endl;
    // clang-format on

    ComplexD dot = innerProduct(src, phi);

    auto relDev = std::abs(imag(dot)) / std::abs(real(dot));
    std::cout << GridLogMessage << "Re(v^dag D^dag D v), Im(v^dag D^dag D v), rel.deviation: "
              << real(dot) << " " << imag(dot) << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test |(Im(v^dag Dooee^dag Dooee v)| = 0 (full grid)         " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector tmp(Grid_c); tmp = zero;
    CoarseVector phi(Grid_c); phi = zero;
    // clang-format on

    // clang-format off
    Dc.Mooee(src, tmp);    std::cout << GridLogMessage << "Applied Mooee"    << std::endl;
    Dc.MooeeDag(tmp, phi); std::cout << GridLogMessage << "Applied MooeeDag" << std::endl;
    // clang-format on

    ComplexD dot = innerProduct(src, phi);

    auto relDev = std::abs(imag(dot)) / std::abs(real(dot));
    std::cout << GridLogMessage << "Re(v^dag Dooee^dag Dooee v), Im(v^dag Dooee^dag Dooee v), rel.deviation: "
              << real(dot) << " " << imag(dot) << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test DooeeInv Dooee = 1 (full grid)                         " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector tmp(Grid_c);   tmp = zero;
    CoarseVector phi(Grid_c);   phi = zero;
    CoarseVector diff(Grid_c); diff = zero;
    // clang-format on

    // clang-format off
    Dc.Mooee(src, tmp);    std::cout << GridLogMessage << "Applied Mooee"    << std::endl;
    Dc.MooeeInv(tmp, phi); std::cout << GridLogMessage << "Applied MooeeInv" << std::endl;
    // clang-format on

    diff        = src - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(src);
    std::cout << GridLogMessage << "norm2(src), norm2(MooeeInv Mooee src), abs. deviation, rel. deviation: "
              << norm2(src) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test Ddagger is the dagger of D by requiring                " << std::endl;
    std::cout << GridLogMessage << "=  < phi | Deo | chi > * = < chi | Deo^dag| phi>              " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector phi(Grid_c); random(pRNG_c, phi);
    CoarseVector chi(Grid_c); random(pRNG_c, chi);
    CoarseVector chi_e(RBGrid_c);   chi_e = zero;
    CoarseVector chi_o(RBGrid_c);   chi_o = zero;
    CoarseVector dchi_e(RBGrid_c); dchi_e = zero;
    CoarseVector dchi_o(RBGrid_c); dchi_o = zero;
    CoarseVector phi_e(RBGrid_c);   phi_e = zero;
    CoarseVector phi_o(RBGrid_c);   phi_o = zero;
    CoarseVector dphi_e(RBGrid_c); dphi_e = zero;
    CoarseVector dphi_o(RBGrid_c); dphi_o = zero;
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, phi_e, phi);
    pickCheckerboard(Odd, phi_o, phi);

    // clang-format off
    Dc.Meooe(chi_e, dchi_o);    std::cout << GridLogMessage << "Applied Meo"    << std::endl;
    Dc.Meooe(chi_o, dchi_e);    std::cout << GridLogMessage << "Applied Moe"    << std::endl;
    Dc.MeooeDag(phi_e, dphi_o); std::cout << GridLogMessage << "Applied MeoDag" << std::endl;
    Dc.MeooeDag(phi_o, dphi_e); std::cout << GridLogMessage << "Applied MoeDag" << std::endl;
    // clang-format on

    ComplexD phiDchi_e = innerProduct(phi_e, dchi_e);
    ComplexD phiDchi_o = innerProduct(phi_o, dchi_o);
    ComplexD chiDphi_e = innerProduct(chi_e, dphi_e);
    ComplexD chiDphi_o = innerProduct(chi_o, dphi_o);

    std::cout << GridLogDebug << "norm dchi_e = " << norm2(dchi_e) << " norm dchi_o = " << norm2(dchi_o) << " norm dphi_e = " << norm2(dphi_e)
              << " norm dphi_o = " << norm2(dphi_e) << std::endl;

    std::cout << GridLogMessage << "e " << phiDchi_e << " " << chiDphi_e << std::endl;
    std::cout << GridLogMessage << "o " << phiDchi_o << " " << chiDphi_o << std::endl;

    std::cout << GridLogMessage << "phiDchi_e - conj(chiDphi_o) " << phiDchi_e - conj(chiDphi_o) << std::endl;
    std::cout << GridLogMessage << "phiDchi_o - conj(chiDphi_e) " << phiDchi_o - conj(chiDphi_e) << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test MooeeInv Mooee = 1 (checkerboards separately)          " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector chi(Grid_c);   random(pRNG_c, chi);
    CoarseVector tmp(Grid_c);   tmp = zero;
    CoarseVector phi(Grid_c);   phi = zero;
    CoarseVector diff(Grid_c); diff = zero;
    CoarseVector chi_e(RBGrid_c); chi_e = zero;
    CoarseVector chi_o(RBGrid_c); chi_o = zero;
    CoarseVector phi_e(RBGrid_c); phi_e = zero;
    CoarseVector phi_o(RBGrid_c); phi_o = zero;
    CoarseVector tmp_e(RBGrid_c); tmp_e = zero;
    CoarseVector tmp_o(RBGrid_c); tmp_o = zero;
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, tmp_e, tmp);
    pickCheckerboard(Odd, tmp_o, tmp);

    // clang-format off
    Dc.Mooee(chi_e, tmp_e);    std::cout << GridLogMessage << "Applied Mee"    << std::endl;
    Dc.MooeeInv(tmp_e, phi_e); std::cout << GridLogMessage << "Applied MeeInv" << std::endl;
    Dc.Mooee(chi_o, tmp_o);    std::cout << GridLogMessage << "Applied Moo"    << std::endl;
    Dc.MooeeInv(tmp_o, phi_o); std::cout << GridLogMessage << "Applied MooInv" << std::endl;
    // clang-format on

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    diff = chi - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(chi);
    std::cout << GridLogMessage << "norm2(chi), norm2(MeeInv Mee chi), abs. deviation, rel. deviation: "
              << norm2(chi) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Test MooeeDag MooeeInvDag = 1 (checkerboards separately)    " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector chi(Grid_c);   random(pRNG_c, chi);
    CoarseVector tmp(Grid_c);   tmp = zero;
    CoarseVector phi(Grid_c);   phi = zero;
    CoarseVector diff(Grid_c); diff = zero;
    CoarseVector chi_e(RBGrid_c); chi_e = zero;
    CoarseVector chi_o(RBGrid_c); chi_o = zero;
    CoarseVector phi_e(RBGrid_c); phi_e = zero;
    CoarseVector phi_o(RBGrid_c); phi_o = zero;
    CoarseVector tmp_e(RBGrid_c); tmp_e = zero;
    CoarseVector tmp_o(RBGrid_c); tmp_o = zero;
    // clang-format on

    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);
    pickCheckerboard(Even, tmp_e, tmp);
    pickCheckerboard(Odd, tmp_o, tmp);

    // clang-format off
    Dc.MooeeDag(chi_e, tmp_e);    std::cout << GridLogMessage << "Applied MeeDag"    << std::endl;
    Dc.MooeeInvDag(tmp_e, phi_e); std::cout << GridLogMessage << "Applied MeeInvDag" << std::endl;
    Dc.MooeeDag(chi_o, tmp_o);    std::cout << GridLogMessage << "Applied MooDag"    << std::endl;
    Dc.MooeeInvDag(tmp_o, phi_o); std::cout << GridLogMessage << "Applied MooInvDag" << std::endl;
    // clang-format on

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    diff = chi - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(chi);
    std::cout << GridLogMessage << "norm2(chi), norm2(MeeDag MeeInvDag chi), abs. deviation, rel. deviation: "
              << norm2(chi) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Testing EO operator is equal to the unprec                  " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    // clang-format off
    CoarseVector chi(Grid_c);   chi = zero;
    CoarseVector phi(Grid_c);   phi = zero;
    CoarseVector ref(Grid_c);   ref = zero;
    CoarseVector diff(Grid_c); diff = zero;
    CoarseVector src_e(RBGrid_c); src_e = zero;
    CoarseVector src_o(RBGrid_c); src_o = zero;
    CoarseVector phi_e(RBGrid_c); phi_e = zero;
    CoarseVector phi_o(RBGrid_c); phi_o = zero;
    CoarseVector chi_e(RBGrid_c); chi_e = zero;
    CoarseVector chi_o(RBGrid_c); chi_o = zero;
    // clang-format on

    pickCheckerboard(Even, src_e, src);
    pickCheckerboard(Odd, src_o, src);
    pickCheckerboard(Even, phi_e, phi);
    pickCheckerboard(Odd, phi_o, phi);
    pickCheckerboard(Even, chi_e, chi);
    pickCheckerboard(Odd, chi_o, chi);

    // M phi = (Mooee src_e + Meooe src_o , Mooee src_o + Meooe src_e)

    Dc.M(src, ref); // Reference result from the unpreconditioned operator

    // EO matrix
    // clang-format off
    Dc.Mooee(src_e, chi_e); std::cout << GridLogMessage << "Applied Mee" << std::endl;
    Dc.Mooee(src_o, chi_o); std::cout << GridLogMessage << "Applied Moo" << std::endl;
    Dc.Meooe(src_o, phi_e); std::cout << GridLogMessage << "Applied Moe" << std::endl;
    Dc.Meooe(src_e, phi_o); std::cout << GridLogMessage << "Applied Meo" << std::endl;
    // clang-format on

    phi_o += chi_o;
    phi_e += chi_e;

    setCheckerboard(phi, phi_e);
    setCheckerboard(phi, phi_o);

    std::cout << GridLogDebug << "norm phi_e = " << norm2(phi_e) << " norm phi_o = " << norm2(phi_o) << " norm phi = " << norm2(phi) << std::endl;

    diff = ref - phi;
    auto absDev = norm2(diff);
    auto relDev = absDev / norm2(ref);
    std::cout << GridLogMessage << "norm2(Dunprec), norm2(Deoprec), abs. deviation, rel. deviation: "
              << norm2(ref) << " " << norm2(phi) << " " << absDev << " " << relDev
              << " -> check " << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  }

  {
    std::cout << GridLogMessage << "==============================================================" << std::endl;
    std::cout << GridLogMessage << "= Comparing EO solve is with unprec one                       " << std::endl;
    std::cout << GridLogMessage << "==============================================================" << std::endl;

    GridStopWatch Timer;

    RealD   solverTolerance = 1e-12;
    Integer maxIter         = 10000;
    Integer restartLength   = 25;

    // clang-format off
    CoarseVector resultCG(Grid_c);           resultCG = zero;
    CoarseVector resultRBCG(Grid_c);       resultRBCG = zero;
    CoarseVector resultGMRES(Grid_c);     resultGMRES = zero;
    CoarseVector resultRBGMRES(Grid_c); resultRBGMRES = zero;
    // clang-format on

    ConjugateGradient<CoarseVector> CG(solverTolerance, maxIter);
    SchurRedBlackDiagMooeeSolve<CoarseVector> RBCG(CG);
    GeneralisedMinimalResidual<CoarseVector> GMRES(solverTolerance, maxIter, restartLength);
    SchurRedBlackDiagMooeeNonHermSolve<CoarseVector> RBGMRES(GMRES);

    // clang-format off
    Timer.Reset(); Timer.Start();
    CG(MdagMOp_Dc, src, resultCG);
    Timer.Stop(); std::cout << "CG took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    RBCG(Dc, src, resultRBCG);
    Timer.Stop(); std::cout << "RBCG took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    GMRES(MdagMOp_Dc, src, resultGMRES);
    Timer.Stop(); std::cout << "GMRES took " << Timer.Elapsed() << std::endl;
    Timer.Reset(); Timer.Start();
    RBGMRES(Dc, src, resultRBGMRES);
    Timer.Stop(); std::cout << "RBGMRES took " << Timer.Elapsed() << std::endl;
    // clang-format on
  }

  Grid_finalize();
}
