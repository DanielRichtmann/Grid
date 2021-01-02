/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_clover_speedup.cc

    Copyright (C) 2015 - 2020

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>
            Nils Meyer       <nils.meyer@ur.de>

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
#include "FasterWilsonCloverFermion.h"

using namespace Grid;


int readFromCommandLineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}


#define grid_printf(...) \
{\
  char _buf[1024];\
  sprintf(_buf, __VA_ARGS__);\
  std::cout << GridLogMessage << _buf;\
  fflush(stdout);\
}


template<typename Field>
bool resultsAgree(const Field& ref, const Field& res, const std::string& name) {
  RealD checkTolerance = (getPrecision<Field>::value == 1) ? 1e-7 : 1e-15;
  Field diff(ref.Grid());
  diff = ref - res;
  auto absDev = norm2(diff);
  auto relDev = absDev / norm2(ref);
  std::cout << GridLogMessage
            << "norm2(reference), norm2(" << name << "), abs. deviation, rel. deviation: " << norm2(ref) << " "
            << norm2(res) << " " << absDev << " " << relDev << " -> check "
            << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;

  return relDev <= checkTolerance;
}


template<typename vCoeff_t>
void runBenchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // setup grids
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vCoeff_t::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  // clang-format on

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG(UGrid);
  pRNG.SeedFixedIntegers(seeds);

  // type definitions
  typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffReal> WImpl;
  typedef WilsonCloverFermion<WImpl> WilsonCloverOperator;
  typedef FasterWilsonCloverFermion<WImpl> FasterWilsonCloverOperator;
  typedef typename WilsonCloverOperator::FermionField Fermion;
  typedef typename WilsonCloverOperator::GaugeField Gauge;

  // setup fields
  Fermion src(UGrid); random(pRNG, src);
  Fermion ref(UGrid); ref = Zero();
  Fermion res(UGrid); res = Zero();
  Fermion hop(UGrid); hop = Zero();
  Fermion diff(UGrid); diff = Zero();
  Gauge   Umu(UGrid); SU3::HotConfiguration(pRNG, Umu);

  // setup boundary phases
  typename WilsonCloverOperator::ImplParams implParams;
  std::vector<Complex> boundary_phases(Nd, 1.);
  if(GridCmdOptionExists(*argv, *argv + *argc, "--antiperiodic")) boundary_phases[Nd-1] = -1.;
  implParams.boundary_phases = boundary_phases;
  WilsonAnisotropyCoefficients anisParams;

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid->_fdimensions[mu];

  // setup fermion operators
  WilsonCloverOperator     Dwc(Umu, *UGrid, *UrbGrid, 0.5, 1.0, 1.0, anisParams, implParams);
  FasterWilsonCloverOperator Dwc_faster(Umu, *UGrid, *UrbGrid, 0.5, 1.0, 1.0, anisParams, implParams);

  // performance per site (use minimal values necessary)
  double hop_flop_per_site            = 1320; // Rich's Talk + what Peter uses
  double hop_byte_per_site            = (8 * 9 + 9 * 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_flop_per_site           = 504; // Rich's Talk and 1412.2629
  double clov_byte_per_site           = (2 * 18 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_flop_per_site_performed = 1128;
  double clov_byte_per_site_performed = (12 * 12 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;

  // total performance numbers
  double hop_gflop_total            = volume * nIter * hop_flop_per_site / 1e9;
  double hop_gbyte_total            = volume * nIter * hop_byte_per_site / 1e9;
  double clov_gflop_total           = volume * nIter * clov_flop_per_site / 1e9;
  double clov_gbyte_total           = volume * nIter * clov_byte_per_site / 1e9;
  double clov_gflop_performed_total = volume * nIter * clov_flop_per_site_performed / 1e9;
  double clov_gbyte_performed_total = volume * nIter * clov_byte_per_site_performed / 1e9;

  // warmup + measure dhop
  for(auto n : {1, 2, 3, 4, 5}) Dwc.Dhop(src, hop, 0);
  double t0 = usecond();
  for(int n = 0; n < nIter; n++) Dwc.Dhop(src, hop, 0);
  double t1 = usecond();
  double secs_hop = (t1-t0)/1e6;
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "hop", precision.c_str(), secs_hop, hop_gflop_total/secs_hop, hop_gbyte_total/secs_hop, 0.0, secs_hop/secs_hop);

#define BENCH_CLOVER_KERNEL(KERNEL) { \
  /* warmup + measure reference clover */ \
  for(auto n : {1, 2, 3, 4, 5}) Dwc.KERNEL(src, hop); \
  double t2 = usecond(); \
  for(int n = 0; n < nIter; n++) Dwc.KERNEL(src, hop); \
  double t3 = usecond(); \
  double secs_ref = (t3-t2)/1e6; \
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n", \
              "reference_"#KERNEL, precision.c_str(), secs_ref, clov_gflop_total/secs_ref, clov_gbyte_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop); \
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n", /* to see how well the ET performs */  \
              "reference_"#KERNEL"_performed", precision.c_str(), secs_ref, clov_gflop_performed_total/secs_ref, clov_gbyte_performed_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop); \
\
  /* warmup + measure improved clover */ \
  for(auto n : {1, 2, 3, 4, 5}) Dwc_faster.KERNEL(src, hop); \
  double t4 = usecond(); \
  for(int n = 0; n < nIter; n++) Dwc_faster.KERNEL(src, hop); \
  double t5 = usecond(); \
  double secs_res = (t5-t4)/1e6; \
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n", \
              "improved_"#KERNEL, precision.c_str(), secs_res, clov_gflop_total/secs_res, clov_gbyte_total/secs_res, secs_ref/secs_res, secs_res/secs_hop); \
}

  BENCH_CLOVER_KERNEL(Mooee);
  BENCH_CLOVER_KERNEL(MooeeDag);
  BENCH_CLOVER_KERNEL(MooeeInv);
  BENCH_CLOVER_KERNEL(MooeeInvDag);

  grid_printf("finalize %s\n", precision.c_str());
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
