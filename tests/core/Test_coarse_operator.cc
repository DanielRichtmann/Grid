/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_coarse_operator.cc

    Copyright (C) 2015 - 2020

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>

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


#ifndef NBASIS
#define NBASIS 40
#endif


int readFromCommandlineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
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
}
#define grid_printf_flush(...) \
{ \
  grid_printf(__VA_ARGS__); \
  fflush(stdout); \
}


template<typename vCoeff_t>
void run_benchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // compile-time constants
  const int nbasis = NBASIS; static_assert((nbasis & 0x1) == 0, "");
  const int nsingle = nbasis/2;

  // command line arguments
  const int nIter = readFromCommandlineInt(argc, argv, "--niter", 1000);
  const int nvec = readFromCommandlineInt(argc, argv, "--nvec", 1);

  // print info about run
  std::cout << GridLogMessage << "Compiled with nbasis = " << nbasis << " -> nb = " << nsingle << std::endl;

  // Grids
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
    GridDefaultLatt(), GridDefaultSimd(Nd, vCoeff_t::Nsimd()), GridDefaultMpi());
  UGrid->show_decomposition();
  GridRedBlackCartesian* UGridRB = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  UGridRB->show_decomposition();

  // print info about run
  grid_printf("\n");
  grid_printf("Coarse operator Benchmark with\n");
  grid_printf("fdimensions         : [%d %d %d %d]\n", UGrid->_fdimensions[0], UGrid->_fdimensions[1], UGrid->_fdimensions[2], UGrid->_fdimensions[3]);
  grid_printf("precision           : %s\n", precision.c_str());
  grid_printf("nbasis              : %d\n", nbasis);
  grid_printf("nsingle             : %d\n", nsingle);
  grid_printf("nvec                : %d\n", nvec);
  grid_printf("nsimd               : %d\n", vCoeff_t::Nsimd());
  grid_printf("acc_threads         : %d\n", acceleratorThreads());
  grid_printf("blk_threads         : %d\n", vCoeff_t::Nsimd()*acceleratorThreads());
  grid_printf_flush("\n");

  // setup rng
  GridParallelRNG pRNG(UGrid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  // type definitions
  typedef CoarsenedMatrix<vSpinColourVector, iSinglet<vCoeff_t>, nbasis> CoarseOperator;
  typedef typename CoarseOperator::CoarseVector                          CoarseVector;

  // setup coarse operator
  CoarseOperator op(*UGrid, *UGridRB, 0); for(auto& elem : op.A) gaussian(pRNG, elem);

  // setup fields -- fermions
  std::vector<CoarseVector> src_full(nvec, UGrid); for(auto& elem: src_full) random(pRNG, elem);
  std::vector<CoarseVector> res_full(nvec, UGrid); for(auto& elem: res_full) elem = Zero();
  std::vector<CoarseVector> src_e(nvec, UGridRB); for(int v=0; v<nvec; v++) pickCheckerboard(Even, src_e[v], src_full[v]);
  std::vector<CoarseVector> res_e(nvec, UGridRB);
  std::vector<CoarseVector> res_o(nvec, UGridRB);

  // point field aliases to correct fields
  std::vector<CoarseVector>& src_M           = src_full, res_M           = res_full;
  std::vector<CoarseVector>& src_M_overlapped_comms        = src_full, res_M_overlapped_comms        = res_full;
  std::vector<CoarseVector>& src_Mdag        = src_full, res_Mdag        = res_full;
  std::vector<CoarseVector>& src_Meooe       = src_e,    res_Meooe       = res_o;
  std::vector<CoarseVector>& src_MeooeDag    = src_e,    res_MeooeDag    = res_o;
  std::vector<CoarseVector>& src_Mooee       = src_e,    res_Mooee       = res_e;
  std::vector<CoarseVector>& src_MooeeDag    = src_e,    res_MooeeDag    = res_e;
  std::vector<CoarseVector>& src_MooeeInv    = src_e,    res_MooeeInv    = res_e;
  std::vector<CoarseVector>& src_MooeeInvDag = src_e,    res_MooeeInvDag = res_e;

  // misc stuff needed for performance figures
  double flops_per_cmul = 6;
  double flops_per_cadd = 2;
  double complex_words  = 2;
  double site_complex   = nbasis;
  double site_floats    = site_complex * complex_words;
  double prec_bytes     = getPrecision<vCoeff_t>::value * 4; // 4 for float, 8 for double
  double volume         = std::accumulate(UGrid->_fdimensions.begin(),UGrid->_fdimensions.end(),1,std::multiplies<int>());
  double volume_rb      = std::accumulate(UGridRB->_fdimensions.begin(),UGridRB->_fdimensions.end(),1,std::multiplies<int>());

  // performance figures -- M
  double flops_per_site_M = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * nvec;
  double words_per_site_M = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * nvec;
  double bytes_per_site_M = words_per_site_M * complex_words * prec_bytes;
  double flops_M          = flops_per_site_M * UGrid->gSites() * nIter;
  double words_M          = words_per_site_M * UGrid->gSites() * nIter;
  double nbytes_M         = bytes_per_site_M * UGrid->gSites() * nIter;

  // performance figures -- M_overlapped_comms
  double flops_per_site_M_overlapped_comms = flops_per_site_M;
  double words_per_site_M_overlapped_comms = words_per_site_M;
  double bytes_per_site_M_overlapped_comms = bytes_per_site_M;
  double flops_M_overlapped_comms          = flops_M;
  double words_M_overlapped_comms          = words_M;
  double nbytes_M_overlapped_comms         = nbytes_M;

  // performance figures -- Mdag
  double flops_per_site_Mdag = flops_per_site_M;
  double words_per_site_Mdag = words_per_site_M;
  double bytes_per_site_Mdag = bytes_per_site_M;
  double flops_Mdag          = flops_M;
  double words_Mdag          = words_M;
  double nbytes_Mdag         = nbytes_M;

  // performance figures -- Meooe
  double flops_per_site_Meooe = 1.0 * (2 * nbasis * (32 * nbasis - 1)) * nvec;
  double words_per_site_Meooe = 1.0 * (8 * nbasis + 8 * nbasis * nbasis + nbasis) * nvec;
  double bytes_per_site_Meooe = words_per_site_Meooe * complex_words * prec_bytes;
  double flops_Meooe          = flops_per_site_Meooe * UGridRB->gSites() * nIter;
  double words_Meooe          = words_per_site_Meooe * UGridRB->gSites() * nIter;
  double nbytes_Meooe         = bytes_per_site_Meooe * UGridRB->gSites() * nIter;

  // performance figures -- MeooeDag
  double flops_per_site_MeooeDag = flops_per_site_Meooe;
  double words_per_site_MeooeDag = words_per_site_Meooe;
  double bytes_per_site_MeooeDag = bytes_per_site_Meooe;
  double flops_MeooeDag          = flops_Meooe;
  double words_MeooeDag          = words_Meooe;
  double nbytes_MeooeDag         = nbytes_Meooe;

  // performance figures -- Mooee
  double flops_per_site_Mooee = 1.0 * (2 * nbasis * (4 * nbasis - 1)) * nvec;
  double words_per_site_Mooee = 1.0 * (nbasis + nbasis * nbasis + nbasis) * nvec;
  double bytes_per_site_Mooee = words_per_site_Mooee * complex_words * prec_bytes;
  double flops_Mooee          = flops_per_site_Mooee * UGridRB->gSites() * nIter;
  double words_Mooee          = words_per_site_Mooee * UGridRB->gSites() * nIter;
  double nbytes_Mooee         = bytes_per_site_Mooee * UGridRB->gSites() * nIter;

  // performance figures -- MooeeDag
  double flops_per_site_MooeeDag = flops_per_site_Mooee;
  double words_per_site_MooeeDag = words_per_site_Mooee;
  double bytes_per_site_MooeeDag = bytes_per_site_Mooee;
  double flops_MooeeDag          = flops_Mooee;
  double words_MooeeDag          = words_Mooee;
  double nbytes_MooeeDag         = nbytes_Mooee;

  // performance figures -- MooeeInv
  double flops_per_site_MooeeInv = flops_per_site_Mooee;
  double words_per_site_MooeeInv = words_per_site_Mooee;
  double bytes_per_site_MooeeInv = bytes_per_site_Mooee;
  double flops_MooeeInv          = flops_Mooee;
  double words_MooeeInv          = words_Mooee;
  double nbytes_MooeeInv         = nbytes_Mooee;

  // performance figures -- MooeeInvDag
  double flops_per_site_MooeeInvDag = flops_per_site_Mooee;
  double words_per_site_MooeeInvDag = words_per_site_Mooee;
  double bytes_per_site_MooeeInvDag = bytes_per_site_Mooee;
  double flops_MooeeInvDag          = flops_Mooee;
  double words_MooeeInvDag          = words_Mooee;
  double nbytes_MooeeInvDag         = nbytes_Mooee;

  // report calculated performance figures per site
#define PRINT_PER_SITE_VALUES(METHOD) {\
  grid_printf("%12s: per-site values: flops = %f, words = %f, bytes = %f, flops/bytes = %f\n",\
              #METHOD, flops_per_site_##METHOD, words_per_site_##METHOD, bytes_per_site_##METHOD, flops_per_site_##METHOD/bytes_per_site_##METHOD); \
  }
  PRINT_PER_SITE_VALUES(M);
  PRINT_PER_SITE_VALUES(Mdag);
  PRINT_PER_SITE_VALUES(Meooe);
  PRINT_PER_SITE_VALUES(MeooeDag);
  PRINT_PER_SITE_VALUES(Mooee);
  PRINT_PER_SITE_VALUES(MooeeDag);
  PRINT_PER_SITE_VALUES(MooeeInv);
  PRINT_PER_SITE_VALUES(MooeeInvDag);
#undef PRINT_PER_SITE_VALUES
  grid_printf_flush("\n");


#define BENCH_OPERATOR_METHOD(METHOD)\
  double secs_##METHOD;\
  {\
    for(int v=0; v<nvec; v++) res_##METHOD[v] = Zero();\
    grid_printf_flush("warmup %s %s\n", #METHOD, precision.c_str());\
    for(auto n : {1, 2, 3, 4, 5}) {\
      for(int v=0; v<nvec; v++)\
        op.METHOD(src_##METHOD[v], res_##METHOD[v]);\
    }\
    for(int v=0; v<nvec; v++) res_##METHOD[v] = Zero();\
    op.ZeroCounters();\
    grid_printf_flush("measurement %s %s\n", #METHOD, precision.c_str());\
    double t0 = usecond();\
    for(int n=0; n<nIter; n++) {\
      for(int v=0; v<nvec; v++)\
        op.METHOD(src_##METHOD[v], res_##METHOD[v]);\
    }\
    double t1 = usecond();\
    secs_##METHOD = (t1-t0)/1e6;\
  }

#define PRINT_OPERATOR_METHOD(METHOD) {\
  double GFlopsPerSec_##METHOD = flops_##METHOD  / secs_##METHOD / 1e9;\
  double GBPerSec_##METHOD     = nbytes_##METHOD / secs_##METHOD / 1e9;\
  grid_printf("%d applications of %s\n", nIter, #METHOD);\
  grid_printf("    Time to complete            : %f s\n",        secs_##METHOD);\
  grid_printf("    Total performance           : %f GFlops/s\n", GFlopsPerSec_##METHOD);\
  grid_printf("    Effective memory bandwidth  : %f GB/s\n",     GBPerSec_##METHOD);\
  grid_printf_flush("\n");\
  op.Report();\
  grid_printf_flush("\n");\
}

  BENCH_OPERATOR_METHOD(M);           PRINT_OPERATOR_METHOD(M);
  BENCH_OPERATOR_METHOD(M_overlapped_comms);           PRINT_OPERATOR_METHOD(M_overlapped_comms);
  BENCH_OPERATOR_METHOD(Mdag);        PRINT_OPERATOR_METHOD(Mdag);
  BENCH_OPERATOR_METHOD(Meooe);       PRINT_OPERATOR_METHOD(Meooe);
  BENCH_OPERATOR_METHOD(MeooeDag);    PRINT_OPERATOR_METHOD(MeooeDag);
  BENCH_OPERATOR_METHOD(Mooee);       PRINT_OPERATOR_METHOD(Mooee);
  BENCH_OPERATOR_METHOD(MooeeDag);    PRINT_OPERATOR_METHOD(MooeeDag);
  BENCH_OPERATOR_METHOD(MooeeInv);    PRINT_OPERATOR_METHOD(MooeeInv);
  BENCH_OPERATOR_METHOD(MooeeInvDag); PRINT_OPERATOR_METHOD(MooeeInvDag);

#undef BENCH_OPERATOR_METHOD
#undef PRINT_OPERATOR_METHOD

  grid_printf("DONE WITH COARSE_OP BENCHMARKS in %s precision\n", precision.c_str());
  grid_printf_flush("\n");
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  run_benchmark<vComplexF>(&argc, &argv);
  // run_benchmark<vComplexD>(&argc, &argv);

  Grid_finalize();
}
