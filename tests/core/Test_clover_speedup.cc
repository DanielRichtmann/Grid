/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_clover_speedup.cc

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

template<class Impl>
class CloverTermFast;


accelerator_inline int index2(int i, int j) {
  const int Nred = 6;
  if (i == j)
    return i;
  else if (i < j)
    return Nred + Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
  else
    return Nred + Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
}


template<class CloverFullField, class CloverReducedField>
void convert_clover(const CloverFullField& clover_full, CloverReducedField& clover_red) {
  autoView(clover_full_v, clover_full, AcceleratorRead);
  autoView(clover_red_v, clover_red, AcceleratorWrite);

  accelerator_for(ss, clover_full.Grid()->oSites(), 1, {
    for(int s_row = 0; s_row < Ns; s_row++) {
      for(int s_col = 0; s_col < Ns; s_col++) {
        if(abs(s_row - s_col) > 1 || s_row + s_col == 3) continue;
        int block       = s_row / Nhs;
        int s_row_block = s_row % Nhs;
        int s_col_block = s_col % Nhs;
        for(int c_row = 0; c_row < Nc; c_row++) {
          for(int c_col = 0; c_col < Nc; c_col++) {
            int i = s_row_block * Nc + c_row;
            int j = s_col_block * Nc + c_col;
            if(i > j)
              continue;
            else
              clover_red_v[ss]()(block)(index2(i, j)) = clover_full_v[ss]()(s_row, s_col)(c_row, c_col);
          }
        }
      }
    }
  });
}


template<class Impl>
class CloverTermFast {
  /////////////////////////////////////////////
  // Type definitions
  /////////////////////////////////////////////

public:

  INHERIT_IMPL_TYPES(Impl);
  static_assert(Nd == 4 && Nc == 3 && Ns == 4 && Impl::Dimension == 3, "Wrong dimensions");
  template<typename vtype>
  using iImplCloverReduced = iScalar<iVector<iVector<vtype, 21>, 2>>; // TODO: real numbers on diagonal
  typedef iImplCloverReduced<Simd> SiteCloverReduced;
  typedef Lattice<SiteCloverReduced> CloverReducedField;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:

  CloverTermFast(WilsonCloverFermion<Impl>& clover_full)
    : clov(clover_full.GaugeGrid())
  {
    convert_clover(clover_full.CloverTerm, clov);
  }

  void Mooee(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov.Grid(), in.Grid());
    out.Checkerboard() = in.Checkerboard();

    autoView(clov_v, clov, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);

    // third version
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    accelerator_for(ss, clov.Grid()->oSites(), Simd::Nsimd(), {
      calcSpinor res = Zero();
      calcSpinor in_t = in_v(ss);
      auto clov_t = clov_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          for(int j=0; j<Nred; j++) {
            int si = s_start + i/Nc, ci = i%Nc;
            int sj = s_start + j/Nc, cj = j%Nc;
            if (i <= j) {
              res()(si)(ci) = res()(si)(ci) + clov_t()(block)(index2(i, j)) * in_t()(sj)(cj);
            } else {
              res()(si)(ci) = res()(si)(ci) + conjugate(clov_t()(block)(index2(i, j))) * in_t()(sj)(cj);
            }
          }
        }
      }
      coalescedWrite(out_v[ss], res);
    });

    // NOTE:
    // - The trend seems to be that the larger the lattice is, the more version 2 and 3 outperform version 1
    // - Yet, I still don't reach the theoretical factor 3.4
  }

  accelerator_inline int index(int i, int j) const {
    if (i == j)
      return i;
    else if (i < j)
      return Nred + Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
    else
      return Nred + Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
  }

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:

  CloverReducedField clov;
  static constexpr int Nred = Nc * Nhs;
};


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
}


int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // setup grids
  GridCartesian*         UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  // clang-format on

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG(UGrid);
  pRNG.SeedFixedIntegers(seeds);

  // setup fields
  LatticeFermion    src(UGrid); random(pRNG, src);
  LatticeFermion    ref(UGrid); ref = Zero();
  LatticeFermion    res(UGrid); res = Zero();
  LatticeFermion    hop(UGrid); hop = Zero();
  LatticeFermion    diff(UGrid); diff = Zero();
  LatticeGaugeField Umu(UGrid); SU3::HotConfiguration(pRNG, Umu);

  // setup boundary phases
  typename WilsonCloverFermionR::ImplParams implParams;
  std::vector<Complex> boundary_phases(Nd, 1.);
  if(GridCmdOptionExists(argv, argv + argc, "--antiperiodic")) boundary_phases[Nd-1] = -1.;
  implParams.boundary_phases = boundary_phases;
  WilsonAnisotropyCoefficients anisParams;

  // setup fermion operators
  double t_a = usecond();
  WilsonCloverFermionR Dwc(Umu, *UGrid, *UrbGrid, 0.5, 1.0, 1.0, anisParams, implParams);
  double t_b = usecond();
  CloverTermFast<WilsonImplR> Dwc_fast(Dwc);
  double t_c = usecond();
  grid_printf("Clover term setup times: reference %f s, improved %f s\n", (t_b-t_a)/1e6, (t_c-t_b)/1e6);

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(&argc, &argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid->_fdimensions[mu];

  // warmup dhop
  for(auto n : {1, 2, 3, 4, 5}) Dwc.Dhop(src, hop, 0);

  // measure dhop
  double t0 = usecond();
  for(int n = 0; n < nIter; n++) Dwc.Dhop(src, hop, 0);
  double t1 = usecond();

  // warmup reference clover
  for(auto n : {1, 2, 3, 4, 5}) Dwc.Mooee(src, ref);

  // measure reference clover
  double t2 = usecond();
  for(int n = 0; n < nIter; n++) Dwc.Mooee(src, ref);
  double t3 = usecond();

  // warmup fast clover
  for(auto n : {1, 2, 3, 4, 5}) Dwc_fast.Mooee(src, res);

  // measure fast clover
  double t4 = usecond();
  for(int n = 0; n < nIter; n++) Dwc_fast.Mooee(src, res);
  double t5 = usecond();

  // verify correctness
  auto checkTolerance = 1e-15;
  diff = ref - res;
  auto absDev = norm2(diff);
  auto relDev = absDev / norm2(ref);
  std::cout << GridLogMessage
            << "norm2(reference), norm2(fast), abs. deviation, rel. deviation: " << norm2(ref) << " "
            << norm2(res) << " " << absDev << " " << relDev << " -> check "
            << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;
  assert(relDev <= checkTolerance);

  // performance per site (use minimal values necessary)
  double hop_flop_per_site = 1320; // Rich's Talk + what Peter uses
  double hop_byte_per_site = (8 * 9 + 9 * 12) * 2 * getPrecision<LatticeFermion>::value * 4;
  double clov_flop_per_site = 504; // Rich's Talk and 1412.2629
  double clov_byte_per_site = (2 * 18 + 12 + 12) * 2 * getPrecision<LatticeFermion>::value * 4;

  // total performance numbers
  double hop_gflop_total = volume * nIter * hop_flop_per_site / 1e9;
  double hop_gbyte_total = volume * nIter * hop_byte_per_site / 1e9;
  double clov_gflop_total = volume * nIter * clov_flop_per_site / 1e9;
  double clov_gbyte_total = volume * nIter * clov_byte_per_site / 1e9;
  double secs_hop = (t1-t0)/1e6;
  double secs_ref = (t3-t2)/1e6;
  double secs_res = (t5-t4)/1e6;

  // output
  grid_printf("Performance(%9s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "hop", secs_hop, hop_gflop_total/secs_hop, hop_gbyte_total/secs_hop, secs_ref/secs_hop, secs_hop/secs_hop);
  grid_printf("Performance(%9s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "reference", secs_ref, clov_gflop_total/secs_ref, clov_gbyte_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop);
  grid_printf("Performance(%9s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "improved", secs_res, clov_gflop_total/secs_res, clov_gbyte_total/secs_res, secs_ref/secs_res, secs_res/secs_hop);

  Grid_finalize();
}
