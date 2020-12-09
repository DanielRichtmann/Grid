/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_clover_speedup.cc

    Copyright (C) 2015 - 2020

    Author: Daniel Richtmann <daniel.richtmann@gmail.com>
            Nils Meyer <nils.meyer@ur.de>

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


// needed below
#define VECTOR_VIEW_OPEN(l,v,mode)				\
  Vector< decltype(l[0].View(mode)) > v; v.reserve(l.size());	\
  for(int k=0;k<l.size();k++)				\
    v.push_back(l[k].View(mode));
#define VECTOR_VIEW_CLOSE(v)				\
  for(int k=0;k<v.size();k++) v[k].ViewClose();


int readFromCommandLineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}


std::vector<int> readFromCommandlineIvec(int*                    argc,
                                         char***                 argv,
                                         std::string&&           option,
                                         const std::vector<int>& defaultValue) {
  std::string      arg;
  std::vector<int> ret(defaultValue);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionIntVector(arg, ret);
  }
  return ret;
}


#define grid_printf(...) \
{\
  char _buf[1024];\
  sprintf(_buf, __VA_ARGS__);\
  std::cout << GridLogMessage << _buf;\
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


// need to get types depending on the types we feed in correct
template<typename vtype>       struct ChiralHalf {};
template<typename vtype>       struct ChiralHalf<iSpinColourVector<vtype>>   { typedef iHalfSpinColourVector<vtype> type; };
template<typename vtype,int N> struct ChiralHalf<iVector<iSinglet<vtype>,N>> { typedef iVector<iSinglet<vtype>,N/2> type; };


// hitting cases
template<class vtype,int which> accelerator_inline
typename std::enable_if<isSpinor<iVector<vtype,Ns>>::value && (which == 0 || which == 1), void>::type
extractChirality(iVector<vtype,Nhs>& half, const iVector<vtype,Ns>& full) {
  half(which+0) = full(which+0);
  half(which+1) = full(which+1);
}
template<class vtype,int nbasis,int which> accelerator_inline
typename std::enable_if<isCoarsened<iVector<vtype,nbasis>>::value && nbasis%2 == 0 && (which == 0 || which == 1), void>::type
extractChirality(iVector<vtype,nbasis/2>& half, const iVector<vtype,nbasis>& full) {
  const int nb=nbasis/2;
  const int start=which*nb;
  for(int n=0; n<nb; n++) {
    half(start+n) = full(start+n);
  }
}


// other cases
template<class rtype,class vtype,int N,int which> accelerator_inline
typename std::enable_if<!isSpinor<iVector<vtype,N>>::value && !isCoarsened<iVector<vtype,N>>::value, void>::type
extractChirality(iVector<rtype,N>& half, const iVector<vtype,N>& full) {
  for(int i=0;i<N;i++) {
    extractChirality<which>(half._internal[i],full._internal[i]);
  }
}
template<class rtype,class vtype,int which> accelerator_inline
void extractChirality(iScalar<rtype>& half, const iScalar<vtype>& full) {
  extractChirality<which>(half._internal,full._internal);
}
template<class rtype,class vtype,int N,int which> accelerator_inline
void extractChirality(iMatrix<rtype,N>& half, const iMatrix<vtype,N>& full) {
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      extractChirality<which>(half._internal[i][j],full._internal[i][j]);
    }}
}


// converting run time to compile time parameter
template<class htype, class ftype>
accelerator_inline void extractChirality(htype& half, const ftype& full, int which) {
  if      (which == 0) extractChirality<ftype,0>(half, full);
  else if (which == 1) extractChirality<ftype,1>(half, full);
  else assert(0);
}

  // iScalar<T> ret;
  // ret._internal = a;
  // return ret;

template<typename vobj, typename std::enable_if<isGridFundamental<vobj>::value, vobj>::type* = nullptr>
accelerator_inline iScalar<vobj> getUpperIpElem(const iScalar<iVector<iScalar<vobj>, 2>>& in)
{
  iScalar<vobj> ret;
  ret._internal = TensorRemove(in()(0));
  return ret;
}
template<typename vobj, typename std::enable_if<isGridFundamental<vobj>::value, vobj>::type* = nullptr>
accelerator_inline iScalar<vobj> getLowerIpElem(const iScalar<iVector<iScalar<vobj>, 2>>& in)
{
  iScalar<vobj> ret;
  ret._internal = TensorRemove(in()(1));
  return ret;
}
template<typename vobj, typename std::enable_if<isGridFundamental<vobj>::value, vobj>::type* = nullptr>
accelerator_inline iScalar<vobj> getUpperIpElem(const iVector<iSinglet<vobj>, 2>& in)
{
  iScalar<vobj> ret;
  ret._internal = TensorRemove(in(0)());
  return ret;
}
template<typename vobj, typename std::enable_if<isGridFundamental<vobj>::value, vobj>::type* = nullptr>
accelerator_inline iScalar<vobj> getLowerIpElem(const iVector<iSinglet<vobj>, 2>& in)
{
  iScalar<vobj> ret;
  ret._internal = TensorRemove(in(1)());
  return ret;
}

template<class vobj,class CComplex,int nbasis,class VLattice>
inline void blockProjectChiralityAware(Lattice<iVector<CComplex,nbasis > > &coarse,
			               const             Lattice<vobj>   &fine,
			               const VLattice &Basis)
{
  static_assert(nbasis%2 == 0, "Wrong basis size");

  const int nchiralities = 2;
  const int nvectors = nbasis/nchiralities;

  assert(Basis.size() == nvectors);

  GridBase *fine_grid   = fine.Grid();
  GridBase *coarse_grid = coarse.Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);

  autoView(fine_v,fine,AcceleratorRead);
  autoView(coarse_v,coarse,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN(Basis,basis_v,AcceleratorRead);

  accelerator_for(_idx, nvectors*coarse_osites, vobj::Nsimd(), {
    auto idx       = _idx;
    auto basis_i   = idx % nvectors;     idx  /= nvectors;
    auto sc        = idx % coarse_osites; idx /= coarse_osites;

    decltype(innerProductChiralityAwareD2(basis_v[0](0), fine_v(0))) reduce = Zero();

    // for(long j=0; j<sizes_v[sc]; ++j) {
    //   long sf = lut_v[sc][j];
    for(long j=0; j<1; ++j) {
      long sf = j;
      reduce = reduce + innerProductChiralityAwareD2(basis_v[basis_i](sf), fine_v(sf));
    }

    convertType(coarse_v[sc](basis_i),            getUpperIpElem(reduce));
    convertType(coarse_v[sc](basis_i + nvectors), getLowerIpElem(reduce));
  });
  VECTOR_VIEW_CLOSE(basis_v);

  std::cout << coarse_v[0] << std::endl;
}


template<typename vCoeff_t>
void runBenchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // Compile-time constants
  const int nbasis = NBASIS; static_assert((nbasis & 0x1) == 0, "");
  const int nsingle = nbasis/2;

  // setup grids
  GridCartesian* UGrid_f =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--fgrid", {8, 8, 8, 8}),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());
  GridCartesian* UGgrid_c =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--cgrid", {4, 4, 4, 4}),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG(UGrid_f);
  pRNG.SeedFixedIntegers(seeds);

  // type definitions
  typedef Lattice<iSpinColourVector<vCoeff_t>>                        FineVector;
  typedef Lattice<typename FineVector::vector_object::tensor_reduced> FineComplex;
  typedef Lattice<iVector<iSinglet<vCoeff_t>, nbasis>>                CoarseVector;
  // typedef WilsonImpl<vCoeff_t, FundamentalRepresentation, CoeffReal> WImpl;
  // typedef WilsonCloverFermion<WImpl> WilsonCloverOperator;
  // typedef typename WilsonCloverOperator::FermionField Fermion;
  // typedef typename WilsonCloverOperator::GaugeField Gauge;

  // setup fields
  FineVector src(UGrid_f); random(pRNG, src);
  FineVector ref(UGrid_f); ref = Zero();
  CoarseVector res(UGrid_f); res = Zero();
  FineVector diff(UGrid_f); diff = Zero();
  std::vector<FineVector>   basis(nsingle, UGrid_f);

  // randomize
  for(auto& b : basis) gaussian(pRNG, b);
  gaussian(pRNG, src);

  // randomize
  for(int n=0; n<basis.size(); n++)
    basis[n] = n+1;
  src = 1.0;
  // for(auto& b : basis) gaussian(pRNG, b);
  // gaussian(pRNG, src);

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid_f->_fdimensions[mu];

  // testing
  blockProjectChiralityAware(res, src, basis);

#if 0
  // testing
  {
    autoView(src_v, src, CpuRead);
    autoView(res_v, res, CpuWrite);
    autoView(ref_v, ref, CpuWrite);

    auto src_t = src_v[0];
    auto res_t = res_v[0];
    auto ref_t = ref_v[0];

    decltype(innerProductD2(src_v[0], src_v[0])) orig = Zero();
    orig = orig + innerProductD2(src_v[0], src_v[0]);
    char* dtype_orig = orig;
    convertType(res_v[0](0), TensorRemove(orig));

    decltype(innerProductChiralityAwareD2(src_v[0], src_v[0])) test = Zero();
    test = test + innerProductChiralityAwareD2(src_v[0], src_v[0]);
    char* dtype_test = test;
    // convertType(res_v[0](0), TensorRemove(test));
    convertType(res_v[0](0), TensorRemove(test()(0)));
    convertType(res_v[0](0), TensorRemove(test()(1)));

    // decltype(innerProductD2(basis_v[0](0), fine_v[0](0))) reduce = Zero();
    // for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
    //   for(long j=0; j<sizes_v[sc]; ++j) {
    //     long sf = lut_v[sc][j];
    //     reduce = reduce + innerProductD2(basis_v[basis_i_rel*fine_n_virtual + fine_virtual_i](sf), fine_v[vec_i*fine_n_virtual + fine_virtual_i](sf));

    // auto test = innerProductChiralityAwareD2(src_t, src_t);

    // std::cout << orig << std::endl;
  }

  // performance per site (use minimal values necessary)
  double hop_flop_per_site            = 1320; // Rich's Talk + what Peter uses
  double hop_byte_per_site            = (8 * 9 + 9 * 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_flop_per_site           = 504; // Rich's Talk and 1412.2629
  double clov_byte_per_site           = (2 * 18 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_byte_per_site_performed = (12 * 12 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;

  // total performance numbers
  double hop_gflop_total            = volume * nIter * hop_flop_per_site / 1e9;
  double hop_gbyte_total            = volume * nIter * hop_byte_per_site / 1e9;
  double clov_gflop_total           = volume * nIter * clov_flop_per_site / 1e9;
  double clov_gbyte_total           = volume * nIter * clov_byte_per_site / 1e9;
  double clov_gbyte_performed_total = volume * nIter * clov_byte_per_site_performed / 1e9;

  // output
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "hop", precision.c_str(), secs_hop, hop_gflop_total/secs_hop, hop_gbyte_total/secs_hop, secs_ref/secs_hop, secs_hop/secs_hop);
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "reference", precision.c_str(), secs_ref, clov_gflop_total/secs_ref, clov_gbyte_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop);

  // just so we see how well the ET performs in terms of traffic
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "reference_performed", precision.c_str(), secs_ref, clov_gflop_total/secs_ref, clov_gbyte_performed_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop);


  grid_printf("finalize %s\n", precision.c_str()); fflush(stdout);
#endif
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
