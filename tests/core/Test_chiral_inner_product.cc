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


template<typename Field>
void performChiralDoubling(std::vector<Field>& basisVectors) {
  assert(basisVectors.size()%2 == 0);
  auto nb = basisVectors.size()/2;

  for(int n=0; n<nb; n++) {
    auto tmp1 = basisVectors[n];
    auto tmp2 = tmp1;
    G5C(tmp2, basisVectors[n]);
    axpby(basisVectors[n], 0.5, 0.5, tmp1, tmp2);
    axpby(basisVectors[n+nb], 0.5, -0.5, tmp1, tmp2);
    std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
              << "norm2(vec[" << n << "]) = " << norm2(basisVectors[n]) << ". "
              << "norm2(vec[" << n+nb << "]) = " << norm2(basisVectors[n+nb]) << std::endl;
  }
}


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


// functions needed for the initial implementation of the chirality respecting block project
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
inline void standardBlockProject(Lattice<iVector<CComplex, nbasis>>& coarseData,
                                 const Lattice<vobj>&                fineData,
                                 const VLattice&                     Basis)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  int _ndimension = coarse->_ndimension;

  // checks
  assert(nbasis == Basis.size());
  subdivides(coarse, fine);
  for(int i = 0; i < nbasis; i++) {
    conformable(Basis[i], fineData);
  }

  Coordinate block_r(_ndimension);
  Coordinate fine_rdimensions   = fine->_rdimensions;
  Coordinate coarse_rdimensions = coarse->_rdimensions;

  size_t block_v = 1;
  for(int d = 0; d < _ndimension; ++d) {
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    assert(block_r[d] * coarse->_rdimensions[d] == fine->_rdimensions[d]);
    block_v *= block_r[d];
  }
  assert(block_v == fine->oSites() / coarse->oSites());

  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  accelerator_for(sci, nbasis * coarse->oSites(), vobj::Nsimd(), {
    auto i  = sci % nbasis;
    auto sc = sci / nbasis;

    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, coarse_rdimensions);

    int sf;
    decltype(innerProductD2(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    for(int sb = 0; sb < block_v; ++sb) {
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b, sb, block_r);
      for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      Lexicographic::IndexFromCoor(coor_f, sf, fine_rdimensions);

      reduce = reduce + innerProductD2(Basis_v[i](sf), fineData_v(sf));
    }
    convertType(coarseData_v[sc](i), TensorRemove(reduce));
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class VLattice>
inline void chiralBlockProject(Lattice<iVector<CComplex,nbasis > > &coarseData,
			       const             Lattice<vobj>   &fineData,
			       const VLattice &Basis)
{
  static_assert(nbasis%2 == 0, "Wrong basis size");
  const int nchiralities = 2;
  const int nvectors = nbasis/nchiralities;

  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  int _ndimension = coarse->_ndimension;

  // checks
  assert(nvectors == Basis.size());
  subdivides(coarse, fine);
  for(int i = 0; i < nvectors; i++) {
    conformable(Basis[i], fineData);
  }

  Coordinate block_r(_ndimension);
  Coordinate fine_rdimensions   = fine->_rdimensions;
  Coordinate coarse_rdimensions = coarse->_rdimensions;

  size_t block_v = 1;
  for(int d = 0; d < _ndimension; ++d) {
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    assert(block_r[d] * coarse->_rdimensions[d] == fine->_rdimensions[d]);
    block_v *= block_r[d];
  }
  assert(block_v == fine->oSites() / coarse->oSites());

  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  long coarse_osites = coarse->oSites();

  accelerator_for(_idx, nchiralities * nvectors * coarse_osites, vobj::Nsimd(), {
    auto idx       = _idx;
    auto chirality = idx % nchiralities; idx  /= nchiralities;
    auto basis_i   = idx % nvectors;     idx  /= nvectors;
    auto sc        = idx % coarse_osites; idx /= coarse_osites;

    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, coarse_rdimensions);

    int sf;
    decltype(innerProductLowerPartD2(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    auto coarse_i_offset = chirality * nvectors;

    for(int sb = 0; sb < block_v; ++sb) {
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b, sb, block_r);
      for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      Lexicographic::IndexFromCoor(coor_f, sf, fine_rdimensions);

      if (chirality == 0)
        reduce = reduce + innerProductUpperPartD2(Basis_v[basis_i](sf), fineData_v(sf));
      else if (chirality == 1)
        reduce = reduce + innerProductLowerPartD2(Basis_v[basis_i](sf), fineData_v(sf));
      else
        assert(0);
    }
    convertType(coarseData_v[sc](coarse_i_offset + basis_i), TensorRemove(reduce));
  });
  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<typename vCoeff_t>
void runBenchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // compile-time constants
  const int nbasis = NBASIS; static_assert((nbasis & 0x1) == 0, "");
  const int nsingle = nbasis/2;

  // setup grids
  GridCartesian* UGrid_f =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--fgrid", {8, 8, 8, 8}),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());
  GridCartesian* UGrid_c =
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

  // setup fields
  FineVector src(UGrid_f); random(pRNG, src);
  FineVector ref(UGrid_f); ref = Zero();
  CoarseVector res_single(UGrid_c); res_single = Zero();
  CoarseVector res_normal(UGrid_c); res_normal = Zero();
  CoarseVector diff(UGrid_c); diff = Zero();
  std::vector<FineVector>   basis_single(nsingle, UGrid_f);
  std::vector<FineVector>   basis_normal(nbasis, UGrid_f);

  // randomize
  for(auto& b : basis_single) gaussian(pRNG, b);
  gaussian(pRNG, src);

  // randomize
  for(int n=0; n<basis_single.size(); n++) {
    basis_normal[n] = basis_single[n];
  }
  performChiralDoubling(basis_normal);

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid_f->_fdimensions[mu];

  // warmup + measure standard
  grid_printf("standard warmup %s\n", precision.c_str()); fflush(stdout);
  for(auto n : {1, 2, 3, 4, 5}) standardBlockProject(res_normal, src, basis_normal);
  grid_printf("standard measurement %s\n", precision.c_str()); fflush(stdout);
  double t0 = usecond();
  for(int n = 0; n < nIter; n++) standardBlockProject(res_normal, src, basis_normal);
  double t1 = usecond();
  double secs_standard = (t1-t0)/1e6;

  // warmup + measure chiral
  grid_printf("chiral warmup %s\n", precision.c_str()); fflush(stdout);
  for(auto n : {1, 2, 3, 4, 5}) chiralBlockProject(res_single, src, basis_single);
  grid_printf("chiral measurement %s\n", precision.c_str()); fflush(stdout);
  double t2 = usecond();
  for(int n = 0; n < nIter; n++) chiralBlockProject(res_single, src, basis_single);
  double t3 = usecond();
  double secs_chiral = (t3-t2)/1e6;

  // ensure correctness
  assert(resultsAgree(res_normal, res_single, "chiral"));

  // performance figures
  double flops_per_cmul = 6;
  double flops_per_cadd = 2;
  double fine_complex   = Ns * Nc;
  double fine_floats    = fine_complex * 2;
  double coarse_complex = nbasis;
  double coarse_floats  = coarse_complex * 2;
  double flops_per_site = 1.0 * (fine_complex * flops_per_cmul + (fine_complex - 1) * flops_per_cadd) * nbasis;
  double flops          = flops_per_site * UGrid_f->gSites() * nIter;
  double prec_bytes     = getPrecision<vCoeff_t>::value * 4;
  double nbytes         = (((nsingle + 1) * fine_floats) * UGrid_f->gSites()
                        + coarse_floats * UGrid_c->gSites())
                        * prec_bytes * nIter;

  // report standard
  double dt_standard           = (t1 - t0) / 1e6;
  double GFlopsPerSec_standard = flops / dt_standard / 1e9;
  double GBPerSec_standard     = nbytes / dt_standard / 1e9;
  std::cout << GridLogMessage << nIter << " applications of standardblockProject" << std::endl;
  std::cout << GridLogMessage << "    Time to complete            : " << dt_standard << " s" << std::endl;
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_standard << " GFlops/s" << std::endl;
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_standard << " GB/s" << std::endl << std::endl;

  // report chiral
  double dt_chiral           = (t3 - t2) / 1e6;
  double GFlopsPerSec_chiral = flops / dt_chiral / 1e9;
  double GBPerSec_chiral     = nbytes / dt_chiral / 1e9;
  std::cout << GridLogMessage << nIter << " applications of chiralblockProject" << std::endl;
  std::cout << GridLogMessage << "    Time to complete            : " << dt_chiral << " s" << std::endl;
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_chiral << " GFlops/s" << std::endl;
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_chiral << " GB/s" << std::endl << std::endl;

  grid_printf("finalize %s\n", precision.c_str()); fflush(stdout);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
