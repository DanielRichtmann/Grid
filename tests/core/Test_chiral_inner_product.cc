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


// #define IP_D2
// #define IP_D
// #define IP_NORMAL

#if defined(IP_D2) // as it is originally in gpt
#define INNER_PRODUCT innerProductD2
#define INNER_PRODUCT_LOWER_PART innerProductLowerPartD2
#define INNER_PRODUCT_UPPER_PART innerProductUpperPartD2
#pragma message("compiling with IP_D2")
#elif defined(IP_D) // other version with 'D' rather than 'D2'
#define INNER_PRODUCT innerProductD
#define INNER_PRODUCT_LOWER_PART innerProductLowerPartD
#define INNER_PRODUCT_UPPER_PART innerProductUpperPartD
#pragma message("compiling with IP_D")
#elif defined(IP_NORMAL) // other version with ' ' rather than 'D2'
#define INNER_PRODUCT innerProduct
#define INNER_PRODUCT_LOWER_PART innerProductLowerPart
#define INNER_PRODUCT_UPPER_PART innerProductUpperPart
#pragma message("compiling with IP_NORMAL")
#else
#error Either one of IP_D2, IP_D, or IP_NORMAL needs to be defined
#endif


template<class ScalarField>
class CoarseningLookupTable {
public:

  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  typedef uint64_t index_type;
  typedef uint64_t size_type;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:
  GridBase*                       coarse_;
  GridBase*                       fine_;
  std::vector<Vector<index_type>> lut_vec_;
  Vector<index_type*>             lut_ptr_;
  Vector<size_type>               sizes_;
  Vector<index_type>              reverse_lut_vec_;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:
  CoarseningLookupTable(GridBase* coarse, ScalarField const& mask)
    : coarse_(coarse)
    , fine_(mask.Grid())
    , lut_vec_(coarse_->oSites())
    , lut_ptr_(coarse_->oSites())
    , sizes_(coarse_->oSites())
    , reverse_lut_vec_(fine_->oSites()){
    populate(coarse_, mask);
  }

  virtual accelerator_inline
  std::vector<Vector<index_type>> const& operator()() const {
    return lut_vec_;
  } // CPU access (TODO: remove?)

  virtual accelerator_inline
  index_type const* const* View() const {
    return &lut_ptr_[0];
  } // GPU access

  virtual accelerator_inline
  size_type const* Sizes() const {
    return &sizes_[0];
  }  // also needed for GPU access

  virtual accelerator_inline
  index_type const* ReverseView() const {
    return &reverse_lut_vec_[0];
  }

  virtual bool gridsMatch(GridBase* coarse, GridBase* fine) const {
    return (coarse == coarse_) && (fine == fine_);
  }

private:

  void populate(GridBase* coarse, GridBase* fine) {
    ScalarField fullmask(fine);
    fullmask = 1.;
    populate(coarse, fullmask);
  }

  void populate(GridBase* coarse, ScalarField const& mask) {
    int        _ndimension = coarse_->_ndimension;
    Coordinate block_r(_ndimension);
    int Nsimd = coarse->Nsimd();

    size_type block_v = 1;
    for(int d = 0; d < _ndimension; ++d) {
      block_r[d] = fine_->_rdimensions[d] / coarse_->_rdimensions[d];
      assert(block_r[d] * coarse_->_rdimensions[d] == fine_->_rdimensions[d]);
      block_v *= block_r[d];
    }
    assert(block_v == fine_->oSites()/coarse_->oSites());

    lut_vec_.resize(coarse_->oSites());
    lut_ptr_.resize(coarse_->oSites());
    sizes_.resize(coarse_->oSites());
    reverse_lut_vec_.resize(fine_->oSites());
    for(index_type sc = 0; sc < coarse_->oSites(); ++sc) {
      lut_vec_[sc].resize(block_v);
      lut_ptr_[sc] = &lut_vec_[sc][0];
      sizes_[sc]  = 0;
    }

    typedef typename ScalarField::scalar_type scalar_t;
    typedef typename ScalarField::vector_type vector_t;
    scalar_t zz = {0., 0.,};

    autoView(mask_v, mask, CpuRead);
    thread_for(sc, coarse_->oSites(), {
      Coordinate coor_c(_ndimension);
      Lexicographic::CoorFromIndex(coor_c, sc, coarse_->_rdimensions);

      int sf_tmp, count = 0;
      for(int sb = 0; sb < block_v; ++sb) {
        Coordinate coor_b(_ndimension);
        Coordinate coor_f(_ndimension);

        Lexicographic::CoorFromIndex(coor_b, sb, block_r);
        for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
        Lexicographic::IndexFromCoor(coor_f, sf_tmp, fine_->_rdimensions);

        index_type sf = (index_type)sf_tmp;

	// masks are understood only on reduced SIMD grid, in order to forbid
	// unexpected behavior, force consistency!
	vector_t vmask = TensorRemove(mask_v[sf]);
	scalar_t* fmask = (scalar_t*)&vmask;
	bool bset = fmask[0] != zz;
	for (int lane=1;lane<Nsimd;lane++)
	  assert(bset == (fmask[lane] != zz));
        if(bset) {
          lut_ptr_[sc][count] = sf;
          sizes_[sc]++;
          count++;
        }
        reverse_lut_vec_[sf] = sc; // reverse table will never have holes
      }
      lut_vec_[sc].resize(sizes_[sc]);
    });
  }
};


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


template<typename vobj, int nvectors>
void fillProjector(const std::vector<Lattice<vobj>>& basisVectors, Lattice<iVector<vobj, nvectors>>& projector) {
  assert(basisVectors.size() == nvectors);
  for(int i=0; i<nvectors; i++) {conformable(basisVectors[i], projector);}

  autoView(projector_v, projector, AcceleratorWrite);

  typedef decltype(basisVectors[0].View(AcceleratorRead)) View;
  Vector<View> basisVectors_v; basisVectors_v.reserve(basisVectors.size());
  for(int i=0;i<basisVectors.size();i++){
    basisVectors_v.push_back(basisVectors[i].View(AcceleratorRead));
  }

  GridBase* grid = projector.Grid();
  long osites = grid->oSites();

  accelerator_for(_idx, nvectors * osites, vobj::Nsimd(), {
    auto idx      = _idx;
    auto vector_i = idx % nvectors; idx /= nvectors;
    auto ss       = idx % osites;   idx /= osites;
    coalescedWrite(projector_v[ss](vector_i), basisVectors_v[vector_i](ss));
  });

  for(int i=0;i<basisVectors.size();i++) basisVectors_v[i].ViewClose();
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
inline void blockProject_griddefault(Lattice<iVector<CComplex, nbasis>>& coarseData,
                                     const Lattice<vobj>&                fineData,
                                     const VLattice&                     Basis) {
  blockProject(coarseData, fineData, Basis);
}


template<class vobj,class CComplex,int nbasis,class VLattice>
inline void blockProject_parchange(Lattice<iVector<CComplex, nbasis>>& coarseData,
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

  long coarse_osites = coarse->oSites();

  accelerator_for(sci, nbasis * coarse_osites, vobj::Nsimd(), {
    auto i  = sci % nbasis;
    auto sc = sci / nbasis;

    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, coarse_rdimensions);

    int sf;
    decltype(INNER_PRODUCT(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    for(int sb = 0; sb < block_v; ++sb) {
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b, sb, block_r);
      for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      Lexicographic::IndexFromCoor(coor_f, sf, fine_rdimensions);

      reduce = reduce + INNER_PRODUCT(Basis_v[i](sf), fineData_v(sf));
    }
    convertType(coarseData_v[sc](i), TensorRemove(reduce));
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class VLattice,class ScalarField>
inline void blockProject_parchange_lut(Lattice<iVector<CComplex, nbasis>>& coarseData,
                                       const Lattice<vobj>&                fineData,
                                       const VLattice&                     Basis,
                                       CoarseningLookupTable<ScalarField>& lut)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  // checks
  assert(fine->_ndimension == coarse->_ndimension);
  for(int i = 0; i < nbasis; i++) {conformable(Basis[i], fineData);}
  assert(nbasis == Basis.size());
  assert(lut.gridsMatch(coarse, fine));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();
  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  long coarse_osites = coarse->oSites();

  accelerator_for(sci, nbasis * coarse_osites, vobj::Nsimd(), {
    auto i  = sci % nbasis;
    auto sc = sci / nbasis;

    decltype(INNER_PRODUCT(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    for(int j=0; j<sizes_v[sc]; ++j) {
      int sf = lut_v[sc][j];
      reduce = reduce + INNER_PRODUCT(Basis_v[i](sf), fineData_v(sf));
    }
    convertType(coarseData_v[sc](i), TensorRemove(reduce));
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class VLattice>
inline void blockProject_parchange_chiral(Lattice<iVector<CComplex,nbasis > >& coarseData,
			                  const Lattice<vobj>&                 fineData,
			                  const VLattice&                      Basis)
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
    auto basis_i   = idx % nvectors;     idx  /= nvectors;
    auto chirality = idx % nchiralities; idx  /= nchiralities;
    auto sc        = idx % coarse_osites; idx /= coarse_osites;

    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, coarse_rdimensions);

    int sf;
    decltype(INNER_PRODUCT_UPPER_PART(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    auto coarse_i_offset = chirality * nvectors;

    for(int sb = 0; sb < block_v; ++sb) {
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b, sb, block_r);
      for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      Lexicographic::IndexFromCoor(coor_f, sf, fine_rdimensions);

      if (chirality == 0)
        reduce = reduce + INNER_PRODUCT_UPPER_PART(Basis_v[basis_i](sf), fineData_v(sf));
      else if (chirality == 1)
        reduce = reduce + INNER_PRODUCT_LOWER_PART(Basis_v[basis_i](sf), fineData_v(sf));
      else
        assert(0);
    }
    convertType(coarseData_v[sc](coarse_i_offset + basis_i), TensorRemove(reduce));
  });
  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class VLattice,class ScalarField>
inline void blockProject_parchange_lut_chiral(Lattice<iVector<CComplex, nbasis>>& coarseData,
                                              const Lattice<vobj>&                fineData,
                                              const VLattice&                     Basis,
                                              CoarseningLookupTable<ScalarField>& lut)
{
  static_assert(nbasis%2 == 0, "Wrong basis size");
  const int nchiralities = 2;
  const int nvectors = nbasis/nchiralities;

  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  // checks
  assert(fine->_ndimension == coarse->_ndimension);
  for(int i = 0; i < nvectors; i++) {conformable(Basis[i], fineData);}
  assert(nvectors == Basis.size());
  assert(lut.gridsMatch(coarse, fine));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();
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
    auto basis_i   = idx % nvectors;     idx  /= nvectors;
    auto chirality = idx % nchiralities; idx  /= nchiralities;
    auto sc        = idx % coarse_osites; idx /= coarse_osites;

    auto coarse_i_offset = chirality * nvectors;

    decltype(INNER_PRODUCT_UPPER_PART(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    for(int j=0; j<sizes_v[sc]; ++j) {
      int sf = lut_v[sc][j];
      if (chirality == 0)
        reduce = reduce + INNER_PRODUCT_UPPER_PART(Basis_v[basis_i](sf), fineData_v(sf));
      else if (chirality == 1)
        reduce = reduce + INNER_PRODUCT_LOWER_PART(Basis_v[basis_i](sf), fineData_v(sf));
      else
        assert(0);
    }
    convertType(coarseData_v[sc](coarse_i_offset + basis_i), TensorRemove(reduce));
  });
  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class ScalarField,typename std::enable_if<nbasis%2==0,void>::type* = nullptr>
inline void blockProject_parchange_lut_chiral_fused(Lattice<iVector<CComplex, nbasis>>&     coarseData,
                                                    const Lattice<vobj>&                    fineData,
                                                    const Lattice<iVector<vobj, nbasis/2>>& projector,
                                                    CoarseningLookupTable<ScalarField>&     lut)
{
  static_assert(nbasis%2 == 0, "Wrong basis size");
  const int nchiralities = 2;
  const int nvectors = nbasis/nchiralities;

  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  // checks
  assert(fine->_ndimension == coarse->_ndimension);
  conformable(projector, fineData);
  assert(lut.gridsMatch(coarse, fine));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();
  autoView(projector_v, projector, AcceleratorRead);
  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  long coarse_osites = coarse->oSites();

  accelerator_for(_idx, nchiralities * nvectors * coarse_osites, vobj::Nsimd(), {
    auto idx       = _idx;
    auto basis_i   = idx % nvectors;     idx  /= nvectors;
    auto chirality = idx % nchiralities; idx  /= nchiralities;
    auto sc        = idx % coarse_osites; idx /= coarse_osites;

    auto coarse_i_offset = chirality * nvectors;

    decltype(INNER_PRODUCT_UPPER_PART(coalescedRead(projector_v[0](0)), fineData_v(0))) reduce = Zero();

    for(int j=0; j<sizes_v[sc]; ++j) {
      int sf = lut_v[sc][j];
      if (chirality == 0)
        reduce = reduce + INNER_PRODUCT_UPPER_PART(coalescedRead(projector_v[sf](basis_i)), fineData_v(sf));
      else if (chirality == 1)
        reduce = reduce + INNER_PRODUCT_LOWER_PART(coalescedRead(projector_v[sf](basis_i)), fineData_v(sf));
      else
        assert(0);
    }
    convertType(coarseData_v[sc](coarse_i_offset + basis_i), TensorRemove(reduce));
  });
}


template<class vobj,class CComplex,int nbasis,class VLattice>
inline void blockPromote_griddefault(const Lattice<iVector<CComplex, nbasis>>& coarseData,
                                           Lattice<vobj>&                      fineData,
                                     const VLattice&                           Basis) {
  blockPromote(coarseData, fineData, Basis);
}


template<class vobj,class CComplex,int nbasis,class VLattice>
inline void blockPromote_parchange(const Lattice<iVector<CComplex, nbasis>>& coarseData,
                                         Lattice<vobj>&                      fineData,
                                   const VLattice&                           Basis) {
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

  autoView(coarseData_v, coarseData, AcceleratorRead);
  autoView(fineData_v, fineData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  long fine_osites = fine->oSites();

  accelerator_for(sf, fine_osites, vobj::Nsimd(), {
    int sc;

    Coordinate coor_c(_ndimension);
    Coordinate coor_f(_ndimension);

    Lexicographic::CoorFromIndex(coor_f, sf, fine_rdimensions);
    for(int d = 0; d < _ndimension; ++d) coor_c[d] = coor_f[d] / block_r[d];
    Lexicographic::IndexFromCoor(coor_c, sc, coarse_rdimensions);

    decltype(coalescedRead(fineData_v[0])) fineData_t;
    for(int i=0; i<nbasis; ++i) {
      if(i == 0)
        fineData_t =              coarseData_v(sc)(i) * Basis_v[i](sf);
      else
        fineData_t = fineData_t + coarseData_v(sc)(i) * Basis_v[i](sf);
    }
    coalescedWrite(fineData_v[sf], fineData_t);
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}


template<class vobj,class CComplex,int nbasis,class VLattice,class ScalarField>
inline void blockPromote_parchange_lut(const Lattice<iVector<CComplex, nbasis>>& coarseData,
                                             Lattice<vobj>&                      fineData,
                                       const VLattice&                           Basis,
                                             CoarseningLookupTable<ScalarField>& lut)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  // checks
  assert(fine->_ndimension == coarse->_ndimension);
  for(int i = 0; i < nbasis; i++) {conformable(Basis[i], fineData);}
  assert(nbasis == Basis.size());
  assert(lut.gridsMatch(coarse, fine));

  auto rlut_v = lut.ReverseView();
  autoView(coarseData_v, coarseData, AcceleratorRead);
  autoView(fineData_v, fineData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  long fine_osites = fine->oSites();

  accelerator_for(sf, fine_osites, vobj::Nsimd(), {
    int sc = rlut_v[sf];

    decltype(coalescedRead(fineData_v[0])) fineData_t;
    for(int i=0; i<nbasis; ++i) {
      if(i == 0)
        fineData_t =              coarseData_v(sc)(i) * Basis_v[i](sf);
      else
        fineData_t = fineData_t + coarseData_v(sc)(i) * Basis_v[i](sf);
    }
    coalescedWrite(fineData_v[sf], fineData_t);
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

  // helpers
#define xstr(s) str(s)
#define str(s) #s

  // print info about run
  std::cout << GridLogMessage << "Compiled with nbasis = " << nbasis << " -> nb = " << nsingle << std::endl;
  std::cout << GridLogMessage << "Compiled with INNER_PRODUCT            = " << xstr(INNER_PRODUCT) << std::endl;
  std::cout << GridLogMessage << "Compiled with INNER_PRODUCT_UPPER_PART = " << xstr(INNER_PRODUCT_UPPER_PART) << std::endl;
  std::cout << GridLogMessage << "Compiled with INNER_PRODUCT_LOWER_PART = " << xstr(INNER_PRODUCT_LOWER_PART) << std::endl;

  // get rid of helpers
#undef xstr
#undef str

  // setup grids
  GridCartesian* UGrid_f =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--fgrid", {8, 8, 8, 8}),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());
  UGrid_f->show_decomposition();
  GridCartesian* UGrid_c =
    SpaceTimeGrid::makeFourDimGrid(readFromCommandlineIvec(argc, argv, "--cgrid", {4, 4, 4, 4}),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());
  UGrid_c->show_decomposition();

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG_f(UGrid_f);
  GridParallelRNG  pRNG_c(UGrid_c);
  pRNG_f.SeedFixedIntegers(seeds);
  pRNG_c.SeedFixedIntegers(seeds);

  // type definitions
  typedef Lattice<iSpinColourVector<vCoeff_t>>                          FineVector;
  typedef Lattice<typename FineVector::vector_object::tensor_reduced>   FineComplex;
  typedef Lattice<iVector<iSinglet<vCoeff_t>, nbasis>>                  CoarseVector;
  typedef Lattice<iVector<typename FineVector::vector_object, nsingle>> Projector;

  // setup fields -- fine
  FineVector src_f(UGrid_f); random(pRNG_f, src_f);
  FineVector res_f_griddefault(UGrid_f); res_f_griddefault = Zero();
  FineVector res_f_parchange(UGrid_f); res_f_parchange = Zero();
  FineVector res_f_parchange_lut(UGrid_f); res_f_parchange_lut = Zero();
  FineVector res_f_parchange_chiral(UGrid_f); res_f_parchange_chiral = Zero();
  FineVector res_f_parchange_lut_chiral(UGrid_f); res_f_parchange_lut_chiral = Zero();
  FineVector res_f_parchange_lut_chiral_fused(UGrid_f); res_f_parchange_lut_chiral_fused = Zero();
  FineVector res_f_parchange_play(UGrid_f); res_f_parchange_play = Zero();
  std::vector<FineVector>   basis_single(nsingle, UGrid_f);
  std::vector<FineVector>   basis_normal(nbasis, UGrid_f);
  Projector                 basis_fused(UGrid_f);

  // setup fields -- coarse
  CoarseVector src_c(UGrid_c); random(pRNG_c, src_c);
  CoarseVector res_c_griddefault(UGrid_c); res_c_griddefault = Zero();
  CoarseVector res_c_parchange(UGrid_c); res_c_parchange = Zero();
  CoarseVector res_c_parchange_lut(UGrid_c); res_c_parchange_lut = Zero();
  CoarseVector res_c_parchange_chiral(UGrid_c); res_c_parchange_chiral = Zero();
  CoarseVector res_c_parchange_lut_chiral(UGrid_c); res_c_parchange_lut_chiral = Zero();
  CoarseVector res_c_parchange_lut_chiral_fused(UGrid_c); res_c_parchange_lut_chiral_fused = Zero();
  CoarseVector res_c_parchange_play(UGrid_c); res_c_parchange_play = Zero();

  // lookup table
  FineComplex mask_full(UGrid_f); mask_full = 1.;
  CoarseningLookupTable<FineComplex> lut(UGrid_c, mask_full);

  // randomize
  for(auto& b : basis_single) gaussian(pRNG_f, b);
  gaussian(pRNG_f, src_f);

  // fill basis
  for(int n=0; n<basis_single.size(); n++) {
    basis_normal[n] = basis_single[n];
  }
  performChiralDoubling(basis_normal);
  fillProjector(basis_single, basis_fused);

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid_f->_fdimensions[mu];

  // misc stuff needed for performance figures
  double flops_per_cmul = 6;
  double flops_per_cadd = 2;
  double fine_complex   = Ns * Nc;
  double fine_floats    = fine_complex * 2;
  double coarse_complex = nbasis;
  double coarse_floats  = coarse_complex * 2;
  double prec_bytes     = getPrecision<vCoeff_t>::value * 4;

  // performance figures project
  double project_flops_per_site = 1.0 * (fine_complex * flops_per_cmul + (fine_complex - 1) * flops_per_cadd) * nsingle;
  double flops_project          = project_flops_per_site * UGrid_f->gSites() * nIter;
  double nbytes_project         = (((nsingle + 1) * fine_floats) * UGrid_f->gSites()
                                + coarse_floats * UGrid_c->gSites())
                                * prec_bytes * nIter;
  double flops_wrong_project    = project_flops_per_site * 2 * UGrid_f->gSites() * nIter;   // 2 times more per site
  double nbytes_wrong_project   = (((nbasis + 1) * fine_floats) * UGrid_f->gSites() // nsingle*fine_floats times more per site
                                + coarse_floats * UGrid_c->gSites())
                                * prec_bytes * nIter;
  double flops_mgbench_project  = (8 * fine_complex) * nbasis * UGrid_f->gSites() * nIter; // 2 more per inner product, factor 2 more vectors
  double nbytes_mgbench_project = (2 * 1 + 2 * fine_complex) * nbasis * UGrid_f->gSites()
                                * 2 * prec_bytes * nIter;

  // report calculated performance figures per site
  double factor = UGrid_f->gSites() * nIter;
  std::cout << GridLogMessage << "Project: Minimal per site: flops, words, bytes, flops/bytes: " << project_flops_per_site << " " << nbytes_project / factor << " " << flops_project/nbytes_project<< std::endl;
  std::cout << GridLogMessage << "Project: Wrong   per site: flops, words, bytes, flops/bytes: " << project_flops_per_site * 2 << " " << nbytes_wrong_project / factor << " " << flops_wrong_project/nbytes_wrong_project << std::endl;
  std::cout << GridLogMessage << "Project: MGBENCH per site: flops, words, bytes, flops/bytes: " << (8 * fine_complex) * nbasis << " " << nbytes_mgbench_project / factor << " " << flops_mgbench_project/nbytes_mgbench_project << std::endl;

#define BENCH_PROJECT_VERSION(VERSION, ...)\
  double secs_project_##VERSION;\
  {\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) blockProject_##VERSION(res_c_##VERSION, src_f, __VA_ARGS__);\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) blockProject_##VERSION(res_c_##VERSION, src_f, __VA_ARGS__);\
    double t1 = usecond();\
    secs_project_##VERSION = (t1-t0)/1e6;\
    assert(resultsAgree(res_c_griddefault, res_c_##VERSION, #VERSION));\
  }

#define PRINT_PROJECT_VERSION(VERSION) {\
  double GFlopsPerSec_project_##VERSION       = flops_project / secs_project_##VERSION / 1e9;\
  double GBPerSec_project_##VERSION           = nbytes_project / secs_project_##VERSION / 1e9;\
  double GFlopsPerSec_project_wrong_##VERSION = flops_wrong_project / secs_project_##VERSION / 1e9;\
  double GBPerSec_project_wrong_##VERSION     = nbytes_wrong_project / secs_project_##VERSION / 1e9;\
  std::cout << GridLogMessage << nIter << " applications of blockProject_" << #VERSION << std::endl;\
  std::cout << GridLogMessage << "    Time to complete            : " << secs_project_##VERSION << " s" << std::endl;\
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_project_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_project_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    Wrong     total performance : " << GFlopsPerSec_project_wrong_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Wrong     memory bandwidth  : " << GBPerSec_project_wrong_##VERSION << " GB/s" << std::endl << std::endl;\
}

  BENCH_PROJECT_VERSION(griddefault, basis_normal);                    PRINT_PROJECT_VERSION(griddefault);
  BENCH_PROJECT_VERSION(parchange, basis_normal);                      PRINT_PROJECT_VERSION(parchange);
  BENCH_PROJECT_VERSION(parchange_lut, basis_normal, lut);             PRINT_PROJECT_VERSION(parchange_lut);
  BENCH_PROJECT_VERSION(parchange_chiral, basis_single);               PRINT_PROJECT_VERSION(parchange_chiral);
  BENCH_PROJECT_VERSION(parchange_lut_chiral, basis_single, lut);      PRINT_PROJECT_VERSION(parchange_lut_chiral);
  // BENCH_PROJECT_VERSION(parchange_lut_chiral_fused, basis_fused, lut); PRINT_PROJECT_VERSION(parchange_lut_chiral_fused);
  BENCH_PROJECT_VERSION(parchange_play, basis_normal, lut);            PRINT_PROJECT_VERSION(parchange_play);

#undef BENCH_PROJECT_VERSION
#undef PRINT_PROJECT_VERSION


  grid_printf("DONE WITH PROJECT BENCHMARKS in %s precision\n\n", precision.c_str()); fflush(stdout);


  // performance figures promote
  double promote_flops_per_site = fine_complex * nsingle * flops_per_cmul + fine_complex * (nsingle - 1) * flops_per_cadd;
  double flops_promote          = promote_flops_per_site * UGrid_f->gSites() * nIter;
  double nbytes_promote         = (((nsingle + 1) * fine_floats) * UGrid_f->gSites()
                                + coarse_floats * UGrid_c->gSites())
                                * prec_bytes * nIter;
  double flops_wrong_promote    = 2 * (promote_flops_per_site + fine_complex) * UGrid_f->gSites() * nIter;
  double nbytes_wrong_promote   = (((nbasis + 1) * fine_floats) * UGrid_f->gSites() // nsingle*fine_floats times more per site
                                + coarse_floats * UGrid_c->gSites())
                                * prec_bytes * nIter;
  double flops_mgbench_promote  = (8 * (nbasis - 1) + 6) * fine_complex * UGrid_f->gSites() * nIter;
  double nbytes_mgbench_promote = ((1 * 1 + 3 * fine_complex) * (nbasis - 1) + (1 * 1 + 2 * fine_complex) * 1) * UGrid_f->gSites()
                                * 2 * prec_bytes * nIter;

  // report calculated performance figures per site
  std::cout << GridLogMessage << "Promote: Minimal per site: flops, words, bytes, flops/bytes: " << promote_flops_per_site << " " << nbytes_promote / factor << " " << flops_promote/nbytes_promote<< std::endl;
  std::cout << GridLogMessage << "Promote: Wrong   per site: flops, words, bytes, flops/bytes: " << 2 * (promote_flops_per_site + fine_complex) << " " << nbytes_wrong_promote / factor << " " << flops_wrong_promote/nbytes_wrong_promote << std::endl;
  std::cout << GridLogMessage << "Promote: MGBENCH per site: flops, words, bytes, flops/bytes: " << (8 * (nbasis - 1) + 6) * fine_complex << " " << nbytes_mgbench_promote / factor << " " << flops_mgbench_promote/nbytes_mgbench_promote << std::endl;

#define BENCH_PROMOTE_VERSION(VERSION, ...)\
  double secs_promote_##VERSION;\
  {\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) blockPromote_##VERSION(src_c, res_f_##VERSION, __VA_ARGS__);\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) blockPromote_##VERSION(src_c, res_f_##VERSION, __VA_ARGS__);\
    double t1 = usecond();\
    secs_promote_##VERSION = (t1-t0)/1e6;\
    assert(resultsAgree(res_f_griddefault, res_f_##VERSION, #VERSION));\
  }

#define PRINT_PROMOTE_VERSION(VERSION) {\
  double GFlopsPerSec_promote_##VERSION       = flops_promote / secs_promote_##VERSION / 1e9;\
  double GBPerSec_promote_##VERSION           = nbytes_promote / secs_promote_##VERSION / 1e9;\
  double GFlopsPerSec_promote_wrong_##VERSION = flops_wrong_promote / secs_promote_##VERSION / 1e9;\
  double GBPerSec_promote_wrong_##VERSION     = nbytes_wrong_promote / secs_promote_##VERSION / 1e9;\
  std::cout << GridLogMessage << nIter << " applications of blockPromote_" << #VERSION << std::endl;\
  std::cout << GridLogMessage << "    Time to complete            : " << secs_promote_##VERSION << " s" << std::endl;\
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_promote_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_promote_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    Wrong     total performance : " << GFlopsPerSec_promote_wrong_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Wrong     memory bandwidth  : " << GBPerSec_promote_wrong_##VERSION << " GB/s" << std::endl << std::endl;\
}

  BENCH_PROMOTE_VERSION(griddefault, basis_normal);                    PRINT_PROMOTE_VERSION(griddefault);
  BENCH_PROMOTE_VERSION(parchange, basis_normal);                      PRINT_PROMOTE_VERSION(parchange);
  BENCH_PROMOTE_VERSION(parchange_lut, basis_normal, lut);             PRINT_PROMOTE_VERSION(parchange_lut);
  // BENCH_PROMOTE_VERSION(parchange_chiral, basis_single);               PRINT_PROMOTE_VERSION(parchange_chiral);
  // BENCH_PROMOTE_VERSION(parchange_lut_chiral, basis_single, lut);      PRINT_PROMOTE_VERSION(parchange_lut_chiral);
  // // BENCH_PROMOTE_VERSION(parchange_lut_chiral_fused, basis_fused, lut); PRINT_PROMOTE_VERSION(parchange_lut_chiral_fused);
  // BENCH_PROMOTE_VERSION(parchange_play, basis_normal, lut);            PRINT_PROMOTE_VERSION(parchange_play);

#undef BENCH_PROMOTE_VERSION
#undef PRINT_PROMOTE_VERSION


  grid_printf("DONE WITH PROMOTE BENCHMARKS in %s precision\n\n", precision.c_str()); fflush(stdout);
  grid_printf("finalize %s\n", precision.c_str()); fflush(stdout);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
