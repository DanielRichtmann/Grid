/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_intergrid_operators_speedup_virtual_fields.cc

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


#ifndef NBASIS // use the compiled basis sizes as in gpt here
#define NBASIS 10
#endif


// #define IP_D2 // NOTE: this halves performance on CPUs
// #define IP_D
#define IP_NORMAL

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


template<int lohi,typename vobj,typename std::enable_if<isGridFundamental<vobj>::value && (lohi == 0 || lohi == 1),void>::type* = nullptr>
accelerator_inline auto loadChirality(const iSpinColourVector<vobj>& in) -> iHalfSpinColourVector<vobj> {
  iHalfSpinColourVector<vobj> ret;

  constexpr int s_offset = lohi * 2;

  for(int s=0; s<Nhs; s++)
    ret()(s) = in()(s_offset + s);

  return ret;
}
template<int lohi,typename vobj,int nbasis,typename std::enable_if<isGridFundamental<vobj>::value && (lohi == 0 || lohi == 1) && nbasis%2==0,void>::type* = nullptr>
accelerator_inline auto loadChirality(const iVector<iSinglet<vobj>,nbasis>& in) -> iVector<iSinglet<vobj>,nbasis/2> {
  iVector<iSinglet<vobj>,nbasis/2> ret;

  constexpr int nsingle = nbasis/2;
  constexpr int n_offset = lohi*nsingle;

  for(int n=0; n<nsingle; n++)
    ret(n) = in(n + n_offset);

  return ret;
}

template<int lohi,typename vobj,typename std::enable_if<isGridFundamental<vobj>::value && (lohi == 0 || lohi == 1),void>::type* = nullptr>
accelerator_inline void writeChirality(iSpinColourVector<vobj>& out, const iHalfSpinColourVector<vobj>& in) {
  constexpr int s_offset = lohi * 2;

  for(int s=0; s<Nhs; s++)
    out._internal(s_offset + s) = in()(s);
}
template<int lohi,typename vobj,int nbasis,typename std::enable_if<isGridFundamental<vobj>::value && (lohi == 0 || lohi == 1) && nbasis%2==0,void>::type* = nullptr>
accelerator_inline void writeChirality(iVector<iSinglet<vobj>,nbasis>& out, const iVector<iSinglet<vobj>,nbasis/2>& in) {
  constexpr int nsingle = nbasis/2;
  constexpr int n_offset = lohi*nsingle;

  for(int n=0; n<nsingle; n++)
    out(n_offset + n) = in(n);
}


template<typename vobj,typename std::enable_if<isGridFundamental<vobj>::value,void>::type* = nullptr>
accelerator_inline iScalar<typename GridTypeMapper<vobj>::scalar_object> // TypeMapper because no scalar_object in fundamentals
fineGrainedCoalescedRead(const iSpinColourVector<vobj>& in, int j) {
  assert(0 <= j < Nc * Ns);
  int c = j % Nc;
  int s = j / Nc;

  iScalar<typename GridTypeMapper<vobj>::scalar_object> ret(coalescedRead(in()(s)(c)));
  // ret._internal = coalescedRead(in()(s)(c));
  return ret;
}

template<typename vobj,typename std::enable_if<isGridFundamental<vobj>::value,void>::type* = nullptr>
accelerator_inline void fineGrainedCoalescedWrite(iSpinColourVector<vobj>& out, const iScalar<typename GridTypeMapper<vobj>::scalar_object> & in, int j) {
  assert(0 <= j < Nc * Ns);
  int c = j % Nc;
  int s = j / Nc;

  coalescedWrite(out()(s)(c), TensorRemove(in));
}


// copied here from gpt to cut dependencies
template<typename T>
class PVector {
 protected:
  std::vector<T*> _v;

 public:
  PVector(long size) : _v(size) {
  }

  PVector() : _v() {
  }

  void resize(long size) {
    _v.resize(size);
  }

  void push_back(T* t) {
    _v.push_back(t);
  }

  long size() const {
    return _v.size();
  }

  T& operator[](long i) {
    return *_v[i];
  }

  const T& operator[](long i) const {
    return *_v[i];
  }

  T*& operator()(long i) {
    return _v[i];
  }

  const T*& operator()(long i) const {
    return _v[i];
  }

  PVector slice(long i0, long i1, long step = 1) const {
    if (i0<0)
      i0=0;
    if (i1 > _v.size())
      i1 = _v.size();
    PVector ret;
    for (long i=i0;i<i1;i+=step)
      ret.push_back(_v[i]);
    return ret;
  }
};


// copied here from gpt to cut dependencies
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
void performChiralDoubling(std::vector<Field>& basisVectors, long nvirtual) {
  assert(basisVectors.size()%nvirtual == 0);
  long basis_n = basisVectors.size() / nvirtual;

  assert(basis_n%2 == 0);
  long basis_half_n = basis_n / 2;

  for(int n=0; n<basis_half_n; n++) {
    for(int v=0; v<nvirtual; v++) {
      auto nvL = n*nvirtual+v;
      auto nvR = (basis_half_n+n)*nvirtual+v;

      auto tmp1 = basisVectors[nvL];
      auto tmp2 = tmp1;

      G5C(tmp2, basisVectors[nvL]);
      axpby(basisVectors[nvL], 0.5, 0.5, tmp1, tmp2);
      axpby(basisVectors[nvR], 0.5, -0.5, tmp1, tmp2);
      std::cout << GridLogMessage << "Chirally doubled virtual component " << v << " of vector " << n << ". "
                << "norm2(vec[" << n << ", " << v << "]) = " << norm2(basisVectors[nvL]) << ". "
                << "norm2(vec[" << n+basis_half_n << ", " << v << "]) = " << norm2(basisVectors[nvR]) << std::endl;
    }
  }
}


template<typename Field>
void fillPVector(PVector<Field>& p, std::vector<Field>& f) {
  for(int i=0; i<f.size(); i++)
    p.push_back(&f[i]);
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
#define VECTOR_VIEW_OPEN_POINTER(l,v,p,mode)				\
  Vector< decltype(l[0].View(mode)) > v; v.reserve(l.size());	\
  for(int k=0;k<l.size();k++)				\
    v.push_back(l[k].View(mode)); \
  typename std::remove_reference<decltype(v[0])>::type* p = &v[0];
#define VECTOR_VIEW_CLOSE_POINTER(v,p)                   \
  for(int k=0;k<v.size();k++) v[k].ViewClose(); \
  p = nullptr;


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
void copyFields(PVector<Field>& out, const PVector<Field>& in, long n_virtual) {
  assert(out.size() == in.size());
  assert(out.size() % n_virtual == 0);
  for(int v=0; v<out.size(); v++) {
    out[v] = in[v];
  }
}


template<typename Field>
bool resultsAgree(const PVector<Field>& ref, const PVector<Field>& res, long n_virtual, const std::string& name) {
  assert(ref.size() == res.size());
  assert(ref.size() % n_virtual == 0);
  long  n_vec = ref.size() / n_virtual;
  RealD checkTolerance = (getPrecision<Field>::value == 1) ? 1e-7 : 1e-15;
  Field diff(ref[0].Grid());
  bool ret = true;

  for(int vec_i=0; vec_i<n_vec; vec_i++) {
    for(int virtual_i=0; virtual_i<n_virtual; virtual_i++) {
      auto v = vec_i * n_virtual + virtual_i;

      diff = ref[v] - res[v];
      auto absDev = norm2(diff);
      auto relDev = absDev / norm2(ref[v]);
      std::cout << GridLogMessage
                << "vector index = " << vec_i << " virtual index = " << virtual_i
                << " : norm2(reference), norm2(" << name << "), abs. deviation, rel. deviation: " << norm2(ref[v]) << " "
                << norm2(res[v]) << " " << absDev << " " << relDev << " -> check "
                << ((relDev < checkTolerance) ? "passed" : "failed") << std::endl;

      ret = ret && relDev <= checkTolerance;
    }
  }
  return ret;
}


template<typename Field>
void printNorms(const PVector<Field>& field, long n_virtual, const std::string& name, const std::string& comment) {
  assert(field.size() % n_virtual == 0);
  long  n_vec = field.size() / n_virtual;
  for(int vec_i=0; vec_i<n_vec; vec_i++) {
    for(int virtual_i=0; virtual_i<n_virtual; virtual_i++) {
      auto v = vec_i * n_virtual + virtual_i;
      std::cout << GridLogMessage
                << comment
                << ": vector index = " << vec_i << " virtual index = " << virtual_i
                << " : norm2(" << name << "): " << norm2(field[v]) << std::endl;
    }
  }
}


template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockProject_gptdefault(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				                long                                                       coarse_n_virtual,
				                const PVector<Lattice<vobj>>&                              fine,
				                long                                                       fine_n_virtual,
				                const VLattice&                                            basis,
				                long                                                       basis_n_virtual,
				                const CoarseningLookupTable<T_singlet>&                    lut,
				                long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorWriteDiscard);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,AcceleratorRead);

    accelerator_for(_idx, basis_block*coarse_osites*vec_n, vobj::Nsimd(), {
	auto idx = _idx;
	auto basis_i_rel = idx % basis_block; idx /= basis_block;
	auto basis_i_abs = basis_i_rel + basis_i0;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sc = idx % coarse_osites; idx /= coarse_osites;

	decltype(INNER_PRODUCT(basis_v[0](0), fine_v[0](0))) reduce = Zero();

	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  for(long j=0; j<sizes_v[sc]; ++j) {
	    long sf = lut_v[sc][j];
	    reduce = reduce + INNER_PRODUCT(basis_v[basis_i_rel*fine_n_virtual + fine_virtual_i](sf), fine_v[vec_i*fine_n_virtual + fine_virtual_i](sf));
	  }
	}

	long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	long coarse_i = basis_i_abs % coarse_virtual_size;
	convertType(coarse_v[vec_i*coarse_n_virtual + coarse_virtual_i][sc](coarse_i), TensorRemove(reduce));
      });

    VECTOR_VIEW_CLOSE(basis_v);
  }

  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}


template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockProject_parchange_lut(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				                   long                                                       coarse_n_virtual,
				                   const PVector<Lattice<vobj>>&                              fine,
				                   long                                                       fine_n_virtual,
				                   const VLattice&                                            basis,
				                   long                                                       basis_n_virtual,
				                   const CoarseningLookupTable<T_singlet>&                    lut,
				                   long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  assert(coarse_virtual_size == basis_virtual_size);

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  assert(basis_n == basis_virtual_size*coarse_n_virtual);

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();

  VECTOR_VIEW_OPEN_POINTER(fine,fine_v,fine_p,AcceleratorRead);
  VECTOR_VIEW_OPEN_POINTER(coarse,coarse_v,coarse_p,AcceleratorWriteDiscard);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN_POINTER(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,basis_p,AcceleratorRead);

    accelerator_for(_idx, basis_block*coarse_osites*vec_n, vobj::Nsimd(), {
	auto idx = _idx;
	auto basis_i_rel = idx % basis_block; idx /= basis_block;
	auto basis_i_abs = basis_i_rel + basis_i0;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sc = idx % coarse_osites; idx /= coarse_osites;

	decltype(INNER_PRODUCT(basis_p[0](0), fine_p[0](0))) reduce = Zero();

	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  for(long j=0; j<sizes_v[sc]; ++j) {
	    long sf = lut_v[sc][j];
	    reduce = reduce + INNER_PRODUCT(basis_p[basis_i_rel*fine_n_virtual + fine_virtual_i](sf), fine_p[vec_i*fine_n_virtual + fine_virtual_i](sf));
	  }
	}

	long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	long coarse_i = basis_i_abs % coarse_virtual_size;
	convertType(coarse_p[vec_i*coarse_n_virtual + coarse_virtual_i][sc](coarse_i), TensorRemove(reduce));
      });

    VECTOR_VIEW_CLOSE_POINTER(basis_v,basis_p);
  }

  VECTOR_VIEW_CLOSE_POINTER(fine_v,fine_p);
  VECTOR_VIEW_CLOSE_POINTER(coarse_v,coarse_p);
}
template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockPromote_gptdefault(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				                long                                                       coarse_n_virtual,
				                const PVector<Lattice<vobj>>&                              fine,
				                long                                                       fine_n_virtual,
				                const VLattice&                                            basis,
				                long                                                       basis_n_virtual,
				                const CoarseningLookupTable<T_singlet>&                    lut,
				                long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long fine_osites = fine_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  auto rlut_v = lut.ReverseView();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorRead);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,AcceleratorRead);

    accelerator_for(_idx, fine_osites*vec_n, vobj::Nsimd(), {

	auto idx = _idx;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sf = idx % fine_osites; idx /= fine_osites;
	auto sc = rlut_v[sf];

#ifdef GRID_SIMT
	typename vobj::tensor_reduced::scalar_object cA;
	typename vobj::scalar_object cAx;
#else
	typename vobj::tensor_reduced cA;
	vobj cAx;
#endif

	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  decltype(cAx) fine_t;
	  if (basis_i0 == 0)
	    fine_t = Zero();
	  else
	    fine_t = fine_v[vec_i*fine_n_virtual + fine_virtual_i](sf);

	  for(long basis_i_rel=0; basis_i_rel<basis_block; basis_i_rel++) {
	    long basis_i_abs = basis_i_rel + basis_i0;
	    long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	    long coarse_i = basis_i_abs % coarse_virtual_size;
	    convertType(cA,TensorRemove(coarse_v[vec_i*coarse_n_virtual + coarse_virtual_i](sc)(coarse_i)));
	    auto prod = cA*basis_v[basis_i_rel*fine_n_virtual + fine_virtual_i](sf);
	    convertType(cAx,prod);
	    fine_t = fine_t + cAx;
	  }

	  coalescedWrite(fine_v[vec_i*fine_n_virtual + fine_virtual_i][sf], fine_t);
	}
      });

    VECTOR_VIEW_CLOSE(basis_v);
  }

  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}


template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockPromote_parchange_lut(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				                   long                                                       coarse_n_virtual,
				                   const PVector<Lattice<vobj>>&                              fine,
				                   long                                                       fine_n_virtual,
				                   const VLattice&                                            basis,
				                   long                                                       basis_n_virtual,
				                   const CoarseningLookupTable<T_singlet>&                    lut,
				                   long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long fine_osites = fine_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  auto rlut_v = lut.ReverseView();

  VECTOR_VIEW_OPEN_POINTER(fine,fine_v,fine_p,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN_POINTER(coarse,coarse_v,coarse_p,AcceleratorRead);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN_POINTER(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,basis_p,AcceleratorRead);

    accelerator_for(_idx, fine_osites*vec_n, vobj::Nsimd(), {

	auto idx = _idx;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sf = idx % fine_osites; idx /= fine_osites;
	auto sc = rlut_v[sf];

#ifdef GRID_SIMT
	typename vobj::tensor_reduced::scalar_object cA;
	typename vobj::scalar_object cAx;
#else
	typename vobj::tensor_reduced cA;
	vobj cAx;
#endif

	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  decltype(cAx) fine_t;
	  if (basis_i0 == 0)
	    fine_t = Zero();
	  else
	    fine_t = fine_p[vec_i*fine_n_virtual + fine_virtual_i](sf);

	  for(long basis_i_rel=0; basis_i_rel<basis_block; basis_i_rel++) {
	    long basis_i_abs = basis_i_rel + basis_i0;
	    long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	    long coarse_i = basis_i_abs % coarse_virtual_size;
	    convertType(cA,TensorRemove(coalescedRead(coarse_p[vec_i*coarse_n_virtual + coarse_virtual_i][sc](coarse_i))));
	    auto prod = cA*basis_p[basis_i_rel*fine_n_virtual + fine_virtual_i](sf);
	    convertType(cAx,prod);
	    fine_t = fine_t + cAx;
	  }

	  coalescedWrite(fine_p[vec_i*fine_n_virtual + fine_virtual_i][sf], fine_t);
	}
      });

    VECTOR_VIEW_CLOSE_POINTER(basis_v,basis_p);
  }

  VECTOR_VIEW_CLOSE_POINTER(fine_v,fine_p);
  VECTOR_VIEW_CLOSE_POINTER(coarse_v,coarse_p);
}


template<typename vCoeff_t>
void runBenchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // compile-time constants
  const int nbasis_virtual = NBASIS; static_assert((nbasis_virtual & 0x1) == 0, "");
  const int nsingle_virtual = nbasis_virtual/2;

  // command line arguments
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  const int nvirtual_fine = readFromCommandLineInt(argc, argv, "--nvirtualfine", 1);
  const int nvirtual_coarse = readFromCommandLineInt(argc, argv, "--nvirtualcoarse", 4);
  const int nvec = readFromCommandLineInt(argc, argv, "--nvec", 1);

  // dependent sizes
  const int nvirtual_basis = nvirtual_fine;
  const int nbasis = nbasis_virtual * nvirtual_coarse;
  const int nsingle = nbasis/2; // safe since we have even virtual fields

  // command line arguments
  const int basis_n_block = readFromCommandLineInt(argc, argv, "--basis_n_block", nbasis);

  // helpers
#define xstr(s) str(s)
#define str(s) #s

  // print info about run
  std::cout << GridLogMessage << "Compiled with nbasis_virtual = " << nbasis_virtual << std::endl;
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

  // print info about run
  grid_printf("\n");
  grid_printf("Intergrid operator Benchmark with\n");
  grid_printf("fine fdimensions    : [%d %d %d %d]\n", UGrid_f->_fdimensions[0], UGrid_f->_fdimensions[1], UGrid_f->_fdimensions[2], UGrid_f->_fdimensions[3]);
  grid_printf("coarse fdimensions  : [%d %d %d %d]\n", UGrid_c->_fdimensions[0], UGrid_c->_fdimensions[1], UGrid_c->_fdimensions[2], UGrid_c->_fdimensions[3]);
  grid_printf("precision           : %s\n", precision.c_str());
  grid_printf("nbasis_virtual      : %d\n", nbasis_virtual);
  grid_printf("nvirtual_fine       : %d\n", nvirtual_fine);
  grid_printf("nvirtual_basis      : %d\n", nvirtual_basis);
  grid_printf("nvirtual_coarse     : %d\n", nvirtual_coarse);
  grid_printf("nbasis              : %d\n", nbasis);
  grid_printf("nsingle             : %d\n", nsingle);
  grid_printf("basis_n_block       : %d\n", basis_n_block);
  grid_printf("nvec                : %d\n", nvec);
  grid_printf("\n");

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG_f(UGrid_f);
  GridParallelRNG  pRNG_c(UGrid_c);
  pRNG_f.SeedFixedIntegers(seeds);
  pRNG_c.SeedFixedIntegers(seeds);

  // type definitions -- virtual = compiled fields
  typedef Lattice<iSpinColourVector<vCoeff_t>>                                  VirtualFineVector;
  typedef Lattice<typename VirtualFineVector::vector_object::tensor_reduced>    VirtualFineComplex;
  typedef Lattice<iVector<iSinglet<vCoeff_t>, nbasis_virtual>>                  VirtualCoarseVector;
  // typedef Lattice<iVector<typename FineVector::vector_object, nbasis_virtual>>  VirtualProjectorNormal;
  // typedef Lattice<iVector<typename FineVector::vector_object, nsingle_virtual>> VirtualProjectorSingle;

  // type definitions -- physical fields
  typedef std::vector<VirtualFineVector>      PhysicalFineVector;
  typedef std::vector<VirtualFineComplex>     PhysicalFineComplex;
  typedef std::vector<VirtualCoarseVector>    PhysicalCoarseVector;
  // typedef std::vector<VirtualProjectorNormal> PhysicalProjectorNormal;
  // typedef std::vector<VirtualProjectorSingle> PhysicalProjectorSingle;

  // type definitions -- pointers to physical fields (used by gpt)
  typedef PVector<VirtualFineVector>      FineVector;
  typedef PVector<VirtualFineComplex>     FineComplex;
  typedef PVector<VirtualCoarseVector>    CoarseVector;
  // typedef PVector<VirtualProjectorNormal> ProjectorNormal;
  // typedef PVector<VirtualProjectorSingle> ProjectorSingle;

  // TODO: generalize this
  assert(nvirtual_fine == 1);

  // setup field storage -- fine
  PhysicalFineVector src_f_storage(nvec * nvirtual_fine, UGrid_f);            for(auto& elem: src_f_storage) random(pRNG_f, elem);
  PhysicalFineVector res_f_gptdefault_storage(nvec * nvirtual_fine, UGrid_f); for(auto& elem: res_f_gptdefault_storage) elem = Zero();
  PhysicalFineVector res_f_versions_storage(nvec * nvirtual_fine, UGrid_f);   for(auto& elem: res_f_versions_storage) elem = Zero();

  // point gpt fields to field storage -- fine
  FineVector src_f;            fillPVector(src_f, src_f_storage);
  FineVector res_f_gptdefault; fillPVector(res_f_gptdefault, res_f_gptdefault_storage);
  FineVector res_f_versions;   fillPVector(res_f_versions, res_f_versions_storage);

  // setup field storage -- basis (have 2 indices, i.e., nvirtual_fine and nsingle, flat in one dimension)
  PhysicalFineVector basis_single_storage(nsingle * nvirtual_fine, UGrid_f); for(auto& elem: basis_single_storage) random(pRNG_f, elem);
  PhysicalFineVector basis_normal_storage(nbasis * nvirtual_fine, UGrid_f);  for(auto& elem: basis_normal_storage) random(pRNG_f, elem);
  // PhysicalProjectorNormal basis_normal_fused(?, UGrid_f); // TODO
  // PhysicalProjectorSingle basis_single_fused(?, UGrid_f);

  // point gpt fields to field storage -- basis
  FineVector basis_single; fillPVector(basis_single, basis_single_storage);
  FineVector basis_normal; fillPVector(basis_normal, basis_normal_storage);

  // setup fields -- coarse
  PhysicalCoarseVector src_c_storage(nvec * nvirtual_coarse, UGrid_c);            for(auto& elem: src_c_storage) random(pRNG_c, elem);
  PhysicalCoarseVector res_c_gptdefault_storage(nvec * nvirtual_coarse, UGrid_c); for(auto& elem: res_c_gptdefault_storage) elem = Zero();
  PhysicalCoarseVector res_c_versions_storage(nvec * nvirtual_coarse, UGrid_c);   for(auto& elem: res_c_versions_storage) elem = Zero();

  // point gpt fields to field storage -- coarse
  CoarseVector src_c;            fillPVector(src_c, src_c_storage);
  CoarseVector res_c_gptdefault; fillPVector(res_c_gptdefault, res_c_gptdefault_storage);
  CoarseVector res_c_versions;   fillPVector(res_c_versions, res_c_versions_storage);

  // lookup table
  VirtualFineComplex mask_full(UGrid_f); mask_full = 1.;
  CoarseningLookupTable<VirtualFineComplex> lut(UGrid_c, mask_full);

  // randomize
  for(auto& elem : basis_single_storage) gaussian(pRNG_f, elem);
  for(auto& elem : src_f_storage) gaussian(pRNG_f, elem);

  // fill basis
  for(int n=0; n<basis_single.size(); n++) {
    for(int v=0; v<nvirtual_fine; v++) {
      basis_normal[n*nvirtual_fine+v] = basis_single[n*nvirtual_fine+v];
    }
  }
  performChiralDoubling(basis_normal_storage, nvirtual_fine);
  // fillProjector(basis_single, basis_single_fused); // TODO
  // fillProjector(basis_normal, basis_normal_fused);

  // misc stuff needed for performance figures
  double flops_per_cmul = 6;
  double flops_per_cadd = 2;
  double complex_words  = 2;
  double fine_complex   = Ns * Nc;
  double fine_floats    = fine_complex * complex_words;
  double coarse_complex = nbasis;
  double coarse_floats  = coarse_complex * complex_words;
  double prec_bytes     = getPrecision<vCoeff_t>::value * 4; // 4 for float, 8 for double
  double fine_volume    = std::accumulate(UGrid_f->_fdimensions.begin(),UGrid_f->_fdimensions.end(),1,std::multiplies<int>());
  double coarse_volume  = std::accumulate(UGrid_c->_fdimensions.begin(),UGrid_c->_fdimensions.end(),1,std::multiplies<int>());
  double block_volume   = fine_volume / coarse_volume;

  // project -- my counting with ntv (= minimal required)
  double project_flops_per_fine_site_minimal = (1.0 * nsingle * 2 * (4 * fine_complex - 2)) * nvec;
  double project_words_per_fine_site_minimal = (1.0 * (nsingle + 1) * fine_complex + 1/block_volume * coarse_complex) * nvec;
  double project_bytes_per_fine_site_minimal = project_words_per_fine_site_minimal * complex_words * prec_bytes;
  double project_flops_minimal               = project_flops_per_fine_site_minimal * UGrid_f->gSites() * nIter;
  double project_words_minimal               = project_words_per_fine_site_minimal * UGrid_f->gSites() * nIter;
  double project_nbytes_minimal              = project_bytes_per_fine_site_minimal * UGrid_f->gSites() * nIter;

  // project -- my counting with nbasis
  double project_flops_per_fine_site_nbasis = (1.0 * nbasis * (8 * fine_complex - 2)) * nvec;
  double project_words_per_fine_site_nbasis = (1.0 * (nbasis + 1) * fine_complex + 1/block_volume * coarse_complex) * nvec;
  double project_bytes_per_fine_site_nbasis = project_words_per_fine_site_nbasis * complex_words * prec_bytes;
  double project_flops_nbasis               = project_flops_per_fine_site_nbasis * UGrid_f->gSites() * nIter;
  double project_words_nbasis               = project_words_per_fine_site_nbasis * UGrid_f->gSites() * nIter;
  double project_nbytes_nbasis              = project_bytes_per_fine_site_nbasis * UGrid_f->gSites() * nIter;

  // project -- christoph's counting in the gpt benchmark
  double project_flops_per_fine_site_gpt = (1.0 * (fine_complex * flops_per_cmul + (fine_complex - 1) * flops_per_cadd) * nbasis) * nvec;
  double project_flops_gpt               = project_flops_per_fine_site_gpt * UGrid_f->gSites() * nIter;
  double project_nbytes_gpt              = (((nbasis + 1) * fine_floats) * UGrid_f->gSites()
                                         + coarse_floats * UGrid_c->gSites())
                                         * prec_bytes * nIter * nvec;
  double project_words_gpt               = project_nbytes_gpt / complex_words / prec_bytes;

  // report calculated performance figures per site
  double factor = UGrid_f->gSites() * nIter;
#define PRINT_PER_SITE_VALUES(COUNTING) {\
  grid_printf("Project: per-site values with counting type %10s: flops = %f, words = %f, bytes = %f, flops/bytes = %f\n",\
              #COUNTING, project_flops_##COUNTING/factor, project_words_##COUNTING/factor, project_nbytes_##COUNTING/factor, project_flops_##COUNTING/project_nbytes_##COUNTING);\
  }
  PRINT_PER_SITE_VALUES(minimal);
  PRINT_PER_SITE_VALUES(nbasis);
  PRINT_PER_SITE_VALUES(gpt);
#undef PRINT_PER_SITE_VALUES
  grid_printf("\n");


#define BENCH_PROJECT_VERSION(VERSION, BASIS, ...)\
  double secs_project_##VERSION;\
  {\
    for(int v=0; v<res_c_versions.size(); v++) res_c_versions[v] = Zero();\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) vectorizableBlockProject_##VERSION(res_c_versions, nvirtual_coarse, src_f, nvirtual_fine, BASIS, nvirtual_basis, lut, basis_n_block);\
    for(int v=0; v<res_c_versions.size(); v++) res_c_versions[v] = Zero();\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) vectorizableBlockProject_##VERSION(res_c_versions, nvirtual_coarse, src_f, nvirtual_fine, BASIS, nvirtual_basis, lut, basis_n_block);\
    double t1 = usecond();\
    secs_project_##VERSION = (t1-t0)/1e6;\
    if(strcmp(#VERSION, "gptdefault"))\
      assert(resultsAgree(res_c_gptdefault, res_c_versions, nvirtual_coarse, #VERSION));\
    else\
      copyFields(res_c_gptdefault, res_c_versions, nvirtual_coarse);\
  }

#define PRINT_PROJECT_VERSION(VERSION) {\
  double GFlopsPerSec_project_minimal_##VERSION = project_flops_minimal  / secs_project_##VERSION / 1e9;\
  double GBPerSec_project_minimal_##VERSION     = project_nbytes_minimal / secs_project_##VERSION / 1e9;\
  double GFlopsPerSec_project_nbasis_##VERSION  = project_flops_nbasis   / secs_project_##VERSION / 1e9;\
  double GBPerSec_project_nbasis_##VERSION      = project_nbytes_nbasis  / secs_project_##VERSION / 1e9;\
  double GFlopsPerSec_project_gpt_##VERSION     = project_flops_gpt      / secs_project_##VERSION / 1e9;\
  double GBPerSec_project_gpt_##VERSION         = project_nbytes_gpt     / secs_project_##VERSION / 1e9;\
  std::cout << GridLogMessage << nIter << " applications of blockProject_" << #VERSION << std::endl;\
  std::cout << GridLogMessage << "    Time to complete            : " << secs_project_##VERSION << " s" << std::endl;\
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_project_minimal_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_project_minimal_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    nbasis total performance    : " << GFlopsPerSec_project_nbasis_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    nbasis memory bandwidth     : " << GBPerSec_project_nbasis_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    gpt total performance       : " << GFlopsPerSec_project_gpt_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    gpt memory bandwidth        : " << GBPerSec_project_gpt_##VERSION << " GB/s" << std::endl << std::endl;\
}

  BENCH_PROJECT_VERSION(gptdefault, basis_normal);                                        PRINT_PROJECT_VERSION(gptdefault);
  // BENCH_PROJECT_VERSION(parchange, basis_normal);                                         PRINT_PROJECT_VERSION(parchange);
  BENCH_PROJECT_VERSION(parchange_lut, basis_normal, lut);                                PRINT_PROJECT_VERSION(parchange_lut);
  // BENCH_PROJECT_VERSION(parchange_chiral, basis_single);                                  PRINT_PROJECT_VERSION(parchange_chiral);
  // BENCH_PROJECT_VERSION(parchange_fused, basis_normal_fused);                             PRINT_PROJECT_VERSION(parchange_fused);
  // BENCH_PROJECT_VERSION(parchange_lut_chiral, basis_single, lut);                         PRINT_PROJECT_VERSION(parchange_lut_chiral);
  // BENCH_PROJECT_VERSION(parchange_lut_fused, basis_normal_fused, lut);                    PRINT_PROJECT_VERSION(parchange_lut_fused);
  // BENCH_PROJECT_VERSION(parchange_chiral_fused, basis_single_fused);                      PRINT_PROJECT_VERSION(parchange_chiral_fused);
  // BENCH_PROJECT_VERSION(parchange_lut_chiral_fused, basis_single_fused, lut);             PRINT_PROJECT_VERSION(parchange_lut_chiral_fused);
  // BENCH_PROJECT_VERSION(parchange_finegrained, basis_normal);                             PRINT_PROJECT_VERSION(parchange_finegrained);
  // BENCH_PROJECT_VERSION(parchange_finegrained_lut, basis_normal, lut);                    PRINT_PROJECT_VERSION(parchange_finegrained_lut);
  // BENCH_PROJECT_VERSION(parchange_finegrained_chiral, basis_single);                      PRINT_PROJECT_VERSION(parchange_finegrained_chiral);
  // BENCH_PROJECT_VERSION(parchange_finegrained_fused, basis_normal_fused);                 PRINT_PROJECT_VERSION(parchange_finegrained_fused);
  // BENCH_PROJECT_VERSION(parchange_finegrained_lut_chiral, basis_single, lut);             PRINT_PROJECT_VERSION(parchange_finegrained_lut_chiral);
  // BENCH_PROJECT_VERSION(parchange_finegrained_lut_fused, basis_normal_fused, lut);        PRINT_PROJECT_VERSION(parchange_finegrained_lut_fused);
  // BENCH_PROJECT_VERSION(parchange_finegrained_chiral_fused, basis_single_fused);          PRINT_PROJECT_VERSION(parchange_finegrained_chiral_fused);
  // BENCH_PROJECT_VERSION(parchange_finegrained_lut_chiral_fused, basis_single_fused, lut); PRINT_PROJECT_VERSION(parchange_finegrained_lut_chiral_fused);
  // BENCH_PROJECT_VERSION(parchange_finegrained_play, basis_normal, lut);                   PRINT_PROJECT_VERSION(parchange_finegrained_play);

#undef BENCH_PROJECT_VERSION
#undef PRINT_PROJECT_VERSION


  grid_printf("DONE WITH PROJECT BENCHMARKS in %s precision\n\n", precision.c_str()); fflush(stdout);


  // promote -- my counting with ntv (= minimal required)
  double promote_flops_per_fine_site_minimal = (1.0 * nsingle * 6 * fine_complex + (nsingle - 1) * 2 * fine_complex) * nvec;
  double promote_words_per_fine_site_minimal = (1.0 * (nsingle + 1) * fine_complex + 1/block_volume * coarse_complex) * nvec;
  double promote_bytes_per_fine_site_minimal = promote_words_per_fine_site_minimal * complex_words * prec_bytes;
  double promote_flops_minimal               = promote_flops_per_fine_site_minimal * UGrid_f->gSites() * nIter;
  double promote_words_minimal               = promote_words_per_fine_site_minimal * UGrid_f->gSites() * nIter;
  double promote_nbytes_minimal              = promote_bytes_per_fine_site_minimal * UGrid_f->gSites() * nIter;

  // promote -- my counting with nbasis
  double promote_flops_per_fine_site_nbasis = (1.0 * nbasis * 6 * fine_complex + (nbasis - 1) * 2 * fine_complex) * nvec;
  double promote_words_per_fine_site_nbasis = (1.0 * (nbasis + 1) * fine_complex + 1/block_volume * coarse_complex) * nvec;
  double promote_bytes_per_fine_site_nbasis = promote_words_per_fine_site_nbasis * complex_words * prec_bytes;
  double promote_flops_nbasis               = promote_flops_per_fine_site_nbasis * UGrid_f->gSites() * nIter;
  double promote_words_nbasis               = promote_words_per_fine_site_nbasis * UGrid_f->gSites() * nIter;
  double promote_nbytes_nbasis              = promote_bytes_per_fine_site_nbasis * UGrid_f->gSites() * nIter;

  // promote -- christoph's counting in the gpt benchmark
  double promote_flops_per_fine_site_gpt = (1.0 * (fine_complex * nbasis * flops_per_cmul + fine_complex * (nbasis - 1)) * flops_per_cadd) * nvec;
  double promote_flops_gpt               = promote_flops_per_fine_site_gpt * UGrid_f->gSites() * nIter;
  double promote_nbytes_gpt              = (((nbasis + 1) * fine_floats) * UGrid_f->gSites()
                                         + coarse_floats * UGrid_c->gSites())
                                         * prec_bytes * nIter * nvec;
  double promote_words_gpt               = promote_nbytes_gpt / complex_words / prec_bytes;

  // report calculated performance figures per site
#define PRINT_PER_SITE_VALUES(COUNTING) {\
  grid_printf("Promote: per-site values with counting type %10s: flops = %f, words = %f, bytes = %f, flops/bytes = %f\n",\
              #COUNTING, promote_flops_##COUNTING/factor, promote_words_##COUNTING/factor, promote_nbytes_##COUNTING/factor, promote_flops_##COUNTING/promote_nbytes_##COUNTING);\
  }
  PRINT_PER_SITE_VALUES(minimal);
  PRINT_PER_SITE_VALUES(nbasis);
  PRINT_PER_SITE_VALUES(gpt);
#undef PRINT_PER_SITE_VALUES
  grid_printf("\n");

#define BENCH_PROMOTE_VERSION(VERSION, BASIS, ...)\
  double secs_promote_##VERSION;\
  {\
    for(int v=0; v<res_f_versions.size(); v++) res_f_versions[v] = Zero();\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) vectorizableBlockPromote_##VERSION(src_c, nvirtual_coarse, res_f_versions, nvirtual_fine, BASIS, nvirtual_basis, lut, basis_n_block);\
    for(int v=0; v<res_f_versions.size(); v++) res_f_versions[v] = Zero();\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) vectorizableBlockPromote_##VERSION(src_c, nvirtual_coarse, res_f_versions, nvirtual_fine, BASIS, nvirtual_basis, lut, basis_n_block);\
    double t1 = usecond();\
    secs_promote_##VERSION = (t1-t0)/1e6;\
    if(strcmp(#VERSION, "gptdefault")) {\
      assert(resultsAgree(res_f_gptdefault, res_f_versions, nvirtual_fine, #VERSION));\
    } else {\
      copyFields(res_f_gptdefault, res_f_versions, nvirtual_fine);\
    }\
  }

#define PRINT_PROMOTE_VERSION(VERSION) {\
  double GFlopsPerSec_promote_minimal_##VERSION = promote_flops_minimal  / secs_promote_##VERSION / 1e9;\
  double GBPerSec_promote_minimal_##VERSION     = promote_nbytes_minimal / secs_promote_##VERSION / 1e9;\
  double GFlopsPerSec_promote_nbasis_##VERSION  = promote_flops_nbasis   / secs_promote_##VERSION / 1e9;\
  double GBPerSec_promote_nbasis_##VERSION      = promote_nbytes_nbasis  / secs_promote_##VERSION / 1e9;\
  double GFlopsPerSec_promote_gpt_##VERSION     = promote_flops_gpt      / secs_promote_##VERSION / 1e9;\
  double GBPerSec_promote_gpt_##VERSION         = promote_nbytes_gpt     / secs_promote_##VERSION / 1e9;\
  std::cout << GridLogMessage << nIter << " applications of blockPromote_" << #VERSION << std::endl;\
  std::cout << GridLogMessage << "    Time to complete            : " << secs_promote_##VERSION << " s" << std::endl;\
  std::cout << GridLogMessage << "    Total performance           : " << GFlopsPerSec_promote_minimal_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    Effective memory bandwidth  : " << GBPerSec_promote_minimal_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    nbasis total performance    : " << GFlopsPerSec_promote_nbasis_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    nbasis memory bandwidth     : " << GBPerSec_promote_nbasis_##VERSION << " GB/s" << std::endl;\
  std::cout << GridLogMessage << "    gpt total performance       : " << GFlopsPerSec_promote_gpt_##VERSION << " GFlops/s" << std::endl;\
  std::cout << GridLogMessage << "    gpt memory bandwidth        : " << GBPerSec_promote_gpt_##VERSION << " GB/s" << std::endl << std::endl;\
}

  BENCH_PROMOTE_VERSION(gptdefault, basis_normal);                                        PRINT_PROMOTE_VERSION(gptdefault);
  // BENCH_PROMOTE_VERSION(parchange, basis_normal);                                         PRINT_PROMOTE_VERSION(parchange);
  BENCH_PROMOTE_VERSION(parchange_lut, basis_normal, lut);                                PRINT_PROMOTE_VERSION(parchange_lut);
  // BENCH_PROMOTE_VERSION(parchange_chiral, basis_single);                                  PRINT_PROMOTE_VERSION(parchange_chiral);
  // BENCH_PROMOTE_VERSION(parchange_fused, basis_normal_fused);                             PRINT_PROMOTE_VERSION(parchange_fused);
  // BENCH_PROMOTE_VERSION(parchange_lut_chiral, basis_single, lut);                         PRINT_PROMOTE_VERSION(parchange_lut_chiral);
  // BENCH_PROMOTE_VERSION(parchange_lut_fused, basis_normal_fused, lut);                    PRINT_PROMOTE_VERSION(parchange_lut_fused);
  // BENCH_PROMOTE_VERSION(parchange_chiral_fused, basis_single_fused);                      PRINT_PROMOTE_VERSION(parchange_chiral_fused);
  // BENCH_PROMOTE_VERSION(parchange_lut_chiral_fused, basis_single_fused, lut);             PRINT_PROMOTE_VERSION(parchange_lut_chiral_fused);
  // BENCH_PROMOTE_VERSION(parchange_finegrained, basis_normal);                             PRINT_PROMOTE_VERSION(parchange_finegrained);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_lut, basis_normal, lut);                    PRINT_PROMOTE_VERSION(parchange_finegrained_lut);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_chiral, basis_single);                      PRINT_PROMOTE_VERSION(parchange_finegrained_chiral);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_fused, basis_normal_fused);                 PRINT_PROMOTE_VERSION(parchange_finegrained_fused);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_lut_chiral, basis_single, lut);             PRINT_PROMOTE_VERSION(parchange_finegrained_lut_chiral);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_lut_fused, basis_normal_fused, lut);        PRINT_PROMOTE_VERSION(parchange_finegrained_lut_fused);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_chiral_fused, basis_single_fused);          PRINT_PROMOTE_VERSION(parchange_finegrained_chiral_fused);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_lut_chiral_fused, basis_single_fused, lut); PRINT_PROMOTE_VERSION(parchange_finegrained_lut_chiral_fused);
  // BENCH_PROMOTE_VERSION(parchange_finegrained_play, basis_normal, lut);                   PRINT_PROMOTE_VERSION(parchange_finegrained_play);

#undef BENCH_PROMOTE_VERSION
#undef PRINT_PROMOTE_VERSION


  grid_printf("DONE WITH PROMOTE BENCHMARKS in %s precision\n\n", precision.c_str()); fflush(stdout);
  grid_printf("finalize %s\n", precision.c_str()); fflush(stdout);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
