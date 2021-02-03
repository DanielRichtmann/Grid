/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_coarse_operator_virtual_fields_speedup.cc

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


// use grid's print prefix but more conveniently
#define grid_printf(...)\
{\
  char _buf[1024];\
  sprintf(_buf, __VA_ARGS__);\
  std::cout << GridLogMessage << _buf;\
}
#define grid_printf_flush(...)\
{\
  grid_printf(__VA_ARGS__);\
  fflush(stdout);\
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



// helper function to handle PVector
template<typename Field>
void fillPVector(PVector<Field>& p, std::vector<Field>& f) {
  for(int i=0; i<f.size(); i++)
    p.push_back(&f[i]);
}


// helper function to handle PVector
template<typename Field>
void copyFields(PVector<Field>& out, const PVector<Field>& in, long n_virtual) {
  assert(out.size() == in.size());
  assert(out.size() % n_virtual == 0);
  for(int v=0; v<out.size(); v++) {
    out[v] = in[v];
  }
}


// helper function to handle PVector
template<class Field>
void conformable(GridBase* grid, const PVector<Field>& field) {
  for(int v=0; v<field.size(); v++) {
    conformable(grid, field[v].Grid());
  }
}


// helper function to handle PVector
template<class Field>
void constantCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    out[v].Checkerboard() = in[v].Checkerboard();
  }
}


// helper function to handle PVector
template<class Field>
void changingCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    if      (in[v].Checkerboard() == Even) out[v].Checkerboard() = Odd;
    else if (in[v].Checkerboard() == Odd)  out[v].Checkerboard() = Even;
    else assert(0);
  }
}


// helper function to handle PVector
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
      grid_printf("vector index = %d virtual index = %d : "
                  "norm2(reference), norm2(%s), abs.deviation, rel.deviation: "
                  "%f %f %f %f -> check %s\n",
                  vec_i, virtual_i, name.c_str(),
                  norm2(ref[v]), norm2(res[v]), absDev, relDev, ((relDev < checkTolerance) ? "passed" : "failed")
                  );

      ret = ret && relDev <= checkTolerance;
    }
  }
  return ret;
}


// helper function to handle PVector
template<typename Field>
void printNorms(const PVector<Field>& field, long n_virtual, const std::string& name, const std::string& comment) {
  assert(field.size() % n_virtual == 0);
  long  n_vec = field.size() / n_virtual;
  for(int vec_i=0; vec_i<n_vec; vec_i++) {
    for(int virtual_i=0; virtual_i<n_virtual; virtual_i++) {
      auto v = vec_i * n_virtual + virtual_i;
      grid_printf("%s: vector index = %d virtual index = %d : norm2(%s): %f\n",
                  comment.c_str(), vec_i, virtual_i, name, norm2(field[v]));
    }
  }
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
    v.push_back(l[k].View(mode));\
  typename std::remove_reference<decltype(v[0])>::type* p = &v[0];
#define VECTOR_VIEW_CLOSE_POINTER(v,p)                   \
  for(int k=0;k<v.size();k++) v[k].ViewClose();\
  p = nullptr;


// helper function to ease command line interaction
int readFromCommandLineInt(int* argc, char*** argv, const std::string& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}


// helper function to ease command line interaction
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


template<class FermionField>
class MultiArgFermionOperatorBase {
};


// this class makes CoarsenedMatrix compatible with virtual fields by looping over instances
template<class Fobj,class CComplex,int NbasisVirtual>
class CoarsenedMatrixWrapper : public MultiArgFermionOperatorBase<PVector<Lattice<iVector<CComplex,NbasisVirtual>>>> {
public: // type definitions ///////////////////////////////////////////////////

  // the type it wraps
  typedef CoarsenedMatrix<Fobj,CComplex,NbasisVirtual> VirtualCoarsenedMatrix;

  // site-wise types
  typedef         iVector<CComplex, NbasisVirtual>          SiteSpinor;
  typedef         iMatrix<CComplex, NbasisVirtual>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, NbasisVirtual>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types = virtual fields
  typedef Lattice<SiteSpinor>             VirtualFermionField;
  typedef Lattice<SiteMatrix>             VirtualLinkField;
  typedef Lattice<DoubleStoredSiteMatrix> VirtualDoubleStoredGaugeField;

  // physical fields, used internally
  typedef std::vector<VirtualFermionField>           PhysicalFermionField;
  typedef std::vector<VirtualLinkField>              PhysicalLinkField;
  typedef std::vector<VirtualDoubleStoredGaugeField> PhysicalGaugeField;

  // used by the outside world
  typedef PVector<VirtualFermionField>                FermionField;
  typedef PVector<VirtualLinkField>                   LinkField;
  typedef PVector<VirtualLinkField>                   GaugeField;
  typedef CartesianStencil<SiteSpinor,SiteSpinor,int> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;

  // sanity checks
  static_assert(std::is_same<SiteSpinor,          typename VirtualCoarsenedMatrix::siteVector>::value,   "types must match");
  static_assert(std::is_same<SiteMatrix,          typename VirtualCoarsenedMatrix::Cobj>::value,         "types must match");
  static_assert(std::is_same<VirtualFermionField, typename VirtualCoarsenedMatrix::CoarseVector>::value, "types must match");
  static_assert(std::is_same<VirtualLinkField,    typename VirtualCoarsenedMatrix::CoarseMatrix>::value, "types must match");

private: // member data ///////////////////////////////////////////////////////

  uint64_t link_n_virtual_;
  uint64_t fermion_n_virtual_;

  std::vector<VirtualCoarsenedMatrix> mat;
  PhysicalFermionField tmp;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MAccumTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

public: // member functions ///////////////////////////////////////////////////

  void M_gptdefault_loop_gpt(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MCalls++;
    MTotalTime -= usecond();
    const int Narg = numArg(in, in_n_virtual, out, out_n_virtual);
    for(int arg=0; arg<Narg; arg++) {
      for(int v_col=0; v_col<fermion_n_virtual_; v_col++) {
        for(int v_row=0; v_row<fermion_n_virtual_; v_row++) {
          int idx_mat = v_col * fermion_n_virtual_ + v_row;
          int idx_in = arg*fermion_n_virtual_+v_col;
          if (v_col == 0) {
            mat[idx_mat].M(in[idx_in], out[arg*fermion_n_virtual_+v_row]);
          } else {
            mat[idx_mat].M(in[idx_in], tmp[v_row]);
          }
        }
        if (v_col != 0) {
          for(int v_row=0; v_row<fermion_n_virtual_; v_row++) {
            MAccumTime -= usecond();
            out[arg*fermion_n_virtual_+v_row] += tmp[v_row];
            MAccumTime += usecond();
          }
        }
      }
    }
    MTotalTime += usecond();
  }

  void M_gptdefault_loop_natural(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MCalls++;
    MTotalTime -= usecond();
    const int Narg = numArg(in, in_n_virtual, out, out_n_virtual);
    for(int arg=0; arg<Narg; arg++) {
      for(int v_row=0; v_row<fermion_n_virtual_; v_row++) {
        for(int v_col=0; v_col<fermion_n_virtual_; v_col++) {
          // int idx_mat = v_row * fermion_n_virtual_ + v_col; // this would be true if this was also stored this way but it isn't
          int idx_mat = v_col * fermion_n_virtual_ + v_row;
          int idx_in = arg*fermion_n_virtual_+v_col;
          if (v_col == 0) {
            mat[idx_mat].M(in[idx_in], out[arg*fermion_n_virtual_+v_row]);
          } else {
            mat[idx_mat].M(in[idx_in], tmp[v_row]);
            MAccumTime -= usecond();
            out[arg*fermion_n_virtual_+v_row] += tmp[v_row];
            MAccumTime += usecond();
          }
        }
      }
    }
    MTotalTime += usecond();
  }

  void Report(int Nvec) {
    // sum contributions from instances
    for(int v=0; v<link_n_virtual_; v++) {
      MMiscTime += mat[v].MMiscTime;
      MViewTime += mat[v].MViewTime;
      MCommTime += mat[v].MCommTime;
      MComputeTime += mat[v].MComputeTime;
    }

    GridBase* grid_ = mat[0]._grid;
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];
    RealD nbasis = NbasisVirtual * fermion_n_virtual_;

    if ( MCalls > 0 ) {
      grid_printf("#### M calls report\n");
      grid_printf("CoarsenedMatrixList Number of Calls                         : %d\n", (int)MCalls);
      grid_printf("CoarsenedMatrixList MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_printf("CoarsenedMatrixList ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_printf("CoarsenedMatrixList AccumTime  /Calls, AccumTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MAccumTime  /MCalls, MAccumTime,   MAccumTime  /MTotalTime*100);
      grid_printf("CoarsenedMatrixList CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_printf("CoarsenedMatrixList ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %%)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_printf("CoarsenedMatrixList TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);

      // Average the compute time
      grid_->GlobalSum(MComputeTime);
      MComputeTime/=Nproc;
      RealD complex_words = 2;
      RealD prec_bytes    = getPrecision<typename CComplex::vector_type>::value * 4; // 4 for float, 8 for double
      RealD flop_per_site = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * Nvec;
      RealD word_per_site = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * Nvec;
      RealD byte_per_site = word_per_site * complex_words * prec_bytes;
      RealD mflops = flop_per_site*volume*MCalls/MComputeTime;
      RealD mbytes = byte_per_site*volume*MCalls/MComputeTime;
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call                : %.0f, %.0f\n", mflops, mbytes);
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call per rank       : %.0f, %.0f\n", mflops/Nproc, mbytes/Nproc);
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call per node       : %.0f, %.0f\n", mflops/Nnode, mbytes/Nnode);

      RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime);
      RealD Fullmbytes = byte_per_site*volume*MCalls/(MTotalTime);
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call (full)         : %.0f, %.0f\n", Fullmflops, Fullmbytes);
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call per rank (full): %.0f, %.0f\n", Fullmflops/Nproc, Fullmbytes/Nproc);
      grid_printf("CoarsenedMatrixList Average mflops/s, mbytes/s per call per node (full): %.0f, %.0f\n", Fullmflops/Nnode, Fullmbytes/Nnode);
    }
  }

  void ZeroCounters() {
    MCalls       = 0;
    MMiscTime    = 0;
    MViewTime    = 0;
    MAccumTime   = 0;
    MCommTime    = 0;
    MComputeTime = 0;
    MTotalTime   = 0;
    for(int v=0; v<link_n_virtual_; v++) {
      mat[v].ZeroCounters();
    }
  }

  CoarsenedMatrixWrapper(const PVector<VirtualLinkField>& Uc,
                         const PVector<VirtualLinkField>& UcSelfInv,
                         GridCartesian&                   grid,
                         GridRedBlackCartesian&           rbGrid,
                         int                              makeHermitian)
    : link_n_virtual_(UcSelfInv.size())
    , fermion_n_virtual_(uint64_t(sqrt(UcSelfInv.size())))
    // , mat(UcSelfInv.size(), {grid, rbGrid, makeHermitian})
    , tmp(fermion_n_virtual_, &grid)
  {
    // necessary for nvcc -- begin ////////////////////////////////////////////
    mat.reserve(UcSelfInv.size());
    for(int i=0; i<UcSelfInv.size(); i++) {
      mat.emplace_back(grid, rbGrid, makeHermitian);
    }
    // necessary for nvcc -- end //////////////////////////////////////////////
    for(int i=0; i<Uc.size(); i++) {
      int p=i/link_n_virtual_; int v=i%link_n_virtual_; // p = point = slower, v = virtual = faster
      mat[v].A[p] = Uc[i];
    }
    ZeroCounters();
  }

private: // member functions //////////////////////////////////////////////////

  int numArg(const FermionField& a, uint64_t a_n_virtual, const FermionField& b, uint64_t b_n_virtual) const {
    int a_size = a.size(); int b_size = b.size();
    assert(a_size == b_size);
    assert(a_n_virtual == b_n_virtual);
    assert(a_n_virtual == fermion_n_virtual_);
    assert(a_size >= a_n_virtual);
    assert(a_size % a_n_virtual == 0);
    return a_size / a_n_virtual;
  }
};


template<class CComplex,int NbasisVirtual>
class MultiArgVirtualCoarsenedMatrix : public MultiArgFermionOperatorBase<PVector<Lattice<iVector<CComplex,NbasisVirtual>>>> {
public: // type definitions ///////////////////////////////////////////////////

  // site-wise types
  typedef         iVector<CComplex, NbasisVirtual>          SiteSpinor;
  typedef         iMatrix<CComplex, NbasisVirtual>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, NbasisVirtual>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types = virtual fields
  typedef Lattice<SiteSpinor>             VirtualFermionField;
  typedef Lattice<SiteMatrix>             VirtualLinkField;
  typedef Lattice<SiteMatrix>             VirtualGridLayoutGaugeField;
  typedef Lattice<DoubleStoredSiteMatrix> VirtualDoubleStoredGaugeField;

  // physical fields, used internally
  typedef std::vector<VirtualFermionField>           PhysicalFermionField;
  typedef std::vector<VirtualLinkField>              PhysicalLinkField;
  typedef std::vector<VirtualGridLayoutGaugeField>   PhysicalGridLayoutGaugeField;
  typedef std::vector<VirtualDoubleStoredGaugeField> PhysicalGaugeField;

  // used by the outside world
  typedef PVector<VirtualFermionField>                FermionField;
  typedef PVector<VirtualLinkField>                   LinkField;
  typedef PVector<VirtualGridLayoutGaugeField>        GridLayoutGaugeField;
  typedef CartesianStencil<SiteSpinor,SiteSpinor,int> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;

private: // member data ///////////////////////////////////////////////////////

  Geometry geom_;
  Geometry geomMultiArg_;

  GridBase* grid_;
  GridBase* cbGrid_;
  GridBase* gridMultiArg_;
  GridBase* cbGridMultiArg_;

  bool hermitianOverall_;
  bool hermitianSelf_;

  uint64_t link_n_virtual_;
  uint64_t fermion_n_virtual_;
  uint64_t n_arg_;

  Stencil stencil_;
  // Stencil stencilEven_;
  // Stencil stencilOdd_;
  Stencil stencilMultiArg_;
  // Stencil stencilEvenMultiArg_;
  // Stencil stencilOddMultiArg_;

  PhysicalGaugeField Uc_;
  // PhysicalGaugeField UcEven_;
  // PhysicalGaugeField UcOdd_;

  PhysicalGridLayoutGaugeField UcGridLayout_;

  PhysicalLinkField UcSelfInv_;
  // PhysicalLinkField UcSelfInvEven_;
  // PhysicalLinkField UcSelfInvOdd_;

  VirtualFermionField tmpMultiArg_;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MView2Time;
  double MCopyTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

public: // member functions (implementing interface) //////////////////////////

  GridBase* Grid()                { return FermionGrid(); }
  GridBase* RedBlackGrid()        { return FermionRedBlackGrid(); }
  GridBase* FermionGrid()         { return grid_; }
  GridBase* FermionRedBlackGrid() { return cbGrid_; }
  GridBase* GaugeGrid()           { return grid_; }
  GridBase* GaugeRedBlackGrid()   { return cbGrid_; }

  int ConstEE()     { return 0; }
  int isTrivialEE() { return 0; }

  void M_loopinternal_gridlayout(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: most basic version looping over Grid's code internally -- Lorentz in std::vector
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, UcGridLayout_v, UcGridLayout_p, AcceleratorRead);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_row=0; v_row<NvirtualFermion; v_row++) {
        const int v_arg_row = arg*NvirtualFermion+v_row;

        MViewTime -= usecond();
        autoView(out_v , out[v_arg_row], AcceleratorWrite);
        MViewTime += usecond();
        for(int v_col=0; v_col<NvirtualFermion; v_col++) {
          const int v_arg_col = arg*NvirtualFermion+v_col;

          MCommTime -= usecond();
          stencil_.HaloExchange(in[v_arg_col], compressor);
          MCommTime += usecond();

          MViewTime -= usecond();
          autoView(stencil_v  , stencil_,  AcceleratorRead);
          autoView(in_v ,       in[v_arg_col], AcceleratorRead);
          MViewTime += usecond();

          typedef decltype(coalescedRead(in_v[0])) calcVector;
          typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          MComputeTime -= usecond();
          accelerator_for(idx, Nsite, Nsimd, {
            const int ss = idx;

            calcVector res;
            calcVector nbr;
            int ptype;
            StencilEntry *SE;

            if (v_col == 0)
              res = Zero();
            else
              res = coalescedRead(out_v[ss]);

            for(int point=0; point<Npoint; point++) {
              SE=stencil_v.GetEntry(ptype,point,ss);

              if(SE->_is_local) {
                nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
              } else {
                nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
              }
              acceleratorSynchronise();

              // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_row_col][ss])*nbr;
              // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_col_row][ss])*nbr;
              res = res + coalescedRead(UcGridLayout_p[v_row*NvirtualFermion*Npoint+v_col*Npoint+point][ss])*nbr;
              // res = res + coalescedRead(UcGridLayout_p[v_col*NvirtualFermion*Npoint+v_row*Npoint+point][ss])*nbr;
            }
            coalescedWrite(out_v[ss],res);
          });
          MComputeTime += usecond();
        }
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcGridLayout_v, UcGridLayout_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_loopinternal_tensorlayout(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: most basic version looping over Grid's code internally -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_row=0; v_row<NvirtualFermion; v_row++) {
        const int v_arg_row = arg*NvirtualFermion+v_row;

        MViewTime -= usecond();
        autoView(out_v , out[v_arg_row], AcceleratorWrite);
        MViewTime += usecond();
        for(int v_col=0; v_col<NvirtualFermion; v_col++) {
          const int v_arg_col = arg*NvirtualFermion+v_col;

          MCommTime -= usecond();
          stencil_.HaloExchange(in[v_arg_col], compressor);
          MCommTime += usecond();

          MViewTime -= usecond();
          autoView(stencil_v  , stencil_,  AcceleratorRead);
          autoView(in_v ,       in[v_arg_col], AcceleratorRead);
          MViewTime += usecond();

          typedef decltype(coalescedRead(in_v[0])) calcVector;
          typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          MComputeTime -= usecond();
          accelerator_for(idx, Nsite, Nsimd, {
            const int ss = idx;

            calcVector res;
            calcVector nbr;
            int ptype;
            StencilEntry *SE;

            if (v_col == 0)
              res = Zero();
            else
              res = coalescedRead(out_v[ss]);

            for(int point=0; point<Npoint; point++) {
              SE=stencil_v.GetEntry(ptype,point,ss);

              if(SE->_is_local) {
                nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
              } else {
                nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
              }
              acceleratorSynchronise();

              res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point))*nbr;
              // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point))*nbr;
            }
            coalescedWrite(out_v[ss],res);
          });
          MComputeTime += usecond();
        }
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_loopinternal_gridlayout_parchange(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index -- Lorentz in std::vector
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, UcGridLayout_v, UcGridLayout_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_col=0; v_col<NvirtualFermion; v_col++) {
        const int v_arg_col = arg*NvirtualFermion+v_col;

        MCommTime -= usecond();
        stencil_.HaloExchange(in[v_arg_col], compressor);
        MCommTime += usecond();

        MViewTime -= usecond();
        autoView(stencil_v  , stencil_,  AcceleratorRead);
        autoView(in_v ,       in[v_arg_col], AcceleratorRead);
        MViewTime += usecond();

        typedef decltype(coalescedRead(in_v[0])) calcVector;
        typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

        MComputeTime -= usecond();
        accelerator_for(idx, Nsite*NvirtualFermion, Nsimd, {
                int _idx  = idx;
          const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
          const int ss    = _idx%Nsite; _idx/=Nsite;

          const int v_arg_row = arg*NvirtualFermion+v_row;
          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          calcVector res;
          calcVector nbr;
          int ptype;
          StencilEntry *SE;

          if (v_col == 0)
            res = Zero();
          else
            res = coalescedRead(out_p[v_arg_row][ss]);

          for(int point=0; point<Npoint; point++) {
            SE=stencil_v.GetEntry(ptype,point,ss);

            if(SE->_is_local) {
              nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
            } else {
              nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
            }
            acceleratorSynchronise();

            // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_row_col][ss])*nbr;
            // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_col_row][ss])*nbr;
            res = res + coalescedRead(UcGridLayout_p[v_row*NvirtualFermion*Npoint+v_col*Npoint+point][ss])*nbr;
            // res = res + coalescedRead(UcGridLayout_p[v_col*NvirtualFermion*Npoint+v_row*Npoint+point][ss])*nbr;
          }
          coalescedWrite(out_p[v_arg_row][ss],res);
        });
        MComputeTime += usecond();
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcGridLayout_v, UcGridLayout_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_loopinternal_tensorlayout_parchange(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_col=0; v_col<NvirtualFermion; v_col++) {
        const int v_arg_col = arg*NvirtualFermion+v_col;

        MCommTime -= usecond();
        stencil_.HaloExchange(in[v_arg_col], compressor);
        MCommTime += usecond();

        MViewTime -= usecond();
        autoView(stencil_v  , stencil_,  AcceleratorRead);
        autoView(in_v ,       in[v_arg_col], AcceleratorRead);
        MViewTime += usecond();

        typedef decltype(coalescedRead(in_v[0])) calcVector;
        typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

        MComputeTime -= usecond();
        accelerator_for(idx, Nsite*NvirtualFermion, Nsimd, {
                int _idx  = idx;
          const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
          const int ss    = _idx%Nsite; _idx/=Nsite;

          const int v_arg_row = arg*NvirtualFermion+v_row;
          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          calcVector res;
          calcVector nbr;
          int ptype;
          StencilEntry *SE;

          if (v_col == 0)
            res = Zero();
          else
            res = coalescedRead(out_p[v_arg_row][ss]);

          for(int point=0; point<Npoint; point++) {
            SE=stencil_v.GetEntry(ptype,point,ss);

            if(SE->_is_local) {
              nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
            } else {
              nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
            }
            acceleratorSynchronise();

            res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point))*nbr;
            // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point))*nbr;
          }
          coalescedWrite(out_p[v_arg_row][ss],res);
        });
        MComputeTime += usecond();
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_loopinternal_tensorlayout_parchange_commsreduce(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    {
      MView2Time-=usecond();
      VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
      autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorWrite);
      MView2Time+=usecond();
      MCopyTime-=usecond();
      accelerator_for(sF, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmpMultiArg_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // printf("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
      MCopyTime+=usecond();
      MView2Time-=usecond();
      VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
      MView2Time+=usecond();
    }

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _idx  = idx;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        calcVector res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_p[v_arg_row][ss]);

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point))*nbr;
          // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point))*nbr;
        }
        coalescedWrite(out_p[v_arg_row][ss],res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_finegrained_loopinternal_gridlayout(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: most basic version looping over Grid's code internally -- Lorentz in std::vector
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, UcGridLayout_v, UcGridLayout_p, AcceleratorRead);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_row=0; v_row<NvirtualFermion; v_row++) {
        const int v_arg_row = arg*NvirtualFermion+v_row;

        MViewTime -= usecond();
        autoView(out_v , out[v_arg_row], AcceleratorWrite);
        MViewTime += usecond();
        for(int v_col=0; v_col<NvirtualFermion; v_col++) {
          const int v_arg_col = arg*NvirtualFermion+v_col;

          MCommTime -= usecond();
          stencil_.HaloExchange(in[v_arg_col], compressor);
          MCommTime += usecond();

          MViewTime -= usecond();
          autoView(stencil_v  , stencil_,  AcceleratorRead);
          autoView(in_v ,       in[v_arg_col], AcceleratorRead);
          MViewTime += usecond();

          typedef decltype(coalescedRead(in_v[0])) calcVector;
          typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          MComputeTime -= usecond();
          accelerator_for(idx, Nsite*NbasisVirtual, Nsimd, {
                  int _idx = idx;
            const int b    = _idx%NbasisVirtual; _idx/=NbasisVirtual;
            const int ss   = _idx%Nsite; _idx/=Nsite;

            calcComplex res;
            calcVector nbr;
            int ptype;
            StencilEntry *SE;

            if (v_col == 0)
              res = Zero();
            else
              res = coalescedRead(out_v[ss](b));

            for(int point=0; point<Npoint; point++) {
              SE=stencil_v.GetEntry(ptype,point,ss);

              if(SE->_is_local) {
                nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
              } else {
                nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
              }
              acceleratorSynchronise();

              for(int bb=0;bb<NbasisVirtual;bb++) {
                // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_row_col][ss](b,bb))*nbr(bb);
                // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_col_row][ss](b,bb))*nbr(bb);
                res = res + coalescedRead(UcGridLayout_p[v_row*NvirtualFermion*Npoint+v_col*Npoint+point][ss](b,bb))*nbr(bb);
                // res = res + coalescedRead(UcGridLayout_p[v_col*NvirtualFermion*Npoint+v_row*Npoint+point][ss](b,bb))*nbr(bb);
              }
            }
            coalescedWrite(out_v[ss](b),res);
          });
          MComputeTime += usecond();
        }
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcGridLayout_v, UcGridLayout_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_finegrained_loopinternal_tensorlayout(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: most basic version looping over Grid's code internally -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_row=0; v_row<NvirtualFermion; v_row++) {
        const int v_arg_row = arg*NvirtualFermion+v_row;

        MViewTime -= usecond();
        autoView(out_v , out[v_arg_row], AcceleratorWrite);
        MViewTime += usecond();
        for(int v_col=0; v_col<NvirtualFermion; v_col++) {
          const int v_arg_col = arg*NvirtualFermion+v_col;

          MCommTime -= usecond();
          stencil_.HaloExchange(in[v_arg_col], compressor);
          MCommTime += usecond();

          MViewTime -= usecond();
          autoView(stencil_v  , stencil_,  AcceleratorRead);
          autoView(in_v ,       in[v_arg_col], AcceleratorRead);
          MViewTime += usecond();

          typedef decltype(coalescedRead(in_v[0])) calcVector;
          typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          MComputeTime -= usecond();
          accelerator_for(idx, Nsite*NbasisVirtual, Nsimd, {
                  int _idx = idx;
            const int b    = _idx%NbasisVirtual; _idx/=NbasisVirtual;
            const int ss   = _idx%Nsite; _idx/=Nsite;

            calcComplex res;
            calcVector nbr;
            int ptype;
            StencilEntry *SE;

            if (v_col == 0)
              res = Zero();
            else
              res = coalescedRead(out_v[ss](b));

            for(int point=0; point<Npoint; point++) {
              SE=stencil_v.GetEntry(ptype,point,ss);

              if(SE->_is_local) {
                nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
              } else {
                nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
              }
              acceleratorSynchronise();

              for(int bb=0;bb<NbasisVirtual;bb++) {
                res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point)(b,bb))*nbr(bb);
                // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point)(b,bb))*nbr(bb);
              }
            }
            coalescedWrite(out_v[ss](b),res);
          });
          MComputeTime += usecond();
        }
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_finegrained_loopinternal_gridlayout_parchange(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index -- Lorentz in std::vector
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, UcGridLayout_v, UcGridLayout_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_col=0; v_col<NvirtualFermion; v_col++) {
        const int v_arg_col = arg*NvirtualFermion+v_col;

        MCommTime -= usecond();
        stencil_.HaloExchange(in[v_arg_col], compressor);
        MCommTime += usecond();

        MViewTime -= usecond();
        autoView(stencil_v  , stencil_,  AcceleratorRead);
        autoView(in_v ,       in[v_arg_col], AcceleratorRead);
        MViewTime += usecond();

        typedef decltype(coalescedRead(in_v[0])) calcVector;
        typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

        MComputeTime -= usecond();
        accelerator_for(idx, Nsite*NvirtualFermion*NbasisVirtual, Nsimd, {
                int _idx  = idx;
          const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
          const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
          const int ss    = _idx%Nsite; _idx/=Nsite;

          const int v_arg_row = arg*NvirtualFermion+v_row;
          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          calcComplex res;
          calcVector nbr;
          int ptype;
          StencilEntry *SE;

          if (v_col == 0)
            res = Zero();
          else
            res = coalescedRead(out_p[v_arg_row][ss](b));

          for(int point=0; point<Npoint; point++) {
            SE=stencil_v.GetEntry(ptype,point,ss);

            if(SE->_is_local) {
              nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
            } else {
              nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
            }
            acceleratorSynchronise();

            for(int bb=0;bb<NbasisVirtual;bb++) {
              // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_row_col][ss](b,bb))*nbr(bb);
              // res = res + coalescedRead(UcGridLayout_p[point*NvirtualLink+v_col_row][ss](b,bb))*nbr(bb);
              res = res + coalescedRead(UcGridLayout_p[v_row*NvirtualFermion*Npoint+v_col*Npoint+point][ss](b,bb))*nbr(bb);
              // res = res + coalescedRead(UcGridLayout_p[v_col*NvirtualFermion*Npoint+v_row*Npoint+point][ss](b,bb))*nbr(bb);
            }
          }
          coalescedWrite(out_p[v_arg_row][ss](b),res);
        });
        MComputeTime += usecond();
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcGridLayout_v, UcGridLayout_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_finegrained_loopinternal_tensorlayout_parchange(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int arg=0; arg<Narg; arg++) {
      for(int v_col=0; v_col<NvirtualFermion; v_col++) {
        const int v_arg_col = arg*NvirtualFermion+v_col;

        MCommTime -= usecond();
        stencil_.HaloExchange(in[v_arg_col], compressor);
        MCommTime += usecond();

        MViewTime -= usecond();
        autoView(stencil_v  , stencil_,  AcceleratorRead);
        autoView(in_v ,       in[v_arg_col], AcceleratorRead);
        MViewTime += usecond();

        typedef decltype(coalescedRead(in_v[0])) calcVector;
        typedef decltype(coalescedRead(in_v[0](0))) calcComplex;

        MComputeTime -= usecond();
        accelerator_for(idx, Nsite*NvirtualFermion*NbasisVirtual, Nsimd, {
                int _idx  = idx;
          const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
          const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
          const int ss    = _idx%Nsite; _idx/=Nsite;

          const int v_arg_row = arg*NvirtualFermion+v_row;
          const int v_row_col = v_row * NvirtualFermion + v_col;
          const int v_col_row = v_col * NvirtualFermion + v_row;

          calcComplex res;
          calcVector nbr;
          int ptype;
          StencilEntry *SE;

          if (v_col == 0)
            res = Zero();
          else
            res = coalescedRead(out_p[v_arg_row][ss](b));

          for(int point=0; point<Npoint; point++) {
            SE=stencil_v.GetEntry(ptype,point,ss);

            if(SE->_is_local) {
              nbr = coalescedReadPermute(in_v[SE->_offset],ptype,SE->_permute);
            } else {
              nbr = coalescedRead(stencil_v.CommBuf()[SE->_offset]);
            }
            acceleratorSynchronise();

            for(int bb=0;bb<NbasisVirtual;bb++) {
              res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point)(b,bb))*nbr(bb);
              // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point)(b,bb))*nbr(bb);
            }
          }
          coalescedWrite(out_p[v_arg_row][ss](b),res);
        });
        MComputeTime += usecond();
      }
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

  void M_finegrained_loopinternal_tensorlayout_parchange_commsreduce(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(Grid(), in);
    conformable(Grid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    {
      MView2Time-=usecond();
      VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
      autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorWrite);
      MView2Time+=usecond();
      MCopyTime-=usecond();
      accelerator_for(sF, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmpMultiArg_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // printf("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
      MCopyTime+=usecond();
      MView2Time-=usecond();
      VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
      MView2Time+=usecond();
    }

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_p[v_arg_row][ss](b));

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
            res = res + coalescedRead(Uc_p[v_row*NvirtualFermion+v_col][ss](point)(b,bb))*nbr(bb);
            // res = res + coalescedRead(Uc_p[v_col*NvirtualFermion+v_row][ss](point)(b,bb))*nbr(bb);
          }
        }
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
  }

public: // member functions (additional) //////////////////////////////////////

  void ImportGauge(const PVector<VirtualLinkField>& Uc, const PVector<VirtualLinkField>& UcSelfInv) {
    assert(Uc.size() == geom_.npoint * Uc_.size());
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(Grid(), Uc);
    conformable(Grid(), UcSelfInv);

    const int Nsite           = Grid()->oSites();
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Npoint          = geom_.npoint;
    const int NvirtualLink    = link_n_virtual_;
    const int NvirtualFermion = fermion_n_virtual_;

    // NOTE: can't use PokeIndex here because of different tensor depths
    VECTOR_VIEW_OPEN_POINTER(Uc_,           Uc_member_v,             Uc_member_p,             AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(UcGridLayout_, Uc_member_grid_layout_v, Uc_member_grid_layout_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(Uc,            Uc_arg_v,                Uc_arg_p,                AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order
      const int link_idx_row_col = v_row * NvirtualFermion + v_col;
      const int link_idx_col_row = v_col * NvirtualFermion + v_row;
      for(int p=0; p<Npoint; p++) {
        const int gauge_idx_point_row_col = p * NvirtualLink + v_row * NvirtualFermion + v_col;
        const int gauge_idx_point_col_row = p * NvirtualLink + v_col * NvirtualFermion + v_row;
        const int gauge_idx_row_col_point = v_row * NvirtualFermion * Npoint + v_col * Npoint + p;
        const int gauge_idx_col_row_point = v_col * NvirtualFermion * Npoint + v_row * Npoint + p;

        accelerator_for(ss, Nsite, Nsimd, {
          // new layout with Lorentz in tensor
          coalescedWrite(Uc_member_p[link_idx_row_col][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // change to col faster index -> row-major order -> with transpose
          // coalescedWrite(Uc_member_p[link_idx_col_row][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // keep      row faster index -> col-major order -> without transpose

          // grid's layout with Lorentz in std::vector
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_point_row_col][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // point slow, virtual fast, col faster index -> virtual in row-major order
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_point_col_row][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // point slow, virtual fast, row faster index -> virtual in col-major order
          coalescedWrite(Uc_member_grid_layout_p[gauge_idx_row_col_point][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // virtual slow, point fast, col faster index -> virtual in row-major order
          // coalescedWrite(Uc_member_grid_layout_p[gauge_idx_col_row_point][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss])); // virtual slow, point fast, row faster index -> virtual in col-major order
        });
      }
    }
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_v, Uc_member_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_grid_layout_v, Uc_member_grid_layout_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_arg_v, Uc_arg_p);

    for(int v=0; v<NvirtualLink; v++) UcSelfInv_[v] = UcSelfInv[v];
    grid_printf("ImportGauge of new Coarse Operator finished\n");
  }

  void PickCheckerboards() {
    grid_printf("PickCheckerboards of new Coarse Operator finished\n");
  }

  void Report(int Nvec) {
    assert(Nvec == n_arg_);
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];
    RealD nbasis = NbasisVirtual * fermion_n_virtual_;

    if ( MCalls > 0 ) {
      grid_printf("#### M calls report\n");
      grid_printf("CoarseOperator Number of Calls                         : %d\n", (int)MCalls);
      grid_printf("CoarseOperator MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_printf("CoarseOperator ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_printf("CoarseOperator View2Time  /Calls, View2Time   : %10.2f us, %10.2f us (= %6.2f %%)\n", MView2Time  /MCalls, MView2Time,   MView2Time  /MTotalTime*100);
      grid_printf("CoarseOperator CopyTime   /Calls, CopyTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCopyTime   /MCalls, MCopyTime,    MCopyTime   /MTotalTime*100);
      grid_printf("CoarseOperator CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_printf("CoarseOperator ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %%)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_printf("CoarseOperator TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);

      // Average the compute time
      grid_->GlobalSum(MComputeTime);
      MComputeTime/=Nproc;
      RealD complex_words = 2;
      RealD prec_bytes    = getPrecision<typename CComplex::vector_type>::value * 4; // 4 for float, 8 for double
      RealD flop_per_site = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * Nvec;
      RealD word_per_site = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * Nvec;
      RealD byte_per_site = word_per_site * complex_words * prec_bytes;
      RealD mflops = flop_per_site*volume*MCalls/MComputeTime;
      RealD mbytes = byte_per_site*volume*MCalls/MComputeTime;
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call                : %.0f, %.0f\n", mflops, mbytes);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per rank       : %.0f, %.0f\n", mflops/Nproc, mbytes/Nproc);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per node       : %.0f, %.0f\n", mflops/Nnode, mbytes/Nnode);

      RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime);
      RealD Fullmbytes = byte_per_site*volume*MCalls/(MTotalTime);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call (full)         : %.0f, %.0f\n", Fullmflops, Fullmbytes);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per rank (full): %.0f, %.0f\n", Fullmflops/Nproc, Fullmbytes/Nproc);
      grid_printf("CoarseOperator Average mflops/s, mbytes/s per call per node (full): %.0f, %.0f\n", Fullmflops/Nnode, Fullmbytes/Nnode);

      grid_printf("CoarseOperator Stencil\n"); stencil_.Report();
      // grid_printf("CoarseOperator StencilEven\n"); stencilEven_.Report();
      // grid_printf("CoarseOperator StencilOdd\n"); stencilOdd_.Report();
      grid_printf("CoarseOperator StencilMultiArg\n"); stencilMultiArg_.Report();
      // grid_printf("CoarseOperator StencilMultiArgEven\n"); stencilMultiArgEven_.Report();
      // grid_printf("CoarseOperator StencilMultiArgOdd\n"); stencilMultiArgOdd_.Report();
    }
    grid_printf("Report of new Coarse Operator finished\n");
  }

  void ZeroCounters() {
    MCalls       = 0; // ok
    MMiscTime    = 0;
    MViewTime    = 0;
    MView2Time   = 0;
    MCopyTime    = 0;
    MCommTime    = 0;
    MComputeTime = 0;
    MTotalTime   = 0;

    stencil_.ZeroCounters();
    // stencilEven_.ZeroCounters();
    // stencilOdd_.ZeroCounters();
    stencilMultiArg_.ZeroCounters();
    // stencilMultiArgEven_.ZeroCounters();
    // stencilMultiArgOdd_.ZeroCounters();
  }

  MultiArgVirtualCoarsenedMatrix(const PVector<VirtualLinkField>& Uc,
                                 const PVector<VirtualLinkField>& UcSelfInv,
                                 GridCartesian&                   grid,
                                 GridRedBlackCartesian&           rbGrid,
                                 int                              makeHermitian,
                                 int                              numArg)
    : geom_(grid._ndimension)
    , geomMultiArg_(grid._ndimension+1)
    , grid_(&grid)
    , cbGrid_(&rbGrid)
    , gridMultiArg_(SpaceTimeGrid::makeFiveDimGrid(numArg*uint64_t(sqrt(UcSelfInv.size())), &grid))
    , cbGridMultiArg_(SpaceTimeGrid::makeFiveDimRedBlackGrid(numArg*uint64_t(sqrt(UcSelfInv.size())), &grid))
    , hermitianOverall_(makeHermitian)
    , hermitianSelf_(false)
    , link_n_virtual_(UcSelfInv.size())
    , fermion_n_virtual_(uint64_t(sqrt(UcSelfInv.size())))
    , n_arg_(numArg)
    , stencil_(grid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilEven_(cbGrid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilOdd_(cbGrid_, geom_.npoint, Odd, geom_.directions, geom_.displacements, 0)
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    // , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    // , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Odd, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , Uc_(link_n_virtual_, grid_)
    // , UcEven_(link_n_virtual_, cbGrid_)
    // , UcOdd_(link_n_virtual_, cbGrid_)
    , UcGridLayout_(geom_.npoint*link_n_virtual_, grid_)
    , UcSelfInv_(link_n_virtual_, grid_)
    // , UcSelfInvEven_(link_n_virtual_, cbGrid_)
    // , UcSelfInvOdd_(link_n_virtual_, cbGrid_)
    , tmpMultiArg_(gridMultiArg_)
  {
    ImportGauge(Uc, UcSelfInv);
    PickCheckerboards();
    ZeroCounters();
  }

private: // member functions //////////////////////////////////////////////////

  int numArg(const FermionField& a, uint64_t a_n_virtual, const FermionField& b, uint64_t b_n_virtual) const {
    int a_size = a.size(); int b_size = b.size();
    assert(a_size == b_size);
    assert(a_n_virtual == b_n_virtual);
    assert(a_n_virtual == fermion_n_virtual_);
    assert(a_size >= a_n_virtual);
    assert(a_size % a_n_virtual == 0);
    return a_size / a_n_virtual;
  }
};


template<typename vCoeff_t>
void runBenchmark(int* argc, char*** argv) {
  // precision
  static_assert(getPrecision<vCoeff_t>::value == 2 || getPrecision<vCoeff_t>::value == 1, "Incorrect precision"); // double or single
  std::string precision = (getPrecision<vCoeff_t>::value == 2 ? "double" : "single");

  // compile-time constants
  const int nbasis_virtual = NBASIS; static_assert((nbasis_virtual & 0x1) == 0, "");
  const int nsingle_virtual = nbasis_virtual/2;
  const int npoint = 2*Nd+1;

  // command line arguments
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  const int nvirtual = readFromCommandLineInt(argc, argv, "--nvirtual", 4);
  const int nvec = readFromCommandLineInt(argc, argv, "--nvec", 1);

  // dependent sizes
  const int nvirtual_fermion = nvirtual;
  const int nvirtual_link = nvirtual * nvirtual;
  const int nbasis = nbasis_virtual * nvirtual;
  const int nsingle = nbasis/2; // safe since we have even virtual fields

  // print info about run
  std::cout << GridLogMessage << "Compiled with nbasis_virtual = " << nbasis_virtual << std::endl;

  // setup grids
  GridCartesian* UGrid =
    SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                   GridDefaultSimd(Nd, vCoeff_t::Nsimd()),
                                   GridDefaultMpi());
  UGrid->show_decomposition();
  GridRedBlackCartesian* UGridRB = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  UGridRB->show_decomposition();

  // print info about run
  grid_printf("\n");
  grid_printf("Coarse operator Benchmark with\n");
  grid_printf("fdimensions         : [%d %d %d %d]\n", UGrid->_fdimensions[0], UGrid->_fdimensions[1], UGrid->_fdimensions[2], UGrid->_fdimensions[3]);
  grid_printf("precision           : %s\n", precision.c_str());
  grid_printf("nbasis_virtual      : %d\n", nbasis_virtual);
  grid_printf("nvirtual            : %d\n", nvirtual);
  grid_printf("nbasis              : %d\n", nbasis);
  grid_printf("nsingle             : %d\n", nsingle);
  grid_printf("nvec                : %d\n", nvec);
  grid_printf("nsimd               : %d\n", vCoeff_t::Nsimd());
  grid_printf("acc_threads         : %d\n", acceleratorThreads());
  grid_printf("blk_threads         : %d\n", vCoeff_t::Nsimd()*acceleratorThreads());
  grid_printf_flush("\n");

  // setup rng
  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG  pRNG(UGrid);
  pRNG.SeedFixedIntegers(seeds);

  // type definitions -- operators
  typedef CoarsenedMatrixWrapper<vSpinColourVector, iSinglet<vCoeff_t>, nbasis_virtual> DefaultCoarseOperator;
  typedef MultiArgVirtualCoarsenedMatrix<iSinglet<vCoeff_t>, nbasis_virtual>            NewCoarseOperator;

  // type definitions -- virtual = compiled fields
  typedef typename DefaultCoarseOperator::VirtualLinkField    VirtualDefaultCoarseMatrix;
  typedef typename DefaultCoarseOperator::VirtualFermionField VirtualDefaultCoarseVector;
  typedef typename NewCoarseOperator::VirtualLinkField        VirtualNewCoarseMatrix;
  typedef typename NewCoarseOperator::VirtualFermionField     VirtualNewCoarseVector;

  // type definitions -- physical fields
  typedef std::vector<VirtualDefaultCoarseMatrix>   PhysicalDefaultCoarseMatrix;
  typedef std::vector<VirtualDefaultCoarseVector>   PhysicalDefaultCoarseVector;
  typedef NewCoarseOperator                         PhysicalNewCoarseOperator;
  typedef std::vector<VirtualNewCoarseMatrix>       PhysicalNewCoarseMatrix;
  typedef std::vector<VirtualNewCoarseVector>       PhysicalNewCoarseVector;

  // type definitions -- pointers to physical fields (used by gpt)
  typedef PVector<VirtualDefaultCoarseMatrix>   DefaultCoarseMatrix;
  typedef PVector<VirtualDefaultCoarseVector>   DefaultCoarseVector;
  typedef PVector<VirtualNewCoarseMatrix>       NewCoarseMatrix;
  typedef PVector<VirtualNewCoarseVector>       NewCoarseVector;

  // sanity checks
  static_assert(std::is_same<VirtualDefaultCoarseMatrix,  VirtualNewCoarseMatrix>::value,  "types must match");
  static_assert(std::is_same<VirtualDefaultCoarseVector,  VirtualNewCoarseVector>::value,  "types must match");
  static_assert(std::is_same<PhysicalDefaultCoarseMatrix, PhysicalNewCoarseMatrix>::value, "types must match");
  static_assert(std::is_same<PhysicalDefaultCoarseVector, PhysicalNewCoarseVector>::value, "types must match");
  static_assert(std::is_same<DefaultCoarseMatrix,         NewCoarseMatrix>::value,         "types must match");
  static_assert(std::is_same<DefaultCoarseVector,         NewCoarseVector>::value,         "types must match");

  // more easy names (safe after sanity checks)
  typedef VirtualDefaultCoarseMatrix  VirtualCoarseMatrix;
  typedef VirtualDefaultCoarseVector  VirtualCoarseVector;
  typedef PhysicalDefaultCoarseMatrix PhysicalCoarseMatrix;
  typedef PhysicalDefaultCoarseVector PhysicalCoarseVector;
  typedef DefaultCoarseMatrix         CoarseMatrix;
  typedef DefaultCoarseVector         CoarseVector;

  // setup fields -- links
  PhysicalCoarseMatrix A_storage(npoint * nvirtual_link, UGrid); for(auto& elem: A_storage) random(pRNG, elem);
  PhysicalCoarseMatrix AselfInv_storage(nvirtual_link, UGrid);   for(auto& elem: AselfInv_storage) random(pRNG, elem);

  // point gpt fields to field storage -- links
  CoarseMatrix A;        fillPVector(A, A_storage);
  CoarseMatrix AselfInv; fillPVector(AselfInv, AselfInv_storage);

  // setup fields -- operators
  DefaultCoarseOperator op_default(A, AselfInv, *UGrid, *UGridRB, 0);
  NewCoarseOperator     op_new(A, AselfInv, *UGrid, *UGridRB, 0, nvec);

  // tear down links outside of operators (swap with empty vector safest to completely deallocate vector)
  decltype(A_storage)().swap(A_storage);
  decltype(AselfInv_storage)().swap(AselfInv_storage);
  // A_storage.clear();        A_storage.shrink_to_fit();
  // AselfInv_storage.clear(); AselfInv_storage.shrink_to_fit();

  // setup fields -- fermions
  PhysicalCoarseVector src_storage(nvec * nvirtual_fermion, UGrid);            for(auto& elem: src_storage) random(pRNG, elem);
  PhysicalCoarseVector tmp_gptdefault_storage(nvec * nvirtual_fermion, UGrid); for(auto& elem: tmp_gptdefault_storage) elem = Zero();
  PhysicalCoarseVector res_gptdefault_storage(nvec * nvirtual_fermion, UGrid); for(auto& elem: res_gptdefault_storage) elem = Zero();
  PhysicalCoarseVector res_versions_storage(nvec * nvirtual_fermion, UGrid);   for(auto& elem: res_versions_storage) elem = Zero();

  // point gpt fields to field storage -- fermions
  CoarseVector src;            fillPVector(src, src_storage);
  CoarseVector res_gptdefault; fillPVector(res_gptdefault, res_gptdefault_storage);
  CoarseVector res_versions;   fillPVector(res_versions, res_versions_storage);

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
  double flops_per_site = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * nvec;
  double words_per_site = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * nvec;
  double bytes_per_site = words_per_site * complex_words * prec_bytes;
  double flops          = flops_per_site * UGrid->gSites() * nIter;
  double words          = words_per_site * UGrid->gSites() * nIter;
  double nbytes         = bytes_per_site * UGrid->gSites() * nIter;

  // report calculated performance figures per site
  grid_printf("%12s: per-site values: flops = %f, words = %f, bytes = %f, flops/bytes = %f\n",
              "M", flops_per_site, words_per_site, bytes_per_site, flops_per_site/bytes_per_site);
  grid_printf_flush("\n");

#define BENCH_OPERATOR_KERNELVERSION(OPERATOR, KERNELVERSION, ...)\
  double secs_##KERNELVERSION;\
  {\
    for(int v=0; v<res_versions.size(); v++) res_versions[v] = Zero();\
    grid_printf_flush("warmup %s %s\n", #KERNELVERSION, precision.c_str());\
    for(auto n : {1, 2, 3, 4, 5}) OPERATOR.M_##KERNELVERSION(src, nvirtual_fermion, res_versions, nvirtual_fermion);\
    for(int v=0; v<res_versions.size(); v++) res_versions[v] = Zero();\
    OPERATOR.ZeroCounters();\
    grid_printf_flush("measurement %s %s\n", #KERNELVERSION, precision.c_str());\
    double t0 = usecond();\
    for(int n=0; n<nIter; n++) OPERATOR.M_##KERNELVERSION(src, nvirtual_fermion, res_versions, nvirtual_fermion); \
    double t1 = usecond();\
    secs_##KERNELVERSION = (t1-t0)/1e6;\
    if(strcmp(#KERNELVERSION, "gptdefault_loop_gpt"))\
      assert(resultsAgree(res_gptdefault, res_versions, nvirtual_fermion, #KERNELVERSION));\
    else\
      copyFields(res_gptdefault, res_versions, nvirtual_fermion);\
  }

#define PRINT_OPERATOR_KERNELVERSION(OPERATOR, KERNELVERSION) {          \
  double GFlopsPerSec_##KERNELVERSION = flops  / secs_##KERNELVERSION / 1e9;\
  double GBPerSec_##KERNELVERSION     = nbytes / secs_##KERNELVERSION / 1e9;\
  grid_printf("%d applications of M_%s\n", nIter, #KERNELVERSION);\
  grid_printf("    Time to complete            : %f s\n",        secs_##KERNELVERSION);\
  grid_printf("    Total performance           : %f GFlops/s\n", GFlopsPerSec_##KERNELVERSION);\
  grid_printf("    Effective memory bandwidth  : %f GB/s\n",     GBPerSec_##KERNELVERSION);\
  grid_printf_flush("\n");\
  OPERATOR.Report(nvec);\
  grid_printf_flush("\n");\
}

  BENCH_OPERATOR_KERNELVERSION(op_default, gptdefault_loop_gpt);                                         PRINT_OPERATOR_KERNELVERSION(op_default, gptdefault_loop_gpt);
  BENCH_OPERATOR_KERNELVERSION(op_default, gptdefault_loop_natural);                                     PRINT_OPERATOR_KERNELVERSION(op_default, gptdefault_loop_natural);
  #if !defined(GRID_CUDA) && !defined(GRID_HIP) && !defined(GRID_SYCL) // performs super bad on GPU
  BENCH_OPERATOR_KERNELVERSION(op_new,     loopinternal_gridlayout);                                     PRINT_OPERATOR_KERNELVERSION(op_new,     loopinternal_gridlayout);
  BENCH_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout);                                   PRINT_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout);
  BENCH_OPERATOR_KERNELVERSION(op_new,     loopinternal_gridlayout_parchange);                           PRINT_OPERATOR_KERNELVERSION(op_new,     loopinternal_gridlayout_parchange);
  BENCH_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout_parchange);                         PRINT_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout_parchange);
  BENCH_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout_parchange_commsreduce);             PRINT_OPERATOR_KERNELVERSION(op_new,     loopinternal_tensorlayout_parchange_commsreduce);
  #endif
  BENCH_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_gridlayout);                         PRINT_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_gridlayout);
  BENCH_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout);                       PRINT_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout);
  BENCH_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_gridlayout_parchange);               PRINT_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_gridlayout_parchange);
  BENCH_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout_parchange);             PRINT_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout_parchange);
  BENCH_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout_parchange_commsreduce); PRINT_OPERATOR_KERNELVERSION(op_new,     finegrained_loopinternal_tensorlayout_parchange_commsreduce);

#undef BENCH_OPERATOR_KERNELVERSION
#undef PRINT_OPERATOR_KERNELVERSION

  grid_printf("DONE WITH COARSE_OP BENCHMARKS in %s precision\n", precision.c_str());
  grid_printf_flush("\n");
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
