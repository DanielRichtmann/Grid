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

#ifdef GRID_CUDA
// #define CUDA_PROFILE
#endif

#ifdef CUDA_PROFILE
#include <cuda_profiler_api.h>
#endif

using namespace Grid;


// index within the triangle portion
accelerator_inline int index_triang(int i, int j) {
  const int Nred = 6;
  if (i == j)
    return 0;
  else if (i < j)
    return Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
  else // i > j
    return Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
}


// index within reduced (= diagonal + triangle portion)
accelerator_inline int index_red(int i, int j) {
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
              clover_red_v[ss]()(block)(index_red(i, j)) = clover_full_v[ss]()(s_row, s_col)(c_row, c_col);
          }
        }
      }
    }
  });
}

template<class CloverFullField, class CloverDiagonalField, class CloverTriangleField>
void convert_clover(const CloverFullField& clover_full, CloverDiagonalField& clover_diag, CloverTriangleField& clover_triang) {
  autoView(clover_full_v, clover_full, AcceleratorRead);
  autoView(clover_diag_v, clover_diag, AcceleratorWrite);
  autoView(clover_triang_v, clover_triang, AcceleratorWrite);

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
            if(i == j)
              clover_diag_v[ss]()(block)(i) = clover_full_v[ss]()(s_row, s_col)(c_row, c_col);
            else if(i < j)
              clover_triang_v[ss]()(block)(index_triang(i, j)) = clover_full_v[ss]()(s_row, s_col)(c_row, c_col);
            else
              continue;
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

  static_assert(Nd == 4 && Nc == 3 && Ns == 4 && Impl::Dimension == 3, "Wrong dimensions");
  INHERIT_IMPL_TYPES(Impl);

  // need to use different types on CPU and GPU :/
  template<typename vtype>
  using iImplCloverDiagonal = iScalar<iVector<iVector<vtype, 6>, 2>>; // TODO: real numbers
  template<typename vtype>
  using iImplCloverTriangle = iScalar<iVector<iVector<vtype, 15>, 2>>;
  template<typename vtype>
  using iImplCloverReduced = iScalar<iVector<iVector<vtype, 21>, 2>>;

  typedef iImplCloverDiagonal<Simd> SiteCloverDiagonal;
  typedef iImplCloverTriangle<Simd> SiteCloverTriangle;
  typedef iImplCloverReduced<Simd>  SiteCloverReduced;

  typedef Lattice<SiteCloverDiagonal> CloverDiagonalField;
  typedef Lattice<SiteCloverTriangle> CloverTriangleField;
  typedef Lattice<SiteCloverReduced> CloverReducedField;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:

  CloverTermFast(WilsonCloverFermion<Impl>& clover_full) // initialize both for now, need to #ifdef in the final product
    : clov_red(clover_full.GaugeGrid())
    , clov_diag(clover_full.GaugeGrid())
    , clov_triang(clover_full.GaugeGrid())
  {
    convert_clover(clover_full.CloverTerm, clov_red);
    convert_clover(clover_full.CloverTerm, clov_diag, clov_triang);
  }

  void Mooee(const FermionField& in, FermionField& out) {
    // TODO: select final implementations
// #if defined(GRID_CUDA)||defined(GRID_HIP)
//     Mooee_gpu(in, out);
// #else
//     Mooee_cpu(in, out);
// #endif
  }

  template<typename vCoeff_t> accelerator_inline vCoeff_t
  red_elem(const iImplCloverReduced<vCoeff_t>& red, int block, int i, int j) {
    if (i <= j) {
      return red()(block)(index_red(i, j));
    } else {
      return conjugate(red()(block)(index_red(i, j)));
    }
  }

  template<typename vCoeff_t> accelerator_inline vCoeff_t
  triang_elem(const iImplCloverTriangle<vCoeff_t>& triang, int block, int i, int j) {
    assert(i != j);
    if (i < j) {
      return triang()(block)(index_triang(i, j));
    } else { // i > j
      return conjugate(triang()(block)(index_triang(i, j)));
    }
  }

  // NOTE: This was an idea, might need somewhen, leaving here
  #define unrolled_for( i, num, ... ) DO_PRAGMA(unroll) for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ;

  void Mooee_original_nosplit_nostream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_red.Grid(), in.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_red_v, clov_red, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_red.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res = Zero();
      calcSpinor in_t = in_v(ss);
      auto clov_red_t = clov_red_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = clov_red_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci) + red_elem(clov_red_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }

  void Mooee_original_withsplit_nostream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_diag.Grid(), in.Grid());
    conformable(clov_diag.Grid(), clov_triang.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_diag_v, clov_diag, AcceleratorRead);
    autoView(clov_triang_v, clov_triang, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_diag.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res = Zero();
      calcSpinor in_t = in_v(ss);
      auto clov_diag_t = clov_diag_v(ss);
      auto clov_triang_t = clov_triang_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = clov_diag_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci)+ triang_elem(clov_triang_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }

  void Mooee_handunrolled_nosplit_nostream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_red.Grid(), in.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_red_v, clov_red, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_red.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res;
      calcSpinor in_t = in_v(ss);
      auto clov_red_t = clov_red_v(ss);

      // upper half
      res()(0)(0) =           clov_red_t()(0)( 0)  * in_t()(0)(0)
                  +           clov_red_t()(0)( 6)  * in_t()(0)(1)
                  +           clov_red_t()(0)( 7)  * in_t()(0)(2)
                  +           clov_red_t()(0)( 8)  * in_t()(1)(0)
                  +           clov_red_t()(0)( 9)  * in_t()(1)(1)
                  +           clov_red_t()(0)(10)  * in_t()(1)(2);

      res()(0)(1) = conjugate(clov_red_t()(0)( 6)) * in_t()(0)(0)
                  +           clov_red_t()(0)( 1)  * in_t()(0)(1)
                  +           clov_red_t()(0)(11)  * in_t()(0)(2)
                  +           clov_red_t()(0)(12)  * in_t()(1)(0)
                  +           clov_red_t()(0)(13)  * in_t()(1)(1)
                  +           clov_red_t()(0)(14)  * in_t()(1)(2);

      res()(0)(2) = conjugate(clov_red_t()(0)( 7)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(11)) * in_t()(0)(1)
                  +           clov_red_t()(0)( 2)  * in_t()(0)(2)
                  +           clov_red_t()(0)(15)  * in_t()(1)(0)
                  +           clov_red_t()(0)(16)  * in_t()(1)(1)
                  +           clov_red_t()(0)(17)  * in_t()(1)(2);

      res()(1)(0) = conjugate(clov_red_t()(0)( 8)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(12)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(15)) * in_t()(0)(2)
                  +           clov_red_t()(0)( 3)  * in_t()(1)(0)
                  +           clov_red_t()(0)(18)  * in_t()(1)(1)
                  +           clov_red_t()(0)(19)  * in_t()(1)(2);

      res()(1)(1) = conjugate(clov_red_t()(0)( 9)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(13)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(16)) * in_t()(0)(2)
                  + conjugate(clov_red_t()(0)(18)) * in_t()(1)(0)
                  +           clov_red_t()(0)( 4)  * in_t()(1)(1)
                  +           clov_red_t()(0)(20)  * in_t()(1)(2);

      res()(1)(2) = conjugate(clov_red_t()(0)(10)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(14)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(17)) * in_t()(0)(2)
                  + conjugate(clov_red_t()(0)(19)) * in_t()(1)(0)
                  + conjugate(clov_red_t()(0)(20)) * in_t()(1)(1)
                  +           clov_red_t()(0)( 5)  * in_t()(1)(2);

      // lower half
      res()(2)(0) =           clov_red_t()(1)( 0)  * in_t()(2)(0)
                  +           clov_red_t()(1)( 6)  * in_t()(2)(1)
                  +           clov_red_t()(1)( 7)  * in_t()(2)(2)
                  +           clov_red_t()(1)( 8)  * in_t()(3)(0)
                  +           clov_red_t()(1)( 9)  * in_t()(3)(1)
                  +           clov_red_t()(1)(10)  * in_t()(3)(2);

      res()(2)(1) = conjugate(clov_red_t()(1)( 6)) * in_t()(2)(0)
                  +           clov_red_t()(1)( 1)  * in_t()(2)(1)
                  +           clov_red_t()(1)(11)  * in_t()(2)(2)
                  +           clov_red_t()(1)(12)  * in_t()(3)(0)
                  +           clov_red_t()(1)(13)  * in_t()(3)(1)
                  +           clov_red_t()(1)(14)  * in_t()(3)(2);

      res()(2)(2) = conjugate(clov_red_t()(1)( 7)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(11)) * in_t()(2)(1)
                  +           clov_red_t()(1)( 2)  * in_t()(2)(2)
                  +           clov_red_t()(1)(15)  * in_t()(3)(0)
                  +           clov_red_t()(1)(16)  * in_t()(3)(1)
                  +           clov_red_t()(1)(17)  * in_t()(3)(2);

      res()(3)(0) = conjugate(clov_red_t()(1)( 8)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(12)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(15)) * in_t()(2)(2)
                  +           clov_red_t()(1)( 3)  * in_t()(3)(0)
                  +           clov_red_t()(1)(18)  * in_t()(3)(1)
                  +           clov_red_t()(1)(19)  * in_t()(3)(2);

      res()(3)(1) = conjugate(clov_red_t()(1)( 9)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(13)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(16)) * in_t()(2)(2)
                  + conjugate(clov_red_t()(1)(18)) * in_t()(3)(0)
                  +           clov_red_t()(1)( 4)  * in_t()(3)(1)
                  +           clov_red_t()(1)(20)  * in_t()(3)(2);

      res()(3)(2) = conjugate(clov_red_t()(1)(10)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(14)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(17)) * in_t()(2)(2)
                  + conjugate(clov_red_t()(1)(19)) * in_t()(3)(0)
                  + conjugate(clov_red_t()(1)(20)) * in_t()(3)(1)
                  +           clov_red_t()(1)( 5)  * in_t()(3)(2);
      coalescedWrite(out_v[ss], res);
    });
  }

  void Mooee_handunrolled_withsplit_nostream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_diag.Grid(), in.Grid());
    conformable(clov_diag.Grid(), clov_triang.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_diag_v, clov_diag, AcceleratorRead);
    autoView(clov_triang_v, clov_triang, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_diag.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res;
      calcSpinor in_t = in_v(ss);
      auto clov_diag_t = clov_diag_v(ss);
      auto clov_triang_t = clov_triang_v(ss);

      // upper half
      res()(0)(0) =             clov_diag_t()(0)( 0)  * in_t()(0)(0)
                  +           clov_triang_t()(0)( 0)  * in_t()(0)(1)
                  +           clov_triang_t()(0)( 1)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 2)  * in_t()(1)(0)
                  +           clov_triang_t()(0)( 3)  * in_t()(1)(1)
                  +           clov_triang_t()(0)( 4)  * in_t()(1)(2);

      res()(0)(1) = conjugate(clov_triang_t()(0)( 0)) * in_t()(0)(0)
                  +             clov_diag_t()(0)( 1)  * in_t()(0)(1)
                  +           clov_triang_t()(0)( 5)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 6)  * in_t()(1)(0)
                  +           clov_triang_t()(0)( 7)  * in_t()(1)(1)
                  +           clov_triang_t()(0)( 8)  * in_t()(1)(2);

      res()(0)(2) = conjugate(clov_triang_t()(0)( 1)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 5)) * in_t()(0)(1)
                  +             clov_diag_t()(0)( 2)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 9)  * in_t()(1)(0)
                  +           clov_triang_t()(0)(10)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(11)  * in_t()(1)(2);

      res()(1)(0) = conjugate(clov_triang_t()(0)( 2)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 6)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)( 9)) * in_t()(0)(2)
                  +             clov_diag_t()(0)( 3)  * in_t()(1)(0)
                  +           clov_triang_t()(0)(12)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(13)  * in_t()(1)(2);

      res()(1)(1) = conjugate(clov_triang_t()(0)( 3)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 7)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)(10)) * in_t()(0)(2)
                  + conjugate(clov_triang_t()(0)(12)) * in_t()(1)(0)
                  +             clov_diag_t()(0)( 4)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(14)  * in_t()(1)(2);

      res()(1)(2) = conjugate(clov_triang_t()(0)( 4)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 8)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)(11)) * in_t()(0)(2)
                  + conjugate(clov_triang_t()(0)(13)) * in_t()(1)(0)
                  + conjugate(clov_triang_t()(0)(14)) * in_t()(1)(1)
                  +             clov_diag_t()(0)( 5)  * in_t()(1)(2);

      // lower half
      res()(2)(0) =             clov_diag_t()(1)( 0)  * in_t()(2)(0)
                  +           clov_triang_t()(1)( 0)  * in_t()(2)(1)
                  +           clov_triang_t()(1)( 1)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 2)  * in_t()(3)(0)
                  +           clov_triang_t()(1)( 3)  * in_t()(3)(1)
                  +           clov_triang_t()(1)( 4)  * in_t()(3)(2);

      res()(2)(1) = conjugate(clov_triang_t()(1)( 0)) * in_t()(2)(0)
                  +             clov_diag_t()(1)( 1)  * in_t()(2)(1)
                  +           clov_triang_t()(1)( 5)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 6)  * in_t()(3)(0)
                  +           clov_triang_t()(1)( 7)  * in_t()(3)(1)
                  +           clov_triang_t()(1)( 8)  * in_t()(3)(2);

      res()(2)(2) = conjugate(clov_triang_t()(1)( 1)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 5)) * in_t()(2)(1)
                  +             clov_diag_t()(1)( 2)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 9)  * in_t()(3)(0)
                  +           clov_triang_t()(1)(10)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(11)  * in_t()(3)(2);

      res()(3)(0) = conjugate(clov_triang_t()(1)( 2)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 6)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)( 9)) * in_t()(2)(2)
                  +             clov_diag_t()(1)( 3)  * in_t()(3)(0)
                  +           clov_triang_t()(1)(12)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(13)  * in_t()(3)(2);

      res()(3)(1) = conjugate(clov_triang_t()(1)( 3)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 7)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)(10)) * in_t()(2)(2)
                  + conjugate(clov_triang_t()(1)(12)) * in_t()(3)(0)
                  +             clov_diag_t()(1)( 4)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(14)  * in_t()(3)(2);

      res()(3)(2) = conjugate(clov_triang_t()(1)( 4)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 8)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)(11)) * in_t()(2)(2)
                  + conjugate(clov_triang_t()(1)(13)) * in_t()(3)(0)
                  + conjugate(clov_triang_t()(1)(14)) * in_t()(3)(1)
                  +             clov_diag_t()(1)( 5)  * in_t()(3)(2);
      coalescedWrite(out_v[ss], res);
    });
  }

  void Mooee_original_nosplit_withstream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_red.Grid(), in.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_red_v, clov_red, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_red.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res = Zero();
      calcSpinor in_t = in_v(ss);
      auto clov_red_t = clov_red_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = clov_red_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci) + red_elem(clov_red_t, block, i, j) * in_t()(sj)(cj);
          }
        }
      }
      coalescedWriteNonTemporal(out_v[ss], res);
    });
  }

  void Mooee_original_withsplit_withstream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_diag.Grid(), in.Grid());
    conformable(clov_diag.Grid(), clov_triang.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_diag_v, clov_diag, AcceleratorRead);
    autoView(clov_triang_v, clov_triang, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_diag.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res = Zero();
      calcSpinor in_t = in_v(ss);
      auto clov_diag_t = clov_diag_v(ss);
      auto clov_triang_t = clov_triang_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = clov_diag_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci)+ triang_elem(clov_triang_t, block, i, j) * in_t()(sj)(cj);
          }
        }
      }
      coalescedWriteNonTemporal(out_v[ss], res);
    });
  }

  void Mooee_handunrolled_nosplit_withstream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_red.Grid(), in.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_red_v, clov_red, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_red.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res;
      calcSpinor in_t = in_v(ss);
      auto clov_red_t = clov_red_v(ss);

      // upper half
      res()(0)(0) =           clov_red_t()(0)( 0)  * in_t()(0)(0)
                  +           clov_red_t()(0)( 6)  * in_t()(0)(1)
                  +           clov_red_t()(0)( 7)  * in_t()(0)(2)
                  +           clov_red_t()(0)( 8)  * in_t()(1)(0)
                  +           clov_red_t()(0)( 9)  * in_t()(1)(1)
                  +           clov_red_t()(0)(10)  * in_t()(1)(2);

      res()(0)(1) = conjugate(clov_red_t()(0)( 6)) * in_t()(0)(0)
                  +           clov_red_t()(0)( 1)  * in_t()(0)(1)
                  +           clov_red_t()(0)(11)  * in_t()(0)(2)
                  +           clov_red_t()(0)(12)  * in_t()(1)(0)
                  +           clov_red_t()(0)(13)  * in_t()(1)(1)
                  +           clov_red_t()(0)(14)  * in_t()(1)(2);

      res()(0)(2) = conjugate(clov_red_t()(0)( 7)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(11)) * in_t()(0)(1)
                  +           clov_red_t()(0)( 2)  * in_t()(0)(2)
                  +           clov_red_t()(0)(15)  * in_t()(1)(0)
                  +           clov_red_t()(0)(16)  * in_t()(1)(1)
                  +           clov_red_t()(0)(17)  * in_t()(1)(2);

      res()(1)(0) = conjugate(clov_red_t()(0)( 8)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(12)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(15)) * in_t()(0)(2)
                  +           clov_red_t()(0)( 3)  * in_t()(1)(0)
                  +           clov_red_t()(0)(18)  * in_t()(1)(1)
                  +           clov_red_t()(0)(19)  * in_t()(1)(2);

      res()(1)(1) = conjugate(clov_red_t()(0)( 9)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(13)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(16)) * in_t()(0)(2)
                  + conjugate(clov_red_t()(0)(18)) * in_t()(1)(0)
                  +           clov_red_t()(0)( 4)  * in_t()(1)(1)
                  +           clov_red_t()(0)(20)  * in_t()(1)(2);

      res()(1)(2) = conjugate(clov_red_t()(0)(10)) * in_t()(0)(0)
                  + conjugate(clov_red_t()(0)(14)) * in_t()(0)(1)
                  + conjugate(clov_red_t()(0)(17)) * in_t()(0)(2)
                  + conjugate(clov_red_t()(0)(19)) * in_t()(1)(0)
                  + conjugate(clov_red_t()(0)(20)) * in_t()(1)(1)
                  +           clov_red_t()(0)( 5)  * in_t()(1)(2);

      // lower half
      res()(2)(0) =           clov_red_t()(1)( 0)  * in_t()(2)(0)
                  +           clov_red_t()(1)( 6)  * in_t()(2)(1)
                  +           clov_red_t()(1)( 7)  * in_t()(2)(2)
                  +           clov_red_t()(1)( 8)  * in_t()(3)(0)
                  +           clov_red_t()(1)( 9)  * in_t()(3)(1)
                  +           clov_red_t()(1)(10)  * in_t()(3)(2);

      res()(2)(1) = conjugate(clov_red_t()(1)( 6)) * in_t()(2)(0)
                  +           clov_red_t()(1)( 1)  * in_t()(2)(1)
                  +           clov_red_t()(1)(11)  * in_t()(2)(2)
                  +           clov_red_t()(1)(12)  * in_t()(3)(0)
                  +           clov_red_t()(1)(13)  * in_t()(3)(1)
                  +           clov_red_t()(1)(14)  * in_t()(3)(2);

      res()(2)(2) = conjugate(clov_red_t()(1)( 7)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(11)) * in_t()(2)(1)
                  +           clov_red_t()(1)( 2)  * in_t()(2)(2)
                  +           clov_red_t()(1)(15)  * in_t()(3)(0)
                  +           clov_red_t()(1)(16)  * in_t()(3)(1)
                  +           clov_red_t()(1)(17)  * in_t()(3)(2);

      res()(3)(0) = conjugate(clov_red_t()(1)( 8)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(12)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(15)) * in_t()(2)(2)
                  +           clov_red_t()(1)( 3)  * in_t()(3)(0)
                  +           clov_red_t()(1)(18)  * in_t()(3)(1)
                  +           clov_red_t()(1)(19)  * in_t()(3)(2);

      res()(3)(1) = conjugate(clov_red_t()(1)( 9)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(13)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(16)) * in_t()(2)(2)
                  + conjugate(clov_red_t()(1)(18)) * in_t()(3)(0)
                  +           clov_red_t()(1)( 4)  * in_t()(3)(1)
                  +           clov_red_t()(1)(20)  * in_t()(3)(2);

      res()(3)(2) = conjugate(clov_red_t()(1)(10)) * in_t()(2)(0)
                  + conjugate(clov_red_t()(1)(14)) * in_t()(2)(1)
                  + conjugate(clov_red_t()(1)(17)) * in_t()(2)(2)
                  + conjugate(clov_red_t()(1)(19)) * in_t()(3)(0)
                  + conjugate(clov_red_t()(1)(20)) * in_t()(3)(1)
                  +           clov_red_t()(1)( 5)  * in_t()(3)(2);
      coalescedWriteNonTemporal(out_v[ss], res);
    });
  }

  void Mooee_handunrolled_withsplit_withstream(const FermionField& in, FermionField& out) {
    conformable(in.Grid(), out.Grid());
    conformable(clov_diag.Grid(), in.Grid());
    conformable(clov_diag.Grid(), clov_triang.Grid());
    out.Checkerboard() = in.Checkerboard();
    autoView(clov_diag_v, clov_diag, AcceleratorRead);
    autoView(clov_triang_v, clov_triang, AcceleratorRead);
    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    typedef decltype(coalescedRead(out_v[0])) calcSpinor;
    const uint64_t Nsite = clov_diag.Grid()->oSites();
    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      calcSpinor res;
      calcSpinor in_t = in_v(ss);
      auto clov_diag_t = clov_diag_v(ss);
      auto clov_triang_t = clov_triang_v(ss);

      // upper half
      res()(0)(0) =             clov_diag_t()(0)( 0)  * in_t()(0)(0)
                  +           clov_triang_t()(0)( 0)  * in_t()(0)(1)
                  +           clov_triang_t()(0)( 1)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 2)  * in_t()(1)(0)
                  +           clov_triang_t()(0)( 3)  * in_t()(1)(1)
                  +           clov_triang_t()(0)( 4)  * in_t()(1)(2);

      res()(0)(1) = conjugate(clov_triang_t()(0)( 0)) * in_t()(0)(0)
                  +             clov_diag_t()(0)( 1)  * in_t()(0)(1)
                  +           clov_triang_t()(0)( 5)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 6)  * in_t()(1)(0)
                  +           clov_triang_t()(0)( 7)  * in_t()(1)(1)
                  +           clov_triang_t()(0)( 8)  * in_t()(1)(2);

      res()(0)(2) = conjugate(clov_triang_t()(0)( 1)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 5)) * in_t()(0)(1)
                  +             clov_diag_t()(0)( 2)  * in_t()(0)(2)
                  +           clov_triang_t()(0)( 9)  * in_t()(1)(0)
                  +           clov_triang_t()(0)(10)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(11)  * in_t()(1)(2);

      res()(1)(0) = conjugate(clov_triang_t()(0)( 2)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 6)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)( 9)) * in_t()(0)(2)
                  +             clov_diag_t()(0)( 3)  * in_t()(1)(0)
                  +           clov_triang_t()(0)(12)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(13)  * in_t()(1)(2);

      res()(1)(1) = conjugate(clov_triang_t()(0)( 3)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 7)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)(10)) * in_t()(0)(2)
                  + conjugate(clov_triang_t()(0)(12)) * in_t()(1)(0)
                  +             clov_diag_t()(0)( 4)  * in_t()(1)(1)
                  +           clov_triang_t()(0)(14)  * in_t()(1)(2);

      res()(1)(2) = conjugate(clov_triang_t()(0)( 4)) * in_t()(0)(0)
                  + conjugate(clov_triang_t()(0)( 8)) * in_t()(0)(1)
                  + conjugate(clov_triang_t()(0)(11)) * in_t()(0)(2)
                  + conjugate(clov_triang_t()(0)(13)) * in_t()(1)(0)
                  + conjugate(clov_triang_t()(0)(14)) * in_t()(1)(1)
                  +             clov_diag_t()(0)( 5)  * in_t()(1)(2);

      // lower half
      res()(2)(0) =             clov_diag_t()(1)( 0)  * in_t()(2)(0)
                  +           clov_triang_t()(1)( 0)  * in_t()(2)(1)
                  +           clov_triang_t()(1)( 1)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 2)  * in_t()(3)(0)
                  +           clov_triang_t()(1)( 3)  * in_t()(3)(1)
                  +           clov_triang_t()(1)( 4)  * in_t()(3)(2);

      res()(2)(1) = conjugate(clov_triang_t()(1)( 0)) * in_t()(2)(0)
                  +             clov_diag_t()(1)( 1)  * in_t()(2)(1)
                  +           clov_triang_t()(1)( 5)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 6)  * in_t()(3)(0)
                  +           clov_triang_t()(1)( 7)  * in_t()(3)(1)
                  +           clov_triang_t()(1)( 8)  * in_t()(3)(2);

      res()(2)(2) = conjugate(clov_triang_t()(1)( 1)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 5)) * in_t()(2)(1)
                  +             clov_diag_t()(1)( 2)  * in_t()(2)(2)
                  +           clov_triang_t()(1)( 9)  * in_t()(3)(0)
                  +           clov_triang_t()(1)(10)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(11)  * in_t()(3)(2);

      res()(3)(0) = conjugate(clov_triang_t()(1)( 2)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 6)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)( 9)) * in_t()(2)(2)
                  +             clov_diag_t()(1)( 3)  * in_t()(3)(0)
                  +           clov_triang_t()(1)(12)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(13)  * in_t()(3)(2);

      res()(3)(1) = conjugate(clov_triang_t()(1)( 3)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 7)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)(10)) * in_t()(2)(2)
                  + conjugate(clov_triang_t()(1)(12)) * in_t()(3)(0)
                  +             clov_diag_t()(1)( 4)  * in_t()(3)(1)
                  +           clov_triang_t()(1)(14)  * in_t()(3)(2);

      res()(3)(2) = conjugate(clov_triang_t()(1)( 4)) * in_t()(2)(0)
                  + conjugate(clov_triang_t()(1)( 8)) * in_t()(2)(1)
                  + conjugate(clov_triang_t()(1)(11)) * in_t()(2)(2)
                  + conjugate(clov_triang_t()(1)(13)) * in_t()(3)(0)
                  + conjugate(clov_triang_t()(1)(14)) * in_t()(3)(1)
                  +             clov_diag_t()(1)( 5)  * in_t()(3)(2);
      coalescedWriteNonTemporal(out_v[ss], res);
    });
  }

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:

  CloverDiagonalField clov_diag;
  CloverTriangleField clov_triang;
  CloverReducedField clov_red;
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

  // setup fermion operators
  double t_a = usecond();
  WilsonCloverOperator Dwc(Umu, *UGrid, *UrbGrid, 0.5, 1.0, 1.0, anisParams, implParams);
  double t_b = usecond();
  CloverTermFast<WImpl> Dwc_fast(Dwc);
  double t_c = usecond();
  grid_printf("Clover term setup times: reference %f s, improved %f s\n", (t_b-t_a)/1e6, (t_c-t_b)/1e6);

  // misc stuff needed for benchmarks
  const int nIter = readFromCommandLineInt(argc, argv, "--niter", 1000);
  double volume=1.0; for(int mu=0; mu<Nd; mu++) volume*=UGrid->_fdimensions[mu];

  // warmup + measure dhop
  grid_printf("hop warmup %s\n", precision.c_str()); fflush(stdout);
  for(auto n : {1, 2, 3, 4, 5}) Dwc.Dhop(src, hop, 0);
  grid_printf("hop measurement %s\n", precision.c_str()); fflush(stdout);
  double t0 = usecond();
  for(int n = 0; n < nIter; n++) {
#ifdef CUDA_PROFILE
    if(n == 10) cudaProfilerStart();
#endif
    Dwc.Dhop(src, hop, 0);
#ifdef CUDA_PROFILE
    if(n == 20) cudaProfilerStop();
#endif
  }
  double t1 = usecond();
  double secs_hop = (t1-t0)/1e6;

  // warmup + measure reference clover
  grid_printf("reference warmup %s\n", precision.c_str()); fflush(stdout);
  for(auto n : {1, 2, 3, 4, 5}) Dwc.Mooee(src, ref);
  grid_printf("reference measurement %s\n", precision.c_str()); fflush(stdout);
  double t2 = usecond();
  for(int n = 0; n < nIter; n++) Dwc.Mooee(src, ref);
  double t3 = usecond();
  double secs_ref = (t3-t2)/1e6;

  # if 0
  { // warmup + measure improved clover
  grid_printf("improved warmup %s\n", precision.c_str()); fflush(stdout);
  for(auto n : {1, 2, 3, 4, 5}) Dwc_fast.Mooee(src, res);
  grid_printf("improved measurement %s\n", precision.c_str()); fflush(stdout);
  double t0 = usecond();
  for(int n = 0; n < nIter; n++) Dwc_fast.Mooee(src, res);
  double t1 = usecond();
  assert(resultsAgree(ref, res));
  }
  double secs_res = (t1-t0)/1e6;
  # endif

#if (defined(GRID_CUDA)||defined(GRID_HIP))&&defined(CUDA_PROFILE)
#define BENCH_CLOVER_VERSION(VERSION)\
  double secs_##VERSION;\
  {\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) Dwc_fast.Mooee_##VERSION(src, res);\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) {\
      if(n == 10) cudaProfilerStart();\
      Dwc_fast.Mooee_##VERSION(src, res);\
      if(n == 20) cudaProfilerStop();\
    }\
    double t1 = usecond();\
    secs_##VERSION = (t1-t0)/1e6;\
  }
#elif (defined(GRID_CUDA)||defined(GRID_HIP))&&!defined(CUDA_PROFILE)
#define BENCH_CLOVER_VERSION(VERSION)\
  double secs_##VERSION;\
  {\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) Dwc_fast.Mooee_##VERSION(src, res);\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) Dwc_fast.Mooee_##VERSION(src, res);\
    double t1 = usecond();\
    secs_##VERSION = (t1-t0)/1e6;\
  }
#else
#define BENCH_CLOVER_VERSION(VERSION)\
  double secs_##VERSION;\
  {\
    grid_printf("warmup %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    for(auto n : {1, 2, 3, 4, 5}) Dwc_fast.Mooee_##VERSION(src, res);\
    grid_printf("measurement %s %s\n", #VERSION, precision.c_str()); fflush(stdout);\
    double t0 = usecond();\
    for(int n = 0; n < nIter; n++) Dwc_fast.Mooee_##VERSION(src, res);\
    double t1 = usecond();\
    secs_##VERSION = (t1-t0)/1e6;\
    assert(resultsAgree(ref, res, #VERSION)); \
  }
#endif

  BENCH_CLOVER_VERSION(original_nosplit_nostream);
  BENCH_CLOVER_VERSION(original_nosplit_withstream);
  BENCH_CLOVER_VERSION(original_withsplit_nostream);
  BENCH_CLOVER_VERSION(original_withsplit_withstream);
  BENCH_CLOVER_VERSION(handunrolled_nosplit_nostream);
  BENCH_CLOVER_VERSION(handunrolled_nosplit_withstream);
  BENCH_CLOVER_VERSION(handunrolled_withsplit_nostream);
  BENCH_CLOVER_VERSION(handunrolled_withsplit_withstream);

#undef BENCH_CLOVER_VERSION

  // performance per site (use minimal values necessary)
  double hop_flop_per_site = 1320; // Rich's Talk + what Peter uses
  double hop_byte_per_site = (8 * 9 + 9 * 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_flop_per_site = 504; // Rich's Talk and 1412.2629
  double clov_byte_per_site = (2 * 18 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;
  double clov_byte_per_site_performed = (12 * 12 + 12 + 12) * 2 * getPrecision<vCoeff_t>::value * 4;

  // total performance numbers
  double hop_gflop_total = volume * nIter * hop_flop_per_site / 1e9;
  double hop_gbyte_total = volume * nIter * hop_byte_per_site / 1e9;
  double clov_gflop_total = volume * nIter * clov_flop_per_site / 1e9;
  double clov_gbyte_total = volume * nIter * clov_byte_per_site / 1e9;
  double clov_gbyte_performed_total = volume * nIter * clov_byte_per_site_performed / 1e9;

#define PRINT_CLOVER_VERSION(VERSION)\
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",\
              #VERSION, precision.c_str(), secs_##VERSION, clov_gflop_total/secs_##VERSION, clov_gbyte_total/secs_##VERSION, secs_ref/secs_##VERSION, secs_##VERSION/secs_hop)

  // output
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "hop", precision.c_str(), secs_hop, hop_gflop_total/secs_hop, hop_gbyte_total/secs_hop, secs_ref/secs_hop, secs_hop/secs_hop);
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "reference", precision.c_str(), secs_ref, clov_gflop_total/secs_ref, clov_gbyte_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop);

  PRINT_CLOVER_VERSION(original_nosplit_nostream);
  PRINT_CLOVER_VERSION(original_nosplit_withstream);
  PRINT_CLOVER_VERSION(original_withsplit_nostream);
  PRINT_CLOVER_VERSION(original_withsplit_withstream);
  PRINT_CLOVER_VERSION(handunrolled_nosplit_nostream);
  PRINT_CLOVER_VERSION(handunrolled_nosplit_withstream);
  PRINT_CLOVER_VERSION(handunrolled_withsplit_nostream);
  PRINT_CLOVER_VERSION(handunrolled_withsplit_withstream);

  // just so we see how well the ET performs in terms of traffic
  grid_printf("Performance(%35s, %s): %2.4f s, %6.0f GFlop/s, %6.0f GByte/s, speedup vs ref = %.2f, fraction of hop = %.2f\n",
              "reference_performed", precision.c_str(), secs_ref, clov_gflop_total/secs_ref, clov_gbyte_performed_total/secs_ref, secs_ref/secs_ref, secs_ref/secs_hop);

  grid_printf("finalize %s\n", precision.c_str()); fflush(stdout);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  runBenchmark<vComplexD>(&argc, &argv);
  runBenchmark<vComplexF>(&argc, &argv);

  Grid_finalize();
}
