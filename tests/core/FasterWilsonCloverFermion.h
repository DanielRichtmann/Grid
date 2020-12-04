/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/FasterWilsonCloverFermion.h

    Copyright (C) 2020

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

#pragma once

#include <Grid/Grid.h>

// see Grid/qcd/action/fermion/WilsonCloverFermion for description

// Modifications done here:
//
// Grid: clover term = 12x12 matrix per site
//
// But: Only two diagonal 6x6 hermitian blocks are non-zero (also true in Grid, verified by running)
// Sufficient to store/transfer only the real parts of the diagonal and one triangular part
// 2 * (6 + 15 * 2) = 72 real or 36 complex words to be stored/transfered
//
// Here: Above but diagonal as complex numbers, i.e., need to store/transfer
// 2 * (6 * 2 + 15 * 2) = 84 real or 42 complex words
//
// Words per site and improvement compared to Grid (combined with the input and output spinors:
//
// - Grid:    2*12 + 12*12 = 168 words -> 1.00 x less
// - Minimal: 2*12 + 36    =  60 words -> 2.80 x less
// - Here:    2*12 + 42    =  66 words -> 2.55 x less
//
// These improvements directly translate to wall-clock time

NAMESPACE_BEGIN(Grid);

template<class Impl>
class FasterWilsonCloverFermion : public WilsonCloverFermion<Impl> {
  /////////////////////////////////////////////
  // Sizes
  /////////////////////////////////////////////

public:

  static constexpr int Nred      = Nc * Nhs;        // 6
  static constexpr int Nblock    = Nhs;             // 2
  static constexpr int Ndiag     = Nred;            // 6
  static constexpr int Ntriangle = (Nred - 1) * Nc; // 15

  /////////////////////////////////////////////
  // Type definitions
  /////////////////////////////////////////////

public:

  static_assert(Nd == 4 && Nc == 3 && Ns == 4 && Impl::Dimension == 3, "Wrong dimensions");
  INHERIT_IMPL_TYPES(Impl);

  template<typename vtype> using iImplCloverDiagonal = iScalar<iVector<iVector<vtype, Ndiag>,     Nblock>>;
  template<typename vtype> using iImplCloverTriangle = iScalar<iVector<iVector<vtype, Ntriangle>, Nblock>>;

  typedef iImplCloverDiagonal<Simd> SiteCloverDiagonal;
  typedef iImplCloverTriangle<Simd> SiteCloverTriangle;

  typedef Lattice<SiteCloverDiagonal> CloverDiagonalField;
  typedef Lattice<SiteCloverTriangle> CloverTriangleField;

  typedef WilsonCloverFermion<Impl> WilsonCloverBase;
  typedef typename WilsonCloverBase::SiteCloverType SiteClover;
  typedef typename WilsonCloverBase::CloverFieldType CloverField;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:

  FasterWilsonCloverFermion(GaugeField& _Umu,
                            GridCartesian& Fgrid,
                            GridRedBlackCartesian& Hgrid,
                            const RealD _mass,
                            const RealD _csw_r = 0.0,
                            const RealD _csw_t = 0.0,
                            const WilsonAnisotropyCoefficients& clover_anisotropy = WilsonAnisotropyCoefficients(),
                            const ImplParams& impl_p = ImplParams())
    : WilsonCloverFermion<Impl>(_Umu, Fgrid, Hgrid, _mass, _csw_r, _csw_t, clover_anisotropy, impl_p)
    , Diag(&Fgrid),           Triangle(&Fgrid)
    , DiagEven(&Hgrid),       TriangleEven(&Hgrid)
    , DiagOdd(&Hgrid),        TriangleOdd(&Hgrid)
    , DiagInv(&Fgrid),        TriangleInv(&Fgrid)
    , DiagInvEven(&Hgrid),    TriangleInvEven(&Hgrid)
    , DiagInvOdd(&Hgrid),     TriangleInvOdd(&Hgrid)
    , DiagDag(&Fgrid),        TriangleDag(&Fgrid)
    , DiagDagEven(&Hgrid),    TriangleDagEven(&Hgrid)
    , DiagDagOdd(&Hgrid),     TriangleDagOdd(&Hgrid)
    , DiagInvDag(&Fgrid),     TriangleInvDag(&Fgrid)
    , DiagInvDagEven(&Hgrid), TriangleInvDagEven(&Hgrid)
    , DiagInvDagOdd(&Hgrid),  TriangleInvDagOdd(&Hgrid)
  {
    double t0 = usecond();
    convertLayout(this->CloverTerm, Diag, Triangle);
    convertLayout(this->CloverTermEven, DiagEven, TriangleEven);
    convertLayout(this->CloverTermOdd, DiagOdd, TriangleOdd);

    convertLayout(this->CloverTermInv, DiagInv, TriangleInv);
    convertLayout(this->CloverTermInvEven, DiagInvEven, TriangleInvEven);
    convertLayout(this->CloverTermInvOdd, DiagInvOdd, TriangleInvOdd);

    // TODO: optimally would like to set original fields to zero
    // BUT:  not possible looking at lattice class
    double t1 = usecond();
    std::cout << GridLogMessage << "FasterWilsonCloverFermion: layout conversions took " << (t1-t0)/1e6 << " seconds" << std::endl;
  }


  void convertLayout(const CloverField& full, CloverDiagonalField& diag, CloverTriangleField& triangle) {
    conformable(full.Grid(), diag.Grid());
    conformable(full.Grid(), triangle.Grid());

    diag.Checkerboard()     = full.Checkerboard();
    triangle.Checkerboard() = full.Checkerboard();

    autoView(full_v, full, AcceleratorRead);
    autoView(diag_v, diag, AcceleratorWrite);
    autoView(triangle_v, triangle, AcceleratorWrite);

    // NOTE: this function cannot be 'private' since nvcc forbids this for kernels
    accelerator_for(ss, full.Grid()->oSites(), 1, {
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
                diag_v[ss]()(block)(i) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else if(i < j)
                triangle_v[ss]()(block)(triangle_index(i, j)) = full_v[ss]()(s_row, s_col)(c_row, c_col);
              else
                continue;
            }
          }
        }
      }
    });
  }


  void MooeeInternal(const FermionField& in, FermionField& out, int dag, int inv) override {
    conformable(in.Grid(), out.Grid());

    assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);
    out.Checkerboard() = in.Checkerboard();

    CloverDiagonalField* diag;
    CloverTriangleField* triangle;

    // decision making from original implementation
    if(dag) {
      if(in.Grid()->_isCheckerBoarded) {
        if(in.Checkerboard() == Odd) {
          diag     = (inv) ? &DiagInvDagOdd :     &DiagDagOdd;
          triangle = (inv) ? &TriangleInvDagOdd : &TriangleDagOdd;
        } else {
          diag     = (inv) ? &DiagInvDagEven :     &DiagDagEven;
          triangle = (inv) ? &TriangleInvDagEven : &TriangleDagEven;
        }
      } else {
        diag     = (inv) ? &DiagInvDag :     &DiagDag;
        triangle = (inv) ? &TriangleInvDag : &TriangleDag;
      }
    } else {
      if(in.Grid()->_isCheckerBoarded) {
        if(in.Checkerboard() == Odd) {
          diag     = (inv) ? &DiagInvOdd :     &DiagOdd;
          triangle = (inv) ? &TriangleInvOdd : &TriangleOdd;
        } else {
          diag     = (inv) ? &DiagInvEven :     &DiagEven;
          triangle = (inv) ? &TriangleInvEven : &TriangleEven;
        }
      } else {
        diag     = (inv) ? &DiagInv :     &Diag;
        triangle = (inv) ? &TriangleInv : &Triangle;
      }
    }

    conformable(diag->Grid(), in.Grid());
    conformable(triangle->Grid(), in.Grid());

    MooeeKernel(in, out, *diag, *triangle);
  }


  void MooeeKernel(const FermionField&        in,
                   FermionField&              out,
                   const CloverDiagonalField& diag,
                   const CloverTriangleField& triangle) {
    // #if defined(GRID_CUDA) || defined(GRID_HIP)
    MooeeKernel_gpu(in, out, diag, triangle);
    // #else
    //     MooeeKernel_cpu(in, out, diag, triangle);
    // #endif
  }


  void MooeeKernel_gpu(const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diag,
                       const CloverTriangleField& triangle) {
    autoView(diag_v,     diag,     AcceleratorRead);
    autoView(triangle_v, triangle, AcceleratorRead);
    autoView(in_v,       in,       AcceleratorRead);
    autoView(out_v,      out,      AcceleratorWrite);

    typedef decltype(coalescedRead(out_v[0])) CalcSpinor;
    const uint64_t Nsite = diag.Grid()->oSites();

    accelerator_for(ss, Nsite, Simd::Nsimd(), {
      CalcSpinor res;
      CalcSpinor in_t = in_v(ss);
      auto diag_t = diag_v(ss);
      auto triangle_t = triangle_v(ss);
      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = diag_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci)+ triangle_elem(triangle_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }


  void MooeeKernel_cpu(const FermionField&        in,
                       FermionField&              out,
                       const CloverDiagonalField& diag,
                       const CloverTriangleField& triangle) {
    autoView(diag_v,     diag,     CpuRead);
    autoView(triangle_v, triangle, CpuRead);
    autoView(in_v,       in,       CpuRead);
    autoView(out_v,      out,      CpuWrite);

    typedef SiteSpinor CalcSpinor;
    const uint64_t Nsite = diag.Grid()->oSites();

    // TODO: Add CPU code
    thread_for(ss, Nsite, Simd::Nsimd(), {
      CalcSpinor res;
      CalcSpinor in_t = in_v[ss];
      auto diag_t = diag_v[ss];
      auto triangle_t = triangle_v[ss];

      for(int block=0; block<Nhs; block++) {
        int s_start = block*Nhs;
        for(int i=0; i<Nred; i++) {
          int si = s_start + i/Nc, ci = i%Nc;
          res()(si)(ci) = diag_t()(block)(i) * in_t()(si)(ci);
          for(int j=0; j<Nred; j++) {
            if (j == i) continue;
            int sj = s_start + j/Nc, cj = j%Nc;
            res()(si)(ci) = res()(si)(ci)+ triangle_elem(triangle_t, block, i, j) * in_t()(sj)(cj);
          };
        };
      };
      coalescedWrite(out_v[ss], res);
    });
  }

  /////////////////////////////////////////////
  // Helpers
  /////////////////////////////////////////////

private:

  template<typename vobj>
  accelerator_inline vobj triangle_elem(const iImplCloverTriangle<vobj>& triang, int block, int i, int j) {
    assert(i != j);
    if(i < j) {
      return triang()(block)(triangle_index(i, j));
    } else { // i > j
      return conjugate(triang()(block)(triangle_index(i, j)));
    }
  }

  accelerator_inline int triangle_index(int i, int j) {
    if(i == j)
      return 0;
    else if(i < j)
      return Nred * (Nred - 1) / 2 - (Nred - i) * (Nred - i - 1) / 2 + j - i - 1;
    else // i > j
      return Nred * (Nred - 1) / 2 - (Nred - j) * (Nred - j - 1) / 2 + i - j - 1;
  }

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:

  CloverDiagonalField Diag,       DiagEven,       DiagOdd;
  CloverDiagonalField DiagInv,    DiagInvEven,    DiagInvOdd;
  CloverDiagonalField DiagDag,    DiagDagEven,    DiagDagOdd;
  CloverDiagonalField DiagInvDag, DiagInvDagEven, DiagInvDagOdd;

  CloverTriangleField Triangle,       TriangleEven,       TriangleOdd;
  CloverTriangleField TriangleInv,    TriangleInvEven,    TriangleInvOdd;
  CloverTriangleField TriangleDag,    TriangleDagEven,    TriangleDagOdd;
  CloverTriangleField TriangleInvDag, TriangleInvDagEven, TriangleInvDagOdd;
};

NAMESPACE_END(Grid);
