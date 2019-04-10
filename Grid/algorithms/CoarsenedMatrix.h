    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/algorithms/CoarsenedMatrix.h

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>

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
#ifndef  GRID_ALGORITHM_COARSENED_MATRIX_H
#define  GRID_ALGORITHM_COARSENED_MATRIX_H



namespace Grid {

  class Geometry {
  public:
    int ndimension;
    int npoint;
    std::vector<int> directions   ;
    std::vector<int> displacements;

  Geometry(int _d)  {

      ndimension = _d;
  
      int base = Base();

      // make coarse grid stencil for 4d , not 5d
      if ( _d==5 ) _d=4;

      npoint = 2*_d+1;
      directions.resize(npoint);
      displacements.resize(npoint);
      for(int d=0;d<_d;d++){
	directions[2*d  ] = d+base;
	directions[2*d+1] = d+base;
	displacements[2*d  ] = +1;
	displacements[2*d+1] = -1;
      }
      directions   [2*_d]=0;
      displacements[2*_d]=0;
      
      //// report back
      std::cout<<GridLogMessage<<"directions    :";
      for(int d=0;d<npoint;d++) std::cout<< directions[d]<< " ";
      std::cout <<std::endl;
      std::cout<<GridLogMessage<<"displacements :";
      for(int d=0;d<npoint;d++) std::cout<< displacements[d]<< " ";
      std::cout<<std::endl;
    }
  
    int PointFromDirDisp(int dir, int disp) const {
      int _d = (npoint - 1) / 2;
      int base = Base();

      assert(disp == -1 || disp == 0 || disp == 1);
      if (ndimension == 4) assert(0 <= dir && dir < _d);
      else if (ndimension == 5) assert(0 <= dir && dir <= _d);

      if(dir == 0 and disp == 0)
        return 2*_d;
      else
        return (_d * (dir - base) + 1 - disp) / 2;
    }

    int SelfStencilPoint() const {
      return npoint - 1;
    }

  private:
    int Base() const {
      return (ndimension == 5) ? 1 : 0;
    }

    /*
      // Original cleaner code
    Geometry(int _d) : dimension(_d), npoint(2*_d+1), directions(npoint), displacements(npoint) {
      for(int d=0;d<dimension;d++){
	directions[2*d  ] = d;
	directions[2*d+1] = d;
	displacements[2*d  ] = +1;
	displacements[2*d+1] = -1;
      }
      directions   [2*dimension]=0;
      displacements[2*dimension]=0;
    }
    std::vector<int> GetDelta(int point) {
      std::vector<int> delta(dimension,0);
      delta[directions[point]] = displacements[point];
      return delta;
    };
    */    

  };
  
#define INHERIT_COARSENING_POLICY_TYPES(Policy)               \
  typedef typename Policy::SiteSpinor       SiteSpinor;       \
  typedef typename Policy::SiteLinkField    SiteLinkField;    \
  typedef typename Policy::SiteScalar       SiteScalar;       \
  typedef typename Policy::FermionField     FermionField;     \
  typedef typename Policy::LinkField        LinkField;        \
  typedef typename Policy::ScalarField      ScalarField;      \
  typedef typename Policy::FineSiteSpinor   FineSiteSpinor;   \
  typedef typename Policy::FineSiteScalar   FineSiteScalar;   \
  typedef typename Policy::FineFermionField FineFermionField; \
  typedef typename Policy::FineScalarField  FineScalarField;

#define INHERIT_COARSENING_POLICY_VARIABLES(Policy)         \
  using Policy::Ncs; \
  using Policy::Nbasis; \
  using Policy::Nfs; \
  using Policy::Nsb;

  // Grid uses the policy pattern to generalise the Dirac operators.
  // IMHO, we should do that with the multigrid-related stuff, too.
  // This way, we are able to support multiple coarsening strategies, i.e., the
  // original one Peter uses and also my one with explicit intact chirality = coarse spins.

  template<class _FineFermionField, class Simd, int nbasis>
  class OriginalCoarseningPolicy {
  public:
    /////////////////////////////////////////////
    // Static variables
    /////////////////////////////////////////////

#define SpinIndex 1 // Need to do this temporarily, will be removed once the QCD namespace is gone
    static const int Nbasis            = nbasis;
    static const int Ncs               = 1; // number of coarse spin dofs
           const int Nfs               = indexRank<SpinIndex, typename getVectorType<_FineFermionField>::type>(); // number of fine grid spin dofs
           const int Nsb               = Nfs/Ncs; // spin blocking
    static const bool isTwoSpinVersion = false;
#undef SpinIndex

    /////////////////////////////////////////////
    // Type Definitions
    /////////////////////////////////////////////

    template<typename vtype> using iImplSpinor    = iVector<iScalar<iScalar<iScalar<vtype>>>, nbasis>;
    template<typename vtype> using iImplLinkField = iMatrix<iScalar<iScalar<iScalar<vtype>>>, nbasis>;
    template<typename vtype> using iImplScalar    = iScalar<iScalar<iScalar<vtype>>>;

    typedef iImplSpinor<Simd>    SiteSpinor;
    typedef iImplLinkField<Simd> SiteLinkField;
    typedef iImplScalar<Simd>    SiteScalar;

    // Needs to be called FermionField, too as in the Dirac operators, so that we can use it for coarsening just like the Dirac operators
    typedef Lattice<SiteSpinor>    FermionField;
    typedef Lattice<SiteLinkField> LinkField;
    typedef Lattice<SiteScalar>    ScalarField;

    typedef _FineFermionField                              FineFermionField;
    typedef typename getVectorType<FineFermionField>::type FineSiteSpinor;
    typedef typename FineSiteSpinor::tensor_reduced        FineSiteScalar;
    typedef Lattice<FineSiteScalar>                        FineScalarField;

    /////////////////////////////////////////////
    // Member Functions
    /////////////////////////////////////////////

    std::string name() const { return "OriginalCoarseningPolicy"; }

    // TODO: Think about refactoring these to take site vectors instead -> fewer indices

    strong_inline void projectionKernel(FermionField & CoarseVec, FineFermionField const & BasisVec, FineFermionField const & FineVec,
                                        int sc, int sf, int i) {
      CoarseVec._odata[sc](i) = CoarseVec._odata[sc](i) + innerProduct(BasisVec._odata[sf], FineVec._odata[sf]);
    }

    strong_inline void promotionKernel(FermionField const & CoarseVec, FineFermionField const & BasisVec, FineFermionField & FineVec,
                                       int sc, int sf, int i) {
      if(i == 0)
        FineVec._odata[sf] = CoarseVec._odata[sc](i) * BasisVec._odata[sf];
      else
        FineVec._odata[sf] = FineVec._odata[sf] + CoarseVec._odata[sc](i) * BasisVec._odata[sf];
    }

    strong_inline void multLinkKernel(SiteSpinor & res, std::vector<LinkField> const & Y, SiteSpinor const & nbr, int point, int ss) {
      res = res + Y[point]._odata[ss] * nbr;
    }

    void orthogonaliseKernel(ScalarField & InnerProd, std::vector<FineFermionField> & BasisVecs) {
      blockOrthogonalise(InnerProd, BasisVecs);
    }

    strong_inline void setToOneKernel(SiteSpinor &Vec, int i) {
      SiteScalar tmp; tmp = 1.0;
      Vec(i) = tmp;
    }

    void chiralDoublingKernel(std::vector<FineFermionField> & BasisVecs) {
      assert(BasisVecs.size() % 2 == 0);
      auto nb = BasisVecs.size() / 2;

      for(int n = 0; n < nb; n++) {
        auto tmp1 = BasisVecs[n];
        auto tmp2 = tmp1;
        G5C(tmp2, BasisVecs[n]);
        axpby(BasisVecs[n], 0.5, 0.5, tmp1, tmp2);
        axpby(BasisVecs[n + nb], 0.5, -0.5, tmp1, tmp2);
        std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
                  << "norm2(vec[" << n << "]) = " << norm2(BasisVecs[n]) << ". "
                  << "norm2(vec[" << n + nb << "]) = " << norm2(BasisVecs[n + nb]) << std::endl;
      }
    }
  };


  template<class _FineFermionField, class Simd, int nbasis>
  class TwoSpinCoarseningPolicy {
  public:
    /////////////////////////////////////////////
    // Static variables
    /////////////////////////////////////////////

#define SpinIndex 1 // Need to do this temporarily, will be removed once the QCD namespace is gone
    static const int Nbasis            = nbasis;
    static const int Ncs               = 2; // number of coarse spin dofs
           const int Nfs               = indexRank<SpinIndex, typename getVectorType<_FineFermionField>::type>(); // number of fine grid spin dofs
           const int Nsb               = Nfs/Ncs; // spin blocking
    static const bool isTwoSpinVersion = true;
#undef SpinIndex

    /////////////////////////////////////////////
    // Type Definitions
    /////////////////////////////////////////////

    template<typename vtype> using iImplSpinor    = iScalar<iVector<iVector<vtype, nbasis>, Ncs>>; // Note, nbasis here is equivalent to nbasis/2 above
    template<typename vtype> using iImplLinkField = iScalar<iMatrix<iMatrix<vtype, nbasis>, Ncs>>; // Note, nbasis here is equivalent to nbasis/2 above
    template<typename vtype> using iImplScalar    = iScalar<iScalar<iScalar<vtype>>>;

    typedef iImplSpinor<Simd>    SiteSpinor;
    typedef iImplLinkField<Simd> SiteLinkField;
    typedef iImplScalar<Simd>    SiteScalar;

    // Needs to be called FermionField, too as in the Dirac operators, so that we can use it for coarsening just like the Dirac operators
    typedef Lattice<SiteSpinor>    FermionField;
    typedef Lattice<SiteLinkField> LinkField;
    typedef Lattice<SiteScalar>    ScalarField;

    typedef _FineFermionField                              FineFermionField;
    typedef typename getVectorType<FineFermionField>::type FineSiteSpinor;
    typedef typename FineSiteSpinor::tensor_reduced        FineSiteScalar;
    typedef Lattice<FineSiteScalar>                        FineScalarField;

    /////////////////////////////////////////////
    // Member Functions
    /////////////////////////////////////////////

    std::string name() const { return "TwoSpinCoarseningPolicy"; }

    // TODO: Think about refactoring these to take site vectors instead -> fewer indices

    strong_inline void projectionKernel(FermionField & CoarseVec, FineFermionField const & BasisVec, FineFermionField const & FineVec,
                                        int sc, int sf, int i) {
      for(int s = 0; s < Nfs; s++) {
        CoarseVec._odata[sc]()(s/Nsb)(i) = CoarseVec._odata[sc]()(s/Nsb)(i) +
          TensorRemove(innerProduct(BasisVec._odata[sf]()(s), FineVec._odata[sf]()(s)));
      }
    }

    strong_inline void promotionKernel(FermionField const & CoarseVec, FineFermionField const & BasisVec, FineFermionField & FineVec,
                                       int sc, int sf, int i) {
      iScalar<Simd> CoarseTmp; // -> need to add the tensor structure here, i.e., need to make the CoarseVec elem an iScalar
      if(i == 0) {
        for(int s = 0; s < Nfs; s++) {
          CoarseTmp._internal = CoarseVec._odata[sc]()(s/Nsb)(i);
          FineVec._odata[sf]()(s) = CoarseTmp * BasisVec._odata[sf]()(s);
        }
      } else {
        for(int s = 0; s < Nfs; s++) {
          CoarseTmp._internal = CoarseVec._odata[sc]()(s/Nsb)(i);
          FineVec._odata[sf]()(s) = FineVec._odata[sf]()(s) + CoarseTmp * BasisVec._odata[sf]()(s);
        }
      }
    }

    strong_inline void multLinkKernel(SiteSpinor & res, std::vector<LinkField> const & Y, SiteSpinor const & nbr, int point, int ss) {
      //   res() = res() + Y._odata[ss](point) * nbr();
      res = res + Y[point]._odata[ss] * nbr;
    }

    void orthogonaliseKernel(ScalarField &InnerProd, std::vector<FineFermionField> &BasisVecs) {
      GridBase *            _coarseGrid = InnerProd._grid;
      GridBase *            _fineGrid = BasisVecs[0]._grid;

      subdivides(_coarseGrid, _fineGrid);
      assert(Nbasis == BasisVecs.size());
      for(int i = 0; i < Nbasis; i++) {
        conformable(BasisVecs[i]._grid, _fineGrid);
      }

      CoarseningLookUpTable lookUpTable(_coarseGrid, _fineGrid);

      // Kernel fusion
      parallel_region {
        Vector<SiteScalar> alpha(Ncs, zero); // NOTE: Using Vector instead of std::vector here is of utmost importance! (This cost me hours -.-')
        Vector<SiteScalar> norm(Ncs, zero);
        parallel_for_internal(int sc = 0; sc < _coarseGrid->oSites(); sc++) {
          for(int v = 0; v < Nbasis; v++) {
            for(int u = 0; u < v; u++) {
              for(int k = 0; k < Ncs; k++) alpha[k] = zero;

              for(int sf : lookUpTable()[sc])
                for(int s = 0; s < Nfs; s++)
                  alpha[s / Nsb]()() = alpha[s / Nsb]()() + innerProduct(BasisVecs[u]._odata[sf]()(s), BasisVecs[v]._odata[sf]()(s));

              for(int sf : lookUpTable()[sc])
                for(int s = 0; s < Nfs; s++)
                  BasisVecs[v]._odata[sf]()(s) = BasisVecs[v]._odata[sf]()(s) - alpha[s / Nsb]()() * BasisVecs[u]._odata[sf]()(s);
            }

            for(int k = 0; k < Ncs; k++) norm[k] = zero;

            for(int sf : lookUpTable()[sc])
              for(int s = 0; s < Nfs; s++)
                norm[s / Nsb]()() = norm[s / Nsb]()() + innerProduct(BasisVecs[v]._odata[sf]()(s), BasisVecs[v]._odata[sf]()(s));

            for(int k = 0; k < Ncs; k++) norm[k] = pow(norm[k], -0.5);

            for(int sf : lookUpTable()[sc])
              for(int s = 0; s < Nfs; s++) BasisVecs[v]._odata[sf]()(s) = norm[s / Nsb]()() * BasisVecs[v]._odata[sf]()(s);
          }
        }
      }
    }

    strong_inline void setToOneKernel(SiteSpinor &Vec, int i) {
      for(int s = 0; s < Ncs; s++)
        Vec()(s)(i) = Simd(1.0);
    }

    void chiralDoublingKernel(std::vector<FineFermionField> &BasisVecs) {
      std::cout << GridLogMessage << "Skipping chiral doubling as it is not necessary with twoSpinCoarseningPolicy" << std::endl;
    }
  };

  template<class CoarseningPolicy>
  class AggregationUsingPolicies : public CoarseningPolicy {
  public:

    /////////////////////////////////////////////
    // Type Definitions
    /////////////////////////////////////////////

    INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
    INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);

    /////////////////////////////////////////////
    // Member Data
    /////////////////////////////////////////////

    GridBase *                    _coarseGrid;
    GridBase *                    _fineGrid;
    std::vector<FineFermionField> _subspace;
    int                           _checkerBoard;

    /////////////////////////////////////////////
    // Member Functions
    /////////////////////////////////////////////

    AggregationUsingPolicies(GridBase *CoarseGrid, GridBase *FineGrid, int CheckerBoard)
      : _coarseGrid(CoarseGrid)
      , _fineGrid(FineGrid)
      , _subspace(Nbasis, FineGrid)
      , _checkerBoard(CheckerBoard) {
      subdivides(_coarseGrid, _fineGrid);
    }

    std::vector<FineFermionField> &Subspace() {
      return _subspace;
    }

    void Orthogonalise(int passes = 2) {
      ScalarField InnerProd(_coarseGrid);
      for(int n = 0; n < passes; ++n) {
        std::cout << GridLogMessage << "Gram-Schmidt pass " << n+1 << std::endl;
        CoarseningPolicy::orthogonaliseKernel(InnerProd, _subspace);
      }
      std::cout << GridLogMessage << "Gram-Schmidt checking orthogonality" << std::endl;
      CheckOrthogonal();
    }
    void CheckOrthogonal(void) {
      FermionField iProj(_coarseGrid);
      FermionField eProj(_coarseGrid);
      CoarseningLookUpTable lookUpTable(_coarseGrid, _fineGrid);
      for(int i = 0; i < Nbasis; i++) {
        ProjectToSubspace(iProj, _subspace[i], lookUpTable);
        eProj = zero;
        parallel_for(int ss = 0; ss < _coarseGrid->oSites(); ss++) {
          CoarseningPolicy::setToOneKernel(eProj._odata[ss], i);
        }
        eProj = eProj - iProj;
        std::cout << GridLogMessage << "Orthog check error " << i << " " << norm2(eProj) << std::endl;
      }
      std::cout << GridLogMessage << "CheckOrthog done" << std::endl;
    }

    // This function overload is done in analogy to that of the original "blockProject"
    void ProjectToSubspace(FermionField &CoarseVec, const FineFermionField &FineVec) {
      std::cout << GridLogDebug << "Imlementation of " << __FUNCTION__ << " with 2 (= fewer) args called" << std::endl;
      CoarseningLookUpTable lookUpTable(_coarseGrid, _fineGrid);
      ProjectToSubspace(CoarseVec, FineVec, lookUpTable);
    }
    void ProjectToSubspace(FermionField &CoarseVec,
                           const FineFermionField &FineVec,
                           const CoarseningLookUpTable &lookUpTable) {
      std::cout << GridLogDebug << "Implementation of " << __FUNCTION__ << " with 3 (= more) args called" << std::endl;
      GridBase *fine   = FineVec._grid;
      GridBase *coarse = CoarseVec._grid;

      assert(lookUpTable.gridPointersMatch(coarse, fine));
      assert(Nbasis == _subspace.size());
      for(int i = 0; i < Nbasis; i++) {
        conformable(_subspace[i], FineVec);
      }

      CoarseVec = zero;
      // Thread over coarse sites so we can get rid of the critical region
      parallel_for(int sc = 0; sc < _coarseGrid->oSites(); sc++) {
        for(int i = 0; i < Nbasis; i++) {
          for(auto sf : lookUpTable()[sc]) {
            CoarseningPolicy::projectionKernel(CoarseVec, _subspace[i], FineVec, sc, sf, i);
          }
        }
      }
      return;
    }

    void PromoteFromSubspace(const FermionField &CoarseVec, FineFermionField &FineVec) {
      FineVec.checkerboard  = _subspace[0].checkerboard;
      int       _ndimension = _coarseGrid->_ndimension;

      // checks
      assert(Nbasis == _subspace.size());
      conformable(_coarseGrid, CoarseVec._grid);
      conformable(_fineGrid, FineVec._grid);
      for(int i = 0; i < Nbasis; i++) {
        conformable(_subspace[i], FineVec);
      }

      std::vector<int> block_r(_ndimension);

      for(int d = 0; d < _ndimension; d++) {
        block_r[d] = _fineGrid->_rdimensions[d] / _coarseGrid->_rdimensions[d];
        assert(block_r[d] * _coarseGrid->_rdimensions[d] == _fineGrid->_rdimensions[d]);
      }

      // Loop with a cache friendly loop ordering
      parallel_region {
        int              sc;
        std::vector<int> coor_c(_ndimension);
        std::vector<int> coor_f(_ndimension);

        parallel_for_internal(int sf = 0; sf < _fineGrid->oSites(); sf++) {

          Lexicographic::CoorFromIndex(coor_f, sf, _fineGrid->_rdimensions);
          for(int d = 0; d < _ndimension; d++) coor_c[d] = coor_f[d] / block_r[d];
          Lexicographic::IndexFromCoor(coor_c, sc, _coarseGrid->_rdimensions);

          for(int i = 0; i < Nbasis; i++) {
            CoarseningPolicy::promotionKernel(CoarseVec, _subspace[i], FineVec, sc, sf, i);
          }
        }
      }
      return;
    }

    void CreateSubspaceRandom(GridParallelRNG &RNG) {
      for(int i = 0; i < Nbasis; i++) {
        random(RNG, _subspace[i]);
        std::cout << GridLogMessage << " norm subspace[" << i << "] " << norm2(_subspace[i]) << std::endl;
      }
      Orthogonalise();
    }

    void CreateSubspace(GridParallelRNG &RNG, LinearOperatorBase<FineFermionField> &hermop, int nn = Nbasis) {

      RealD scale;

      ConjugateGradient<FineFermionField> CG(1.0e-2, 10000);
      FineFermionField                    noise(_fineGrid);
      FineFermionField                    Mn(_fineGrid);

      for(int b = 0; b < nn; b++) {

        _subspace[b] = zero;
        gaussian(RNG, noise);
        scale = std::pow(norm2(noise), -0.5);
        noise = noise * scale;

        hermop.Op(noise, Mn);
        std::cout << GridLogMessage << "noise   [" << b << "] <n|MdagM|n> " << norm2(Mn) << std::endl;

        for(int i = 0; i < 1; i++) {

          CG(hermop, noise, _subspace[b]);

          noise = _subspace[b];
          scale = std::pow(norm2(noise), -0.5);
          noise = noise * scale;
        }

        hermop.Op(noise, Mn);
        std::cout << GridLogMessage << "filtered[" << b << "] <f|MdagM|f> " << norm2(Mn) << std::endl;
        _subspace[b] = noise;
      }

      Orthogonalise();
    }

    void CreateSubspaceDDalphaAMG(GridParallelRNG &RNG, LinearOperatorBase<FineFermionField> &linop, bool startFromRandom, int nn = Nbasis, Integer restartLength = 20) {
      RealD            scale;
      FineFermionField tmp(_fineGrid);
      FineFermionField Mphi(_fineGrid);

      // TODO: We shouldn't fix this to GMRES here but accept a general smooter as argument
      GeneralisedMinimalResidual<FineFermionField> GMRES(1.0e-14, restartLength, restartLength, false);

      for(int b = 0; b < nn; b++) {
        if(startFromRandom)
          gaussian(RNG, _subspace[b]);

        linop.Op(_subspace[b], Mphi);
        auto normRandom = norm2(Mphi);

        // NOTE: This is the version MR actually proposes in his PHD thesis
        // tmp = zero;
        // for(int i = 0; i < 3; i++) {
        //   GMRES(linop, _subspace[b], tmp);
        // }
        // scale        = std::pow(norm2(tmp), -0.5);
        // _subspace[b] = tmp * scale;

        // NOTE: This is the version I see better results with
        for(int i = 0; i < 3; i++) {
          tmp = zero;
          GMRES(linop, _subspace[b], tmp);
          _subspace[b] = tmp;
        }
        scale        = std::pow(norm2(_subspace[b]), -0.5);
        _subspace[b] = _subspace[b] * scale;

        linop.Op(_subspace[b], Mphi);
        auto normFiltered = norm2(Mphi);
        std::cout << GridLogMessage             << "Vector [" << b << "]:"
                  << " <n|MdagM|n> random = "   << normRandom
                  << " <n|MdagM|n> filtered = " << normFiltered
                  << " reduction = "            << normFiltered / normRandom
                  << std::endl;
      }

      for(int b = nn; b < Nbasis; b++) {
        _subspace[b] = zero;
      }
    }

    void DoChiralDoubling() {
      CoarseningPolicy::chiralDoublingKernel(_subspace);
    }
  };

  template<class CoarseningPolicy>
  class CoarsenedMatrixUsingPolicies : public SparseMatrixBase<typename CoarseningPolicy::FermionField>
                                     , public CoarseningPolicy {
  public:

    /////////////////////////////////////////////
    // Type Definitions
    /////////////////////////////////////////////

    INHERIT_COARSENING_POLICY_TYPES(CoarseningPolicy);
    INHERIT_COARSENING_POLICY_VARIABLES(CoarseningPolicy);

    /////////////////////////////////////////////
    // Member Data
    /////////////////////////////////////////////

    GridBase * _grid;
    Geometry _geom;
    CartesianStencil<SiteSpinor, SiteSpinor> _stencil;
    std::vector<LinkField> _Y; // Y = Kate's notation. TODO: With the TwoSpin policy, we could also have a Lattice<...> like in the Dirac operators

    /////////////////////////////////////////////
    // Member Functions
    /////////////////////////////////////////////

    CoarsenedMatrixUsingPolicies(GridCartesian &CoarseGrid)
      : _grid(&CoarseGrid)
      , _geom(CoarseGrid._ndimension)
      , _stencil(&CoarseGrid, _geom.npoint, Even, _geom.directions, _geom.displacements)
      , _Y(_geom.npoint, &CoarseGrid)
    {
      std::cout << GridLogMessage << "Called ctor of CoarsenedMatrixUsingPolicies using Policy " << CoarseningPolicy::name() << std::endl;
    }

    GridBase *Grid(void) { return _grid; };

    RealD M(const FermionField &in, FermionField &out) {
      conformable(_grid, in._grid);
      conformable(in._grid, out._grid);

      SimpleCompressor<SiteSpinor> compressor;
      _stencil.HaloExchange(in, compressor);

      parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
        SiteSpinor    res = zero;
        SiteSpinor    nbr;
        int           ptype;
        StencilEntry *SE;
        for(int point = 0; point < _geom.npoint; point++) {
          SE = _stencil.GetEntry(ptype, point, ss);

          if(SE->_is_local && SE->_permute) {
            permute(nbr, in._odata[SE->_offset], ptype);
          } else if(SE->_is_local) {
            nbr = in._odata[SE->_offset];
          } else {
            nbr = _stencil.CommBuf()[SE->_offset];
          }
          CoarseningPolicy::multLinkKernel(res, _Y, nbr, point, ss);
        }
        vstream(out._odata[ss], res);
      }
      return norm2(out);
    }

    RealD Mdag(const FermionField &in, FermionField &out) {
      // // corresponds to Petrov-Galerkin coarsening
      // return M(in,out);

      // corresponds to Galerkin coarsening
      FermionField tmp(Grid());
      G5C(tmp, in);
      M(tmp, out);
      G5C(out, out);
      return norm2(out);
    }

    void Mdiag(const FermionField &in, FermionField &out){
      // use the self-coupling point of the stencil
      auto p    = _geom.SelfStencilPoint();
      auto dir  = _geom.directions[p];
      auto disp = _geom.displacements[p];
      Mdir(in, out, dir, disp);
    }

    void Mdir(const FermionField &in, FermionField &out, int dir, int disp) {
      conformable(Grid(), in._grid);
      conformable(in._grid, out._grid);

      SimpleCompressor<SiteSpinor> compressor;
      _stencil.HaloExchange(in, compressor);

      auto point = _geom.PointFromDirDisp(dir, disp);

      parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
        SiteSpinor    res = zero;
        SiteSpinor    nbr;
        int           ptype;
        StencilEntry *SE;

        SE = _stencil.GetEntry(ptype, point, ss);

        if(SE->_is_local && SE->_permute) {
          permute(nbr, in._odata[SE->_offset], ptype);
        } else if(SE->_is_local) {
          nbr = in._odata[SE->_offset];
        } else {
          nbr = _stencil.CommBuf()[SE->_offset];
        }

        CoarseningPolicy::multLinkKernel(res, _Y, nbr, point, ss);

        vstream(out._odata[ss], res);
      }
    }

    // NOTE: Do this temporarily as I can't put everything into the Policies (since it can't have Aggregation as a parameter)
    template<bool isTwoSpinVersion, typename std::enable_if<isTwoSpinVersion == false>::type * = nullptr>
    void doOperatorCoarsening(GridBase *FineGrid, LinearOperatorBase<FineFermionField> &linop, AggregationUsingPolicies<CoarseningPolicy> &Aggregates) {
      std::map<std::string, GridPerfMonitor> PerfMonitors {
                                                          {"Total" , GridPerfMonitor()},
                                                          {"Misc" , GridPerfMonitor()},
                                                          {"Orthogonalise" , GridPerfMonitor()},
                                                          {"Copy" , GridPerfMonitor()},
                                                          {"LatticeCoord" , GridPerfMonitor()},
                                                          {"ApplyOp" , GridPerfMonitor()},
                                                          {"PickBlocks" , GridPerfMonitor()},
                                                          {"ProjectToSubspace" , GridPerfMonitor()},
                                                          {"ConstructLinks" , GridPerfMonitor()},
#if defined(SAVE_BLOCKPROJECTS)
                                                          {"InnerBlockSummation" , GridPerfMonitor()},
                                                          {"ShiftLinks" , GridPerfMonitor()}
#endif
                                                          };

      PerfMonitors["Total"].Start();
      PerfMonitors["Misc"].Start();
      FineFermionField phi(FineGrid);
      FineFermionField zeroFerm(FineGrid); zeroFerm = zero;
      FineFermionField Mphi(FineGrid);
      FineFermionField iblock(FineGrid);

      FermionField iProj(Grid());
      FermionField oProj(Grid());

      FineScalarField oneScalar(FineGrid); oneScalar   = 1.;
      FineScalarField zeroScalar(FineGrid); zeroScalar = zero;
      FineScalarField oTmp(FineGrid);
      std::vector<FineScalarField> iTmp(_geom.npoint, FineGrid);

      Lattice<iScalar<vInteger>> coor(FineGrid);

      std::vector<CoarseningLookUpTable> iLut(_geom.npoint);
      std::vector<CoarseningLookUpTable> oLut(_geom.npoint);
      PerfMonitors["Misc"].Stop();

      PerfMonitors["Orthogonalise"].Start();
      // Orthogonalise the subblocks over the basis
      Aggregates.Orthogonalise(1);
      PerfMonitors["Orthogonalise"].Stop();

      PerfMonitors["Misc"].Start();
      // Compute the matrix elements of linop between this orthonormal
      // set of vectors.
      int self_stencil = _geom.SelfStencilPoint();
      for(int p = 0; p < _geom.npoint; p++) {
        _Y[p] = zero;
      }
      PerfMonitors["Misc"].Stop();

      ////////////////////////////////////////////////////////////////////////
      // Pick out contributions coming from this cell and neighbour cell and put them in the lookup table
      ////////////////////////////////////////////////////////////////////////
      for(int p=0;p<_geom.npoint;p++) {
        int dir  = _geom.directions[p];
        int disp = _geom.displacements[p];
        PerfMonitors["LatticeCoord"].Start();
        LatticeCoordinate(coor, dir);
        PerfMonitors["LatticeCoord"].Stop();

        PerfMonitors["PickBlocks"].Start();
        Integer block = (FineGrid->_rdimensions[dir]) / (Grid()->_rdimensions[dir]);
        iLut[p].populate(Grid(), FineGrid);
        oLut[p].populate(Grid(), FineGrid);
        if(disp == 0) {
          iTmp[p] = oneScalar;
          oTmp = zeroScalar;
        } else if(disp == 1) {
          oTmp = where(mod(coor, block) == (block - 1), oneScalar, zeroScalar);
          iTmp[p] = where(mod(coor, block) != (block - 1), oneScalar, zeroScalar);
        } else if(disp == -1) {
          oTmp = where(mod(coor, block) == (Integer)0, oneScalar, zeroScalar);
          iTmp[p] = where(mod(coor, block) != (Integer)0, oneScalar, zeroScalar);
        } else {
          assert(0);
        }

        iLut[p].deleteUnneededFineSites(iTmp[p]);
        oLut[p].deleteUnneededFineSites(oTmp);
        PerfMonitors["PickBlocks"].Stop();
      }

      for(int i = 0; i < Nbasis; i++) {
        PerfMonitors["Copy"].Start();
        phi = Aggregates.Subspace()[i];
        PerfMonitors["Copy"].Stop();

        std::cout << GridLogMessage << "(" << i << ") .." << std::endl;

        iblock = zeroFerm;

        for(int p = 0; p < _geom.npoint; p++) {

          PerfMonitors["Misc"].Start();
          int dir  = _geom.directions[p];
          int disp = _geom.displacements[p];
          PerfMonitors["Misc"].Stop();

          PerfMonitors["ApplyOp"].Start();
          if(disp == 0) {
            linop.OpDiag(phi, Mphi);
          } else {
            linop.OpDir(phi, Mphi, dir, disp);
          }
          PerfMonitors["ApplyOp"].Stop();

#if defined(SAVE_BLOCKPROJECTS)
          PerfMonitors["InnerBlockSummation"].Start();
          iblock = iblock + where(iTmp[p] == 1, Mphi, zeroFerm);
          PerfMonitors["InnerBlockSummation"].Stop();

          PerfMonitors["ProjectToSubspace"].Start();
          auto numProject = 0;
          if(p == self_stencil) { // NOTE: Here, we rely on the self-stencil point being the last in the list. This code will break if it isn't!
            numProject++;
            Aggregates.ProjectToSubspace(iProj, iblock, iLut[p]);
          } else if(disp == +1) {
            numProject++;
            Aggregates.ProjectToSubspace(oProj, Mphi, oLut[p]);
          }
          PerfMonitors["ProjectToSubspace"].Stop(numProject); // TODO: This counts the number of calls correctly, but these are no full projections any longer -> Think about what number to put here

          PerfMonitors["ConstructLinks"].Start();
          parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
            for(int j = 0; j < Nbasis; j++) {
              if(p == self_stencil) { // NOTE: Here, we rely on the self-stencil point being the last in the list. This code will break if it isn't!
                _Y[self_stencil]._odata[ss](j, i) = _Y[self_stencil]._odata[ss](j, i) + iProj._odata[ss](j);
              }
              if(disp == +1) {
                _Y[p]._odata[ss](j, i) = oProj._odata[ss](j);
              }
            }
          }
          PerfMonitors["ConstructLinks"].Stop();
        }
      }
      // Relation between forward and backward link matrices taken from M. Rottmann's PHD thesis:
      // D_{A_{q,\kappa}, A_{p,\tau}} = - D^\dag_{A_{p,\tau}, A_{q,\kappa}}
      PerfMonitors["ShiftLinks"].Start();
      for(int p = 0; p < _geom.npoint; p++) {
        if(_geom.displacements[p] == +1) {
          auto tmp = adj(_Y[p]);
          parallel_for(auto ss = tmp.begin(); ss < tmp.end(); ss++) { // TODO: Is there a fancier way to do this?
            Real factor;
            for(int n1 = 0; n1 < Nbasis; n1++) {
              int k = n1/(Nbasis/2);
              for(int n2 = 0; n2 < Nbasis; n2++) {
                int l = n2 / (Nbasis / 2);
                if((k + l) % 2 == 1) {
                  factor = -1.;
                } else {
                  factor = 1.;
                }
                tmp._odata[ss](n1, n2) = factor * tmp._odata[ss](n1, n2);
              }
            }
          }
          _Y[p + 1] = Cshift(tmp, _geom.directions[p], -1);
        }
      }
      PerfMonitors["ShiftLinks"].Stop();

      std::string saveBlockProjects = "true";
#else
          PerfMonitors["ProjectToSubspace"].Start();
          Aggregates.ProjectToSubspace(iProj, Mphi, iLut[p]);
          Aggregates.ProjectToSubspace(oProj, Mphi, oLut[p]);
          PerfMonitors["ProjectToSubspace"].Stop(2);

          PerfMonitors["ConstructLinks"].Start();
          parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
            for(int j = 0; j < Nbasis; j++) {
              if(disp != 0) {
                _Y[p]._odata[ss](j, i) = oProj._odata[ss](j);
              }
              _Y[self_stencil]._odata[ss](j, i) = _Y[self_stencil]._odata[ss](j, i) + iProj._odata[ss](j);
            }
          }
          PerfMonitors["ConstructLinks"].Stop();
        }
      }
      std::string saveBlockProjects = "false";
#endif
      PerfMonitors["Total"].Stop();
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogMessage << "Time breakdown for CoarsenOperator with isTwoSpinVersion == false and saveBlockProjects = " << saveBlockProjects << std::endl;
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogPerformance << PerfMonitors;
    }

    template <bool isTwoSpinVersion, typename std::enable_if<isTwoSpinVersion == true>::type *  = nullptr>
    void extractChiralComponents(std::vector<FineFermionField> &extracted, FineFermionField const &in) {
      assert(extracted.size() == Ncs);

      for(int k = 0; k < Ncs; k++)
        conformable(in._grid, extracted[k]._grid);

      for(int k = 0; k < Ncs; k++) extracted[k] = zero;

#define SpinIndex 1 // Need to do this temporarily, will be removed once the QCD namespace is gone
      parallel_for(int ss = 0; ss < in._grid->oSites(); ss++) {
        for(int s = 0; s < Nfs; s++) {
          auto tmp = peekIndex<SpinIndex>(in._odata[ss], s);
          pokeIndex<SpinIndex>(extracted[s/Nsb]._odata[ss], tmp, s);
        }
      }
#undef SpinIndex
    }

    template<bool isTwoSpinVersion, typename std::enable_if<isTwoSpinVersion == true>::type *  = nullptr>
    void doOperatorCoarsening(GridBase *FineGrid, LinearOperatorBase<FineFermionField> &linop, AggregationUsingPolicies<CoarseningPolicy> &Aggregates) {
      std::map<std::string, GridPerfMonitor> PerfMonitors {
                                                          {"Total", GridPerfMonitor()},
                                                          {"Misc", GridPerfMonitor()},
                                                          {"Orthogonalise", GridPerfMonitor()},
                                                          {"Copy", GridPerfMonitor()},
                                                          {"LatticeCoord", GridPerfMonitor()},
                                                          {"ApplyOp", GridPerfMonitor()},
                                                          {"PickBlocks", GridPerfMonitor()},
                                                          {"ProjectToSubspace", GridPerfMonitor()},
                                                          {"ConstructLinks", GridPerfMonitor()},
#if defined(SAVE_BLOCKPROJECTS)
                                                          {"InnerBlockSummation" , GridPerfMonitor()},
                                                          {"ShiftLinks", GridPerfMonitor()}
#endif
                                                          };

      PerfMonitors["Total"].Start();
      PerfMonitors["Misc"].Start();
      FineFermionField zeroFerm(FineGrid); zeroFerm = zero;
      std::vector<FineFermionField> phiSplit(Ncs, FineGrid);
      std::vector<FineFermionField> MphiSplit(Ncs, FineGrid);
      std::vector<FineFermionField> iBlock(Ncs, FineGrid);

      std::vector<FermionField> iProjSplit(Ncs, Grid());
      std::vector<FermionField> oProjSplit(Ncs, Grid());

      FineScalarField oneScalar(FineGrid); oneScalar = 1.;
      FineScalarField zeroScalar(FineGrid); zeroScalar = zero;
      FineScalarField oTmp(FineGrid);
      std::vector<FineScalarField> iTmp(_geom.npoint, FineGrid);

      Lattice<iScalar<vInteger> > coor(FineGrid);

      std::vector<CoarseningLookUpTable> iLut(_geom.npoint);
      std::vector<CoarseningLookUpTable> oLut(_geom.npoint);
      PerfMonitors["Misc"].Stop();

      PerfMonitors["Orthogonalise"].Start();
      // // Orthogonalise the subblocks over the basis
      Aggregates.Orthogonalise(1);
      PerfMonitors["Orthogonalise"].Stop();

      PerfMonitors["Misc"].Start();
      // Compute the matrix elements of linop between this orthonormal
      // set of vectors.
      int self_stencil = _geom.SelfStencilPoint();
      for(int p = 0; p < _geom.npoint; p++) {
        _Y[p] = zero;
      }
      PerfMonitors["Misc"].Stop();

      ////////////////////////////////////////////////////////////////////////
      // Pick out contributions coming from this cell and neighbour cell and put them in the lookup table
      ////////////////////////////////////////////////////////////////////////
      for(int p=0;p<_geom.npoint;p++) {
        int dir  = _geom.directions[p];
        int disp = _geom.displacements[p];
        PerfMonitors["LatticeCoord"].Start();
        LatticeCoordinate(coor, dir);
        PerfMonitors["LatticeCoord"].Stop();

        PerfMonitors["PickBlocks"].Start();
        Integer block = (FineGrid->_rdimensions[dir]) / (Grid()->_rdimensions[dir]);
        iLut[p].populate(Grid(), FineGrid);
        oLut[p].populate(Grid(), FineGrid);
        if(disp == 0) {
          iTmp[p] = oneScalar;
          oTmp = zeroScalar;
        } else if(disp == 1) {
          oTmp = where(mod(coor, block) == (block - 1), oneScalar, zeroScalar);
          iTmp[p] = where(mod(coor, block) != (block - 1), oneScalar, zeroScalar);
        } else if(disp == -1) {
          oTmp = where(mod(coor, block) == (Integer)0, oneScalar, zeroScalar);
          iTmp[p] = where(mod(coor, block) != (Integer)0, oneScalar, zeroScalar);
        } else {
          assert(0);
        }
        iLut[p].deleteUnneededFineSites(iTmp[p]);
        oLut[p].deleteUnneededFineSites(oTmp);
        PerfMonitors["PickBlocks"].Stop();
      }

      for(int i = 0; i < Nbasis; i++) {
        PerfMonitors["Copy"].Start();
        extractChiralComponents<isTwoSpinVersion>(phiSplit, Aggregates.Subspace()[i]);
        PerfMonitors["Copy"].Stop();

        std::cout << GridLogMessage << "(" << i << ") .." << std::endl;

        for(int k = 0; k < Ncs; k++) iBlock[k] = zeroFerm;

        for(int p = 0; p < _geom.npoint; p++) {
            PerfMonitors["Misc"].Start();
          int dir  = _geom.directions[p];
          int disp = _geom.displacements[p];
          PerfMonitors["Misc"].Stop();

          PerfMonitors["ApplyOp"].Start();
          if(disp == 0) {
            for(int k = 0; k < Ncs; k++) linop.OpDiag(phiSplit[k], MphiSplit[k]);
          } else {
            for(int k = 0; k < Ncs; k++) linop.OpDir(phiSplit[k], MphiSplit[k], dir, disp);
          }
          PerfMonitors["ApplyOp"].Stop(Ncs);

#if defined(SAVE_BLOCKPROJECTS)
          PerfMonitors["InnerBlockSummation"].Start();
          for(int k = 0; k < Ncs; k++) iBlock[k] = iBlock[k] + where(iTmp[p] == 1, MphiSplit[k], zeroFerm);
          PerfMonitors["InnerBlockSummation"].Stop(Ncs);

          PerfMonitors["ProjectToSubspace"].Start();
          auto numProject = 0;
          if(p == self_stencil) { // NOTE: Here, we rely on the self-stencil point being the last in the list. This code will break if it isn't!
            numProject += Ncs;
            for(int k = 0; k < Ncs; k++) Aggregates.ProjectToSubspace(iProjSplit[k], iBlock[k], iLut[p]);
          } else if(disp == +1) {
            numProject += Ncs;
            for(int k = 0; k < Ncs; k++) Aggregates.ProjectToSubspace(oProjSplit[k], MphiSplit[k], oLut[p]);
          }
          PerfMonitors["ProjectToSubspace"].Stop(numProject); // TODO: This counts the number of calls correctly, but these are no full projections any longer -> Think about what number to put here

          PerfMonitors["ConstructLinks"].Start();
          parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
            for(int j = 0; j < Nbasis; j++) {
              if(p == self_stencil) { // NOTE: Here, we rely on the self-stencil point being the last in the list. This code will break if it isn't!
                for(int k = 0; k < Ncs; k++)
                  for(int l = 0; l < Ncs; l++)
                    _Y[self_stencil]._odata[ss]()(l, k)(j, i) = _Y[self_stencil]._odata[ss]()(l, k)(j, i) + iProjSplit[k]._odata[ss]()(l)(j);
              }
              if(disp == +1) {
                for(int k = 0; k < Ncs; k++)
                  for(int l = 0; l < Ncs; l++) _Y[p]._odata[ss]()(l, k)(j, i) = oProjSplit[k]._odata[ss]()(l)(j);
              }
            }
          }
          PerfMonitors["ConstructLinks"].Stop();
        }
      }
      // Relation between forward and backward link matrices taken from M. Rottmann's PHD thesis:
      // D_{A_{q,\kappa}, A_{p,\tau}} = - D^\dag_{A_{p,\tau}, A_{q,\kappa}}
      PerfMonitors["ShiftLinks"].Start();
      for(int p = 0; p < _geom.npoint; p++) {
        if(_geom.displacements[p] == +1) {
          auto tmp = adj(_Y[p]);
          parallel_for(auto ss = tmp.begin(); ss < tmp.end(); ss++) { // TODO: Is there a fancier way to do this?
            Real factor;
            for(int k = 0; k < Ncs; k++) {
              for(int l = 0; l < Ncs; l++) {
                if((k + l) % 2 == 1) {
                  factor = -1.;
                } else {
                  factor = 1.;
                }
                tmp._odata[ss]()(k, l) = factor * tmp._odata[ss]()(k, l);
              }
            }
          }
          _Y[p + 1] = Cshift(tmp, _geom.directions[p], -1);
        }
      }
      PerfMonitors["ShiftLinks"].Stop();

      std::string saveBlockProjects = "true";
#else
          PerfMonitors["ProjectToSubspace"].Start();
          for(int k = 0; k < Ncs; k++) {
            Aggregates.ProjectToSubspace(iProjSplit[k], MphiSplit[k], iLut[p]);
            Aggregates.ProjectToSubspace(oProjSplit[k], MphiSplit[k], oLut[p]);
          }
          PerfMonitors["ProjectToSubspace"].Stop(Ncs*2);

          PerfMonitors["ConstructLinks"].Start();
          parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
            for(int j = 0; j < Nbasis; j++) {
              if(disp != 0) {
                for(int k = 0; k < Ncs; k++)
                  for(int l = 0; l < Ncs; l++)
                    _Y[p]._odata[ss]()(l, k)(j, i) = oProjSplit[k]._odata[ss]()(l)(j);
              }
              for(int k = 0; k < Ncs; k++)
                for(int l = 0; l < Ncs; l++)
                  _Y[self_stencil]._odata[ss]()(l, k)(j, i) = _Y[self_stencil]._odata[ss]()(l, k)(j, i) + iProjSplit[k]._odata[ss]()(l)(j);
            }
          }
          PerfMonitors["ConstructLinks"].Stop();
        }
      }
      std::string saveBlockProjects = "false";
#endif

      PerfMonitors["Total"].Stop();
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogMessage << "Time breakdown for CoarsenOperator with isTwoSpinVersion == true and saveBlockProjects = " << saveBlockProjects << std::endl;
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogPerformance << PerfMonitors;
    }

    void CoarsenOperator(GridBase *FineGrid, LinearOperatorBase<FineFermionField> &linop, AggregationUsingPolicies<CoarseningPolicy> &Aggregates) {
      doOperatorCoarsening<CoarseningPolicy::isTwoSpinVersion>(FineGrid, linop, Aggregates);
    }


  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // Original implementations
  //////////////////////////////////////////////////////////////////////////////////////////


  template<class Fobj,class CComplex,int nbasis>
  class Aggregation   {
  public:
    typedef iVector<CComplex,nbasis >             siteVector;
    typedef Lattice<siteVector>                 CoarseVector;
    typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;

    typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
    typedef Lattice<Fobj >        FineField;

    GridBase *CoarseGrid;
    GridBase *FineGrid;
    std::vector<Lattice<Fobj> > subspace;
    int checkerboard;

  Aggregation(GridBase *_CoarseGrid,GridBase *_FineGrid,int _checkerboard) : 
    CoarseGrid(_CoarseGrid),
      FineGrid(_FineGrid),
      subspace(nbasis,_FineGrid),
      checkerboard(_checkerboard)
	{
	};
  
    std::vector<FineField> &Subspace() {
      return subspace;
    }

    void Orthogonalise(int passes = 2){
      CoarseScalar InnerProd(CoarseGrid);
      for(int n = 0; n < passes; ++n) {
        std::cout << GridLogMessage <<" Gramm-Schmidt pass " << n+1 << std::endl;
        OriginalImpl::blockOrthogonalise(InnerProd, subspace);
      }
      std::cout << GridLogMessage <<" Gramm-Schmidt checking orthogonality" << std::endl;
      CheckOrthogonal();
    } 
    void CheckOrthogonal(void){
      CoarseVector iProj(CoarseGrid); 
      CoarseVector eProj(CoarseGrid); 
      for(int i=0;i<nbasis;i++){
	ProjectToSubspace(iProj, subspace[i]);
	eProj=zero;
	parallel_for(int ss=0;ss<CoarseGrid->oSites();ss++){
	  eProj._odata[ss](i)=CComplex(1.0);
	}
	eProj=eProj - iProj;
	std::cout<<GridLogMessage<<"Orthog check error "<<i<<" " << norm2(eProj)<<std::endl;
      }
      std::cout<<GridLogMessage <<"CheckOrthog done"<<std::endl;
    }
    void ProjectToSubspace(CoarseVector &CoarseVec,const FineField &FineVec){
      OriginalImpl::blockProject(CoarseVec,FineVec,subspace);
    }
    void PromoteFromSubspace(const CoarseVector &CoarseVec,FineField &FineVec){
      FineVec.checkerboard = subspace[0].checkerboard;
      blockPromote(CoarseVec,FineVec,subspace);
    }
    void CreateSubspaceRandom(GridParallelRNG &RNG){
      for(int i=0;i<nbasis;i++){
	random(RNG,subspace[i]);
	std::cout<<GridLogMessage<<" norm subspace["<<i<<"] "<<norm2(subspace[i])<<std::endl;
      }
      Orthogonalise();
    }

    /*
    virtual void CreateSubspaceLanczos(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis) 
    {
      // Run a Lanczos with sloppy convergence
	const int Nstop = nn;
	const int Nk = nn+20;
	const int Np = nn+20;
	const int Nm = Nk+Np;
	const int MaxIt= 10000;
	RealD resid = 1.0e-3;

	Chebyshev<FineField> Cheb(0.5,64.0,21);
	ImplicitlyRestartedLanczos<FineField> IRL(hermop,Cheb,Nstop,Nk,Nm,resid,MaxIt);
	//	IRL.lock = 1;

	FineField noise(FineGrid); gaussian(RNG,noise);
	FineField tmp(FineGrid); 
	std::vector<RealD>     eval(Nm);
	std::vector<FineField> evec(Nm,FineGrid);

	int Nconv;
	IRL.calc(eval,evec,
		 noise,
		 Nconv);

    	// pull back nn vectors
	for(int b=0;b<nn;b++){

	  subspace[b]   = evec[b];

	  std::cout << GridLogMessage <<"subspace["<<b<<"] = "<<norm2(subspace[b])<<std::endl;

	  hermop.Op(subspace[b],tmp); 
	  std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(tmp)<<std::endl;

	  noise = tmp -  sqrt(eval[b])*subspace[b] ;

	  std::cout<<GridLogMessage << " lambda_"<<b<<" = "<< eval[b] <<"  ;  [ M - Lambda ]_"<<b<<" vec_"<<b<<"  = " <<norm2(noise)<<std::endl;

	  noise = tmp +  eval[b]*subspace[b] ;

	  std::cout<<GridLogMessage << " lambda_"<<b<<" = "<< eval[b] <<"  ;  [ M - Lambda ]_"<<b<<" vec_"<<b<<"  = " <<norm2(noise)<<std::endl;

	}
	Orthogonalise();
	for(int b=0;b<nn;b++){
	  std::cout << GridLogMessage <<"subspace["<<b<<"] = "<<norm2(subspace[b])<<std::endl;
	}
    }
    */
    virtual void CreateSubspace(GridParallelRNG  &RNG,LinearOperatorBase<FineField> &hermop,int nn=nbasis) {

      RealD scale;

      ConjugateGradient<FineField> CG(1.0e-2,10000);
      FineField noise(FineGrid);
      FineField Mn(FineGrid);

      for(int b=0;b<nn;b++){
	
	subspace[b] = zero;
	gaussian(RNG,noise);
	scale = std::pow(norm2(noise),-0.5); 
	noise=noise*scale;

	hermop.Op(noise,Mn); std::cout<<GridLogMessage << "noise   ["<<b<<"] <n|MdagM|n> "<<norm2(Mn)<<std::endl;

	for(int i=0;i<1;i++){

	  CG(hermop,noise,subspace[b]);

	  noise = subspace[b];
	  scale = std::pow(norm2(noise),-0.5); 
	  noise=noise*scale;

	}

	hermop.Op(noise,Mn); std::cout<<GridLogMessage << "filtered["<<b<<"] <f|MdagM|f> "<<norm2(Mn)<<std::endl;
	subspace[b]   = noise;

      }

      Orthogonalise();

    }

    virtual void CreateSubspaceDDalphaAMG(GridParallelRNG &RNG, LinearOperatorBase<FineField> &linop, bool startFromRandom, int nn = nbasis, Integer restartLength = 20) {
      RealD     scale;
      FineField tmp(FineGrid);
      FineField Mphi(FineGrid);

      // TODO: We shouldn't fix this to GMRES here but accept a general smooter as argument
      GeneralisedMinimalResidual<FineField> GMRES(1.0e-14, restartLength, restartLength, false);

      for(int b = 0; b < nn; b++) {
        if(startFromRandom)
          gaussian(RNG, subspace[b]);

        linop.Op(subspace[b], Mphi);
        auto normRandom = norm2(Mphi);

        // NOTE: This is the version MR actually proposes in his PHD thesis
        // tmp = zero;
        // for(int i = 0; i < 3; i++) {
        //   GMRES(linop, subspace[b], tmp);
        // }
        // scale       = std::pow(norm2(tmp), -0.5);
        // subspace[b] = tmp * scale;

        // NOTE: This is the version I see better results with
        for(int i = 0; i < 3; i++) {
          tmp = zero;
          GMRES(linop, subspace[b], tmp);
          subspace[b] = tmp;
        }
        scale       = std::pow(norm2(subspace[b]), -0.5);
        subspace[b] = subspace[b] * scale;

        linop.Op(subspace[b], Mphi);
        auto normFiltered = norm2(Mphi);
        std::cout << GridLogMessage             << "Vector [" << b << "]:"
                  << " <n|MdagM|n> random = "   << normRandom
                  << " <n|MdagM|n> filtered = " << normFiltered
                  << " reduction = "            << normFiltered / normRandom
                  << std::endl;
      }
      for(int b = nn; b < nbasis; b++) {
        subspace[b] = zero;
      }
    }

    void DoChiralDoubling() {
      assert(subspace.size() % 2 == 0);
      auto nb = subspace.size() / 2;

      for(int n = 0; n < nb; n++) {
        auto tmp1 = subspace[n];
        auto tmp2 = tmp1;
        G5C(tmp2, subspace[n]);
        axpby(subspace[n], 0.5, 0.5, tmp1, tmp2);
        axpby(subspace[n + nb], 0.5, -0.5, tmp1, tmp2);
        std::cout << GridLogMessage << "Chirally doubled vector " << n << ". "
                  << "norm2(vec[" << n << "]) = " << norm2(subspace[n]) << ". "
                  << "norm2(vec[" << n + nb << "]) = " << norm2(subspace[n + nb]) << std::endl;
      }
    }
  };
  // Fine Object == (per site) type of fine field
  // nbasis      == number of deflation vectors
  template<class Fobj,class CComplex,int nbasis>
  class CoarsenedMatrix : public SparseMatrixBase<Lattice<iVector<CComplex,nbasis > > >  {
  public:
    
    typedef iVector<CComplex,nbasis >             siteVector;
    typedef Lattice<siteVector>                 CoarseVector;
    typedef Lattice<iMatrix<CComplex,nbasis > > CoarseMatrix;

    typedef Lattice< CComplex >   CoarseScalar; // used for inner products on fine field
    typedef Lattice<Fobj >        FineField;

    ////////////////////
    // Data members
    ////////////////////
    Geometry         geom;
    GridBase *       _grid; 
    CartesianStencil<siteVector,siteVector> Stencil; 

    std::vector<CoarseMatrix> A;

      
    ///////////////////////
    // Interface
    ///////////////////////
    GridBase * Grid(void)         { return _grid; };   // this is all the linalg routines need to know

    RealD M (const CoarseVector &in, CoarseVector &out){

      conformable(_grid,in._grid);
      conformable(in._grid,out._grid);

      SimpleCompressor<siteVector> compressor;
      Stencil.HaloExchange(in,compressor);

      parallel_for(int ss=0;ss<Grid()->oSites();ss++){
        siteVector res = zero;
	siteVector nbr;
	int ptype;
	StencilEntry *SE;
	for(int point=0;point<geom.npoint;point++){

	  SE=Stencil.GetEntry(ptype,point,ss);
	  
	  if(SE->_is_local&&SE->_permute) { 
	    permute(nbr,in._odata[SE->_offset],ptype);
	  } else if(SE->_is_local) { 
	    nbr = in._odata[SE->_offset];
	  } else {
	    nbr = Stencil.CommBuf()[SE->_offset];
	  }
	  res = res + A[point]._odata[ss]*nbr;
	}
	vstream(out._odata[ss],res);
      }
      return norm2(out);
    };

    RealD Mdag (const CoarseVector &in, CoarseVector &out){
      // // corresponds to Petrov-Galerkin coarsening
      // return M(in,out);

      // corresponds to Galerkin coarsening
      CoarseVector tmp(Grid());
      G5C(tmp, in);
      M(tmp, out);
      G5C(out, out);
      return norm2(out);
    };

    void Mdir(const CoarseVector &in, CoarseVector &out, int dir, int disp){

      conformable(_grid,in._grid);
      conformable(in._grid,out._grid);

      SimpleCompressor<siteVector> compressor;
      Stencil.HaloExchange(in,compressor);

      auto point = [dir, disp](){
        if(dir == 0 and disp == 0)
          return 8;
        else
          return (4 * dir + 1 - disp) / 2;
      }();

      parallel_for(int ss=0;ss<Grid()->oSites();ss++){
        siteVector res = zero;
        siteVector nbr;
        int ptype;
        StencilEntry *SE;

        SE=Stencil.GetEntry(ptype,point,ss);

        if(SE->_is_local&&SE->_permute) {
          permute(nbr,in._odata[SE->_offset],ptype);
        } else if(SE->_is_local) {
          nbr = in._odata[SE->_offset];
        } else {
          nbr = Stencil.CommBuf()[SE->_offset];
        }

        res = res + A[point]._odata[ss]*nbr;

        vstream(out._odata[ss],res);
      }
    };

    void Mdiag(const CoarseVector &in, CoarseVector &out){
      Mdir(in, out, 0, 0); // use the self coupling (= last) point of the stencil
    };

    CoarsenedMatrix(GridCartesian &CoarseGrid) 	: 

      _grid(&CoarseGrid),
      geom(CoarseGrid._ndimension),
      Stencil(&CoarseGrid,geom.npoint,Even,geom.directions,geom.displacements),
      A(geom.npoint,&CoarseGrid)
    {
    };

    void CoarsenOperator(GridBase *FineGrid,LinearOperatorBase<Lattice<Fobj> > &linop,
			 Aggregation<Fobj,CComplex,nbasis> & Subspace){

      std::map<std::string, GridPerfMonitor> PerfMonitors {
                                                          {"Total" , GridPerfMonitor()},
                                                          {"Misc" , GridPerfMonitor()},
                                                          {"Orthogonalise" , GridPerfMonitor()},
                                                          {"Copy" , GridPerfMonitor()},
                                                          {"LatticeCoord" , GridPerfMonitor()},
                                                          {"ApplyOp" , GridPerfMonitor()},
                                                          {"PickBlocks" , GridPerfMonitor()},
                                                          {"ProjectToSubspace" , GridPerfMonitor()},
                                                          {"ConstructLinks" , GridPerfMonitor()}
                                                          };

      PerfMonitors["Total"].Start();
      PerfMonitors["Misc"].Start();

      FineField iblock(FineGrid); // contributions from within this block
      FineField oblock(FineGrid); // contributions from outwith this block

      FineField     phi(FineGrid);
      FineField     tmp(FineGrid);
      FineField     zz(FineGrid); zz=zero;
      FineField    Mphi(FineGrid);

      Lattice<iScalar<vInteger> > coor(FineGrid);

      CoarseVector iProj(Grid()); 
      CoarseVector oProj(Grid()); 
      CoarseScalar InnerProd(Grid()); 
      PerfMonitors["Misc"].Stop();

      PerfMonitors["Orthogonalise"].Start();
      // Orthogonalise the subblocks over the basis
      Subspace.Orthogonalise(1);
      PerfMonitors["Orthogonalise"].Stop();

      PerfMonitors["Misc"].Start();
      // Compute the matrix elements of linop between this orthonormal
      // set of vectors.
      int self_stencil=-1;
      for(int p=0;p<geom.npoint;p++){ 
	A[p]=zero;
	if( geom.displacements[p]==0){
	  self_stencil=p;
	}
      }
      assert(self_stencil!=-1);
      PerfMonitors["Misc"].Stop();

      for(int i=0;i<nbasis;i++){
        PerfMonitors["Copy"].Start();
	phi=Subspace.Subspace()[i];
        PerfMonitors["Copy"].Stop();

	std::cout<<GridLogMessage<<"("<<i<<").."<<std::endl;

	for(int p=0;p<geom.npoint;p++){ 

          PerfMonitors["Misc"].Start();
	  int dir   = geom.directions[p];
	  int disp  = geom.displacements[p];

	  Integer block=(FineGrid->_rdimensions[dir])/(Grid()->_rdimensions[dir]);
          PerfMonitors["Misc"].Stop();

          PerfMonitors["LatticeCoord"].Start();
	  LatticeCoordinate(coor,dir);
          PerfMonitors["LatticeCoord"].Stop();

          PerfMonitors["ApplyOp"].Start();
	  if ( disp==0 ){
	    linop.OpDiag(phi,Mphi);
	  }
	  else  {
	    linop.OpDir(phi,Mphi,dir,disp); 
	  }
          PerfMonitors["ApplyOp"].Stop();

	  ////////////////////////////////////////////////////////////////////////
	  // Pick out contributions coming from this cell and neighbour cell
	  ////////////////////////////////////////////////////////////////////////
          PerfMonitors["PickBlocks"].Start();
	  if ( disp==0 ) {
	    iblock = Mphi;
	    oblock = zero;
	  } else if ( disp==1 ) {
	    oblock = where(mod(coor,block)==(block-1),Mphi,zz);
	    iblock = where(mod(coor,block)!=(block-1),Mphi,zz);
	  } else if ( disp==-1 ) {
	    oblock = where(mod(coor,block)==(Integer)0,Mphi,zz);
	    iblock = where(mod(coor,block)!=(Integer)0,Mphi,zz);
	  } else {
	    assert(0);
	  }
          PerfMonitors["PickBlocks"].Stop();

          PerfMonitors["ProjectToSubspace"].Start();
	  Subspace.ProjectToSubspace(iProj, iblock);
	  Subspace.ProjectToSubspace(oProj,oblock);
	  //	  blockProject(iProj,iblock,Subspace.Subspace());
	  //	  blockProject(oProj,oblock,Subspace.Subspace());
          PerfMonitors["ProjectToSubspace"].Stop(2);

          PerfMonitors["ConstructLinks"].Start();
	  parallel_for(int ss = 0; ss < Grid()->oSites(); ss++) {
	    for(int j=0;j<nbasis;j++){
	      if( disp!= 0 ) {
		A[p]._odata[ss](j,i) = oProj._odata[ss](j);
	      }
	      A[self_stencil]._odata[ss](j,i) =	A[self_stencil]._odata[ss](j,i) + iProj._odata[ss](j);
	    }
	  }
          PerfMonitors["ConstructLinks"].Stop();
	}
      }
      PerfMonitors["Total"].Stop();
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogMessage << "Time breakdown for CoarsenOperator for original implementation" << std::endl;
      std::cout << GridLogMessage << "***************************************************************************" << std::endl;
      std::cout << GridLogPerformance << PerfMonitors;

#if 0
      ///////////////////////////
      // test code worth preserving in if block
      ///////////////////////////
      std::cout<<GridLogMessage<< " Computed matrix elements "<< self_stencil <<std::endl;
      for(int p=0;p<geom.npoint;p++){
	std::cout<<GridLogMessage<< "A["<<p<<"]" << std::endl;
	std::cout<<GridLogMessage<< A[p] << std::endl;
      }
      std::cout<<GridLogMessage<< " picking by block0 "<< self_stencil <<std::endl;

      phi=Subspace.Subspace()[0];
      std::vector<int> bc(FineGrid->_ndimension,0);

      blockPick(Grid(),phi,tmp,bc);      // Pick out a block
      linop.Op(tmp,Mphi);                // Apply big dop
      OriginalImpl::blockProject(iProj,Mphi,Subspace.Subspace()); // project it and print it
      std::cout<<GridLogMessage<< " Computed matrix elements from block zero only "<<std::endl;
      std::cout<<GridLogMessage<< iProj <<std::endl;
      std::cout<<GridLogMessage<<"Computed Coarse Operator"<<std::endl;
#endif
      //      ForceHermitian();
      // AssertHermitian();
      // ForceDiagonal();
    }
    void ForceDiagonal(void) {


      std::cout<<GridLogMessage<<"**************************************************"<<std::endl;
      std::cout<<GridLogMessage<<"****   Forcing coarse operator to be diagonal ****"<<std::endl;
      std::cout<<GridLogMessage<<"**************************************************"<<std::endl;
      for(int p=0;p<8;p++){
	A[p]=zero;
      }

      GridParallelRNG  RNG(Grid()); RNG.SeedFixedIntegers(std::vector<int>({55,72,19,17,34}));
      Lattice<iScalar<CComplex> > val(Grid()); random(RNG,val);

      Complex one(1.0);

      iMatrix<CComplex,nbasis> ident;  ident=one;

      val = val*adj(val);
      val = val + 1.0;

      A[8] = val*ident;

      //      for(int s=0;s<Grid()->oSites();s++) {
      //	A[8]._odata[s]=val._odata[s];
      //      }
    }
    void ForceHermitian(void) {
      for(int d=0;d<4;d++){
	int dd=d+1;
	A[2*d] = adj(Cshift(A[2*d+1],dd,1));
      }
      //      A[8] = 0.5*(A[8] + adj(A[8]));
    }
    void AssertHermitian(void) {
      CoarseMatrix AA    (Grid());
      CoarseMatrix AAc   (Grid());
      CoarseMatrix Diff  (Grid());
      for(int d=0;d<4;d++){
	
	int dd=d+1;
	AAc = Cshift(A[2*d+1],dd,1);
	AA  = A[2*d];
	
	Diff = AA - adj(AAc);

	std::cout<<GridLogMessage<<"Norm diff dim "<<d<<" "<< norm2(Diff)<<std::endl;
	std::cout<<GridLogMessage<<"Norm dim "<<d<<" "<< norm2(AA)<<std::endl;
	  
      }
      Diff = A[8] - adj(A[8]);
      std::cout<<GridLogMessage<<"Norm diff local "<< norm2(Diff)<<std::endl;
      std::cout<<GridLogMessage<<"Norm local "<< norm2(A[8])<<std::endl;
    }
    
  };

}
#endif
