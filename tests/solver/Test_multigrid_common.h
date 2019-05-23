/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/solver/Test_multigrid_common.h

    Copyright (C) 2015-2018

    Author: Daniel Richtmann <daniel.richtmann@ur.de>

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
#ifndef GRID_TEST_MULTIGRID_COMMON_H
#define GRID_TEST_MULTIGRID_COMMON_H

#define GridLogMGrid(level) GridLogMG << "Level " << level << ": "

namespace Grid {

// TODO: Can think about having one parameter struct per level and then a
// vector of these structs. How well would that work together with the
// serialization strategy of Grid?

// clang-format off
struct MultiGridParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MultiGridParams,
                                  int,                           nLevels,
                                  std::vector<std::vector<int>>, blockSizes,           // size == nLevels - 1
                                  std::vector<int>,              setupIter,            // size == nLevels - 1
                                  bool,                          startSubspaceFromRandom,
                                  std::vector<double>,           smootherTol,          // size == nLevels - 1
                                  std::vector<int>,              smootherMaxOuterIter, // size == nLevels - 1
                                  std::vector<int>,              smootherMaxInnerIter, // size == nLevels - 1
                                  bool,                          kCycle,
                                  std::vector<double>,           kCycleTol,            // size == nLevels - 1
                                  std::vector<int>,              kCycleMaxOuterIter,   // size == nLevels - 1
                                  std::vector<int>,              kCycleMaxInnerIter,   // size == nLevels - 1
                                  double,                        coarseSolverTol,
                                  int,                           coarseSolverMaxOuterIter,
                                  int,                           coarseSolverMaxInnerIter,
                                  bool,                          coarseSolverUseRBPrec
                                  );

  // constructor with default values
  MultiGridParams(int                           _nLevels                  = 2,
                  std::vector<std::vector<int>> _blockSizes               = {{4, 4, 4, 4}},
                  std::vector<int>              _setupIter                = {4},
                  std::vector<double>           _smootherTol              = {1e-14},
                  std::vector<int>              _smootherMaxOuterIter     = {4},
                  std::vector<int>              _smootherMaxInnerIter     = {4},
                  bool                          _startSubspaceFromRandom  = true,
                  bool                          _kCycle                   = false,
                  std::vector<double>           _kCycleTol                = {1e-1},
                  std::vector<int>              _kCycleMaxOuterIter       = {2},
                  std::vector<int>              _kCycleMaxInnerIter       = {5},
                  double                        _coarseSolverTol          = 5e-2,
                  int                           _coarseSolverMaxOuterIter = 10,
                  int                           _coarseSolverMaxInnerIter = 500,
                  bool                          _coarseSolverUseRBPrec    = true
                  )
  : nLevels(_nLevels)
  , blockSizes(_blockSizes)
  , setupIter(_setupIter)
  , startSubspaceFromRandom(_startSubspaceFromRandom)
  , smootherTol(_smootherTol)
  , smootherMaxOuterIter(_smootherMaxOuterIter)
  , smootherMaxInnerIter(_smootherMaxInnerIter)
  , kCycle(_kCycle)
  , kCycleTol(_kCycleTol)
  , kCycleMaxOuterIter(_kCycleMaxOuterIter)
  , kCycleMaxInnerIter(_kCycleMaxInnerIter)
  , coarseSolverTol(_coarseSolverTol)
  , coarseSolverMaxOuterIter(_coarseSolverMaxOuterIter)
  , coarseSolverMaxInnerIter(_coarseSolverMaxInnerIter)
  , coarseSolverUseRBPrec(_coarseSolverUseRBPrec)
  {}
};

class MGTestOtherParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MGTestOtherParams,
                                  RealD,            outerSolverTol,
                                  Integer,          outerSolverMaxOuterIter,
                                  Integer,          outerSolverMaxInnerIter,
                                  RealD,            mass,
                                  RealD,            csw,
                                  std::string,      config,
                                  std::string,      sourceType,
                                  bool,             useAntiPeriodicBC
                                  );

  // constructor with default values
  MGTestOtherParams(RealD       _outerSolverTol          = 1e-12,
                    Integer     _outerSolverMaxOuterIter = 100,
                    Integer     _outerSolverMaxInnerIter = 20,
                    RealD       _mass                    = 0.1,
                    RealD       _csw                     = 1.0,
                    std::string _config                  = "foo",
                    std::string _sourceType              = "random",
                    bool _useAntiPeriodicBC              = false
                    )
  : outerSolverTol(_outerSolverTol)
  , outerSolverMaxOuterIter(_outerSolverMaxOuterIter)
  , outerSolverMaxInnerIter(_outerSolverMaxInnerIter)
  , mass(_mass)
  , csw(_csw)
  , config(_config)
  , sourceType(_sourceType)
  , useAntiPeriodicBC(_useAntiPeriodicBC)
  {}
};

class MGTestParams : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MGTestParams,
                                  MultiGridParams,   mg,
                                  MGTestOtherParams, test
                                  );
};
// clang-format on

void checkParameterValidity(MultiGridParams const &params) {

  auto correctSize = params.nLevels - 1;

  assert(correctSize == params.blockSizes.size());
  assert(correctSize == params.setupIter.size());
  assert(correctSize == params.smootherTol.size());
  assert(correctSize == params.smootherMaxOuterIter.size());
  assert(correctSize == params.smootherMaxInnerIter.size());
  assert(correctSize == params.kCycleTol.size());
  assert(correctSize == params.kCycleMaxOuterIter.size());
  assert(correctSize == params.kCycleMaxInnerIter.size());
}

void checkParameterValidity(MGTestOtherParams const &params) {

  assert(params.sourceType == "ones" || params.sourceType == "random" || params.sourceType == "gaussian");
}

void checkParameterValidity(MGTestParams const &params) {

  checkParameterValidity(params.mg);
  checkParameterValidity(params.test);
}

template<class Field> void analyseTestVectors(LinearOperatorBase<Field> &Linop, std::vector<Field> const &vectors, int nn) {

  auto positiveOnes = 0;

  std::vector<Field> tmp(4, vectors[0]._grid);

  std::cout << GridLogMessage << "Test vector analysis:" << std::endl;

  for(auto i = 0; i < nn; ++i) {

    Linop.Op(vectors[i], tmp[3]);

    G5C(tmp[0], tmp[3]);

    auto lambda = innerProduct(vectors[i], tmp[0]) / innerProduct(vectors[i], vectors[i]);

    tmp[1] = tmp[0] - lambda * vectors[i];

    auto mu = ::sqrt(norm2(tmp[1]) / norm2(vectors[i]));

    auto nrm = ::sqrt(norm2(vectors[i]));

    if(real(lambda) > 0)
      positiveOnes++;

    std::cout << GridLogMessage << std::scientific << std::setprecision(2) << std::setw(2) << std::showpos << "vector " << i << ": "
              << "singular value: " << lambda << ", singular vector precision: " << mu << ", norm: " << nrm << std::endl;
  }
  std::cout << GridLogMessage << std::scientific << std::setprecision(2) << std::setw(2) << std::showpos << positiveOnes << " out of " << nn
            << " vectors were positive" << std::endl;
  std::cout << std::noshowpos;
}

struct LevelInfo {
public:
  std::vector<std::vector<int>>        Seeds;
  std::vector<GridCartesian *>         Grids;
  std::vector<GridRedBlackCartesian *> RBGrids;
  std::vector<GridParallelRNG>         PRNGs;

  LevelInfo(GridCartesian *FineGrid, MultiGridParams const &mgParams) {

    auto nCoarseLevels = mgParams.blockSizes.size();

    assert(nCoarseLevels == mgParams.nLevels - 1);

    // set up values for finest grid
    Grids.push_back(FineGrid);
    RBGrids.push_back((FineGrid->Nd() == 4) ? QCD::SpaceTimeGrid::makeFourDimRedBlackGrid(Grids.back())
                                            : QCD::SpaceTimeGrid::makeFiveDimRedBlackGrid(1, Grids.back()));
    Seeds.push_back((FineGrid->Nd() == 4) ? std::vector<int>{1, 2, 3, 4} : std::vector<int>{5, 6, 7, 8});
    PRNGs.push_back(GridParallelRNG(Grids.back()));
    PRNGs.back().SeedFixedIntegers(Seeds.back());

    // set up values for coarser grids
    for(int level = 1; level < mgParams.nLevels; ++level) {
      auto Nd               = Grids[level - 1]->Nd();
      auto fullDimensions4d = extract4dData(Grids[level - 1]->FullDimensions());

      assert(Nd == 4 || Nd == 5);
      assert(fullDimensions4d.size() == mgParams.blockSizes[level - 1].size());

      Seeds.push_back(std::vector<int>(4));

      for(int d = 0; d < fullDimensions4d.size(); ++d) {
        fullDimensions4d[d] /= mgParams.blockSizes[level - 1][d];
        Seeds[level][d] = (level)*Nd + d + 1;
      }

      auto simdLayout4d = extract4dData(Grids[level - 1]->_simd_layout);
      auto mpiLayout4d  = extract4dData(Grids[level - 1]->_processors);
      GridCartesian * tmpGrid4d = QCD::SpaceTimeGrid::makeFourDimGrid(fullDimensions4d, simdLayout4d, mpiLayout4d);
      if(Nd == 4) {
        Grids.push_back(tmpGrid4d);
        RBGrids.push_back(QCD::SpaceTimeGrid::makeFourDimRedBlackGrid(tmpGrid4d));
      } else {
        Grids.push_back(QCD::SpaceTimeGrid::makeFiveDimGrid(1, tmpGrid4d));
        RBGrids.push_back(QCD::SpaceTimeGrid::makeFiveDimRedBlackGrid(1, tmpGrid4d));
      }

      PRNGs.push_back(GridParallelRNG(Grids[level]));
      PRNGs[level].SeedFixedIntegers(Seeds[level]);
    }

    std::cout << GridLogMessage << "Constructed " << mgParams.nLevels << " levels" << std::endl;

    for(int level = 0; level < mgParams.nLevels; ++level) {
      std::cout << GridLogMessage << "level = " << level << ":" << std::endl;
      Grids[level]->show_decomposition();
      RBGrids[level]->show_decomposition();
    }
  }

private:
  template<typename T>
  std::vector<T> extract4dData(std::vector<T> const &data) {
    assert(data.size() == 4 || data.size() == 5);
    return std::vector<T>(data.begin() + (data.size() - 4), data.end());
  }
};

template<class Field> class MultiGridPreconditionerBase : public LinearFunction<Field> {
public:
  virtual ~MultiGridPreconditionerBase()               = default;
  virtual void initialSetup()                          = 0;
  virtual void iterativeSetup()                        = 0;
  virtual void operator()(Field const &in, Field &out) = 0;
  virtual void runChecks(RealD tolerance)              = 0;
  virtual void reportTimings()                         = 0;
  virtual void resetTimers()                           = 0;
};

template<class Fobj, class CComplex, int nBasis, int nCoarserLevels, class Matrix>
class MultiGridPreconditioner : public MultiGridPreconditionerBase<Lattice<Fobj>> {
public:
  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  // clang-format off
#if defined(USE_TWOSPIN_COARSENING)
  typedef TwoSpinCoarseningPolicy<Lattice<Fobj>, CComplex, nBasis>                                                    CoarseningPolicy;
  typedef AggregationUsingPolicies<CoarseningPolicy>                                                                  Aggregates;
  typedef CoarsenedMatrixUsingPolicies<CoarseningPolicy>                                                              CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField                                                                    CoarseVector;
  typedef typename CoarseDiracMatrix::SiteSpinor                                                                      CoarseSiteVector;
  typedef Matrix                                                                                                      FineDiracMatrix;
  typedef typename CoarseDiracMatrix::FineFermionField                                                                FineVector;
  typedef MultiGridPreconditioner<CoarseSiteVector, CComplex, nBasis, nCoarserLevels - 1, CoarseDiracMatrix>          NextPreconditionerLevel;
#elif defined (USE_ONESPIN_COARSENING)
  typedef OriginalCoarseningPolicy<Lattice<Fobj>, CComplex, nBasis>                                                   CoarseningPolicy;
  typedef AggregationUsingPolicies<CoarseningPolicy>                                                                  Aggregates;
  typedef CoarsenedMatrixUsingPolicies<CoarseningPolicy>                                                              CoarseDiracMatrix;
  typedef typename CoarseDiracMatrix::FermionField                                                                    CoarseVector;
  typedef typename CoarseDiracMatrix::SiteSpinor                                                                      CoarseSiteVector;
  typedef Matrix                                                                                                      FineDiracMatrix;
  typedef typename CoarseDiracMatrix::FineFermionField                                                                FineVector;
  typedef MultiGridPreconditioner<CoarseSiteVector, iScalar<CComplex>, nBasis, nCoarserLevels - 1, CoarseDiracMatrix> NextPreconditionerLevel;
#else
  typedef Aggregation<Fobj, CComplex, nBasis>                                                                         Aggregates;
  typedef CoarsenedMatrix<Fobj, CComplex, nBasis>                                                                     CoarseDiracMatrix;
  typedef typename Aggregates::CoarseVector                                                                           CoarseVector;
  typedef typename Aggregates::siteVector                                                                             CoarseSiteVector;
  typedef Matrix                                                                                                      FineDiracMatrix;
  typedef typename Aggregates::FineField                                                                              FineVector;
  typedef MultiGridPreconditioner<CoarseSiteVector, iScalar<CComplex>, nBasis, nCoarserLevels - 1, CoarseDiracMatrix> NextPreconditionerLevel;
  #endif
  // clang-format on

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

#if defined(USE_TWOSPIN_COARSENING)
    const int nB = nBasis;
#else
    static_assert((nBasis & 0x1) == 0, "MG Preconditioner only supports an even number of basis vectors");
    const int nB = nBasis / 2;
#endif

  int _CurrentLevel;
  int _NextCoarserLevel;

  MultiGridParams &_MultiGridParams;
  LevelInfo &      _LevelInfo;

  FineDiracMatrix & _FineMatrix;
  FineDiracMatrix & _SmootherMatrix;
  Aggregates        _Aggregates;
  CoarseDiracMatrix _CoarseMatrix;

  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineMdagMOp;
  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineSmootherMdagMOp;

  FineVector _FineSrc;
  FineVector _FineSol;

  std::vector<FineVector> _TmpTestVectors;

  bool _StartSubspaceFromRandom;

  std::unique_ptr<NextPreconditionerLevel> _NextPreconditionerLevel;

  GridStopWatch _InitialSetupTotalTimer;
  GridStopWatch _InitialSetupCreateSubspaceTimer;
  GridStopWatch _InitialSetupCopySubspaceToTmpVectorsTimer;
  GridStopWatch _InitialSetupOrthogonaliseSubspaceTimer;
  GridStopWatch _InitialSetupProjectSubspaceTimer;
  GridStopWatch _InitialSetupProjectToChiralitiesTimer;
  GridStopWatch _InitialSetupCoarsenOperatorTimer;
  GridStopWatch _InitialSetupNextLevelTimer;
  GridStopWatch _IterativeSetupTotalTimer;
  GridStopWatch _IterativeSetupOrthogonaliseTestVectorsTimer;
  GridStopWatch _IterativeSetupOperatorTimer;
  GridStopWatch _IterativeSetupUpdateTestVectorTimer;
  GridStopWatch _IterativeSetupCopyTmpVectorsToSubspace;
  GridStopWatch _IterativeSetupCoarsenOperatorTimer;
  GridStopWatch _IterativeSetupNextLevelTimer;
  GridStopWatch _SolveTotalTimer;
  GridStopWatch _SolveRestrictionTimer;
  GridStopWatch _SolveProlongationTimer;
  GridStopWatch _SolveSmootherTimer;
  GridStopWatch _SolveMiscTimer;
  GridStopWatch _SolveNextLevelTimer;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

  MultiGridPreconditioner(MultiGridParams &mgParams, LevelInfo &LvlInfo, FineDiracMatrix &FineMat, FineDiracMatrix &SmootherMat)
    : _CurrentLevel(mgParams.nLevels - (nCoarserLevels + 1)) // _Level = 0 corresponds to finest
    , _NextCoarserLevel(_CurrentLevel + 1)                   // incremented for instances on coarser levels
    , _MultiGridParams(mgParams)
    , _LevelInfo(LvlInfo)
    , _FineMatrix(FineMat)
    , _SmootherMatrix(SmootherMat)
    , _Aggregates(_LevelInfo.Grids[_NextCoarserLevel], _LevelInfo.Grids[_CurrentLevel], 0)
    , _CoarseMatrix(*_LevelInfo.Grids[_NextCoarserLevel], *_LevelInfo.RBGrids[_NextCoarserLevel])
    , _FineMdagMOp(_FineMatrix)
    , _FineSmootherMdagMOp(_SmootherMatrix)
    , _FineSrc(_LevelInfo.Grids[_CurrentLevel])
    , _FineSol(_LevelInfo.Grids[_CurrentLevel])
    , _TmpTestVectors(nB, _LevelInfo.Grids[_CurrentLevel])
    , _StartSubspaceFromRandom((_CurrentLevel == 0) ? true : _MultiGridParams.startSubspaceFromRandom) {

    if(_StartSubspaceFromRandom)
      std::cout << GridLogMGrid(_CurrentLevel) << "Will be generating Subspace from random vectors" << std::endl;
    else
      std::cout << GridLogMGrid(_CurrentLevel) << "Will be using projected subspace from previous level" << std::endl;

    _NextPreconditionerLevel
      = std::unique_ptr<NextPreconditionerLevel>(new NextPreconditionerLevel(_MultiGridParams, _LevelInfo, _CoarseMatrix, _CoarseMatrix));

    resetTimers();
  }

  void initialSetup() {

    _InitialSetupTotalTimer.Start();

    std::cout << GridLogMGrid(_CurrentLevel) << "Running initial setup phase" << std::endl;

    _InitialSetupCreateSubspaceTimer.Start();
    _Aggregates.CreateSubspaceDDalphaAMG(
      _LevelInfo.PRNGs[_CurrentLevel], _FineSmootherMdagMOp, _StartSubspaceFromRandom, nB, _MultiGridParams.smootherMaxInnerIter[_CurrentLevel]);
    _InitialSetupCreateSubspaceTimer.Stop();

    _InitialSetupCopySubspaceToTmpVectorsTimer.Start();
    copySubspaceToTmpVectors();
    _InitialSetupCopySubspaceToTmpVectorsTimer.Stop();

    _InitialSetupOrthogonaliseSubspaceTimer.Start();
    _Aggregates.Orthogonalise();
    _InitialSetupOrthogonaliseSubspaceTimer.Stop();

    _InitialSetupProjectToChiralitiesTimer.Start();
    _Aggregates.DoChiralDoubling();
    _InitialSetupProjectToChiralitiesTimer.Stop();

    if(!_NextPreconditionerLevel->_StartSubspaceFromRandom) {
      _InitialSetupProjectSubspaceTimer.Start();
      projectSubspaceDownwardsIfNecessary(); // NOTE: function handles internally which levels need to project
      _InitialSetupProjectSubspaceTimer.Stop();
    }

    _InitialSetupCoarsenOperatorTimer.Start();
    _CoarseMatrix.CoarsenOperator(_LevelInfo.Grids[_CurrentLevel], _FineMdagMOp, _Aggregates);
    _InitialSetupCoarsenOperatorTimer.Stop();

    _Aggregates.CheckOrthogonal();

    _InitialSetupNextLevelTimer.Start();
    _NextPreconditionerLevel->initialSetup();
    _InitialSetupNextLevelTimer.Stop();

    _InitialSetupTotalTimer.Stop();
  }

  void iterativeSetup() { // this corresponds to inv_iter_inv_fcycle_P in wuppertal codebase

    if(_MultiGridParams.setupIter[_CurrentLevel] > 0) {
      _IterativeSetupTotalTimer.Start();

      std::cout << GridLogMGrid(_CurrentLevel) << "Running iterative setup phase with " << _MultiGridParams.setupIter[_CurrentLevel] << " iterations" << std::endl;

      for(auto i = 0; i < _MultiGridParams.setupIter[_CurrentLevel]; ++i) {
        std::cout << GridLogMGrid(_CurrentLevel) << "Running setup iteration " << i + 1 << std::endl;

        _IterativeSetupOrthogonaliseTestVectorsTimer.Start();
        orthogonalise(_TmpTestVectors);
        _IterativeSetupOrthogonaliseTestVectorsTimer.Stop();

        for(auto n = 0; n < nB; ++n) {
          _FineSol = zero;

          _IterativeSetupOperatorTimer.Start();
          operator()(_TmpTestVectors[n], _FineSol); // maybe _FineSol needs to be set to zero beforehand
          _IterativeSetupOperatorTimer.Stop();

          _IterativeSetupUpdateTestVectorTimer.Start();
          updateTestVector(n);
          _IterativeSetupUpdateTestVectorTimer.Stop();
        }

        recreateSubspacesAndCoarseOperators(); // profiled inside the function
      }

      _Aggregates.CheckOrthogonal();

      _IterativeSetupNextLevelTimer.Start();
      _NextPreconditionerLevel->iterativeSetup();
      _IterativeSetupNextLevelTimer.Stop();

      resetSolveTimers(); // functions needed in solve also get called in iterative setup which makes profiling output incorrect

      _IterativeSetupTotalTimer.Stop();
    }
  }

  virtual void operator()(FineVector const &in, FineVector &out) {

    conformable(_LevelInfo.Grids[_CurrentLevel], in._grid);
    conformable(in, out);

    // TODO: implement a W-cycle
    if(_MultiGridParams.kCycle)
      kCycle(in, out);
    else
      vCycle(in, out);
  }

  void vCycle(FineVector const &in, FineVector &out) {

    _SolveTotalTimer.Start();

    _SolveMiscTimer.Start();
    RealD inputNorm = norm2(in);

    _NextPreconditionerLevel->_FineSol = zero;

    FineVector fineTmp(in._grid);

    auto maxSmootherIter = _MultiGridParams.smootherMaxOuterIter[_CurrentLevel] * _MultiGridParams.smootherMaxInnerIter[_CurrentLevel];

    GeneralisedMinimalResidual<FineVector> fineGMRES(_MultiGridParams.smootherTol[_CurrentLevel],
                                                     maxSmootherIter,
                                                     _MultiGridParams.smootherMaxInnerIter[_CurrentLevel],
                                                     false);
    _SolveMiscTimer.Stop();

    _SolveRestrictionTimer.Start();
    _Aggregates.ProjectToSubspace(_NextPreconditionerLevel->_FineSrc, in);
    _SolveRestrictionTimer.Stop();

    _SolveNextLevelTimer.Start();
    (*_NextPreconditionerLevel)(_NextPreconditionerLevel->_FineSrc, _NextPreconditionerLevel->_FineSol);
    _SolveNextLevelTimer.Stop();

    _SolveProlongationTimer.Start();
    _Aggregates.PromoteFromSubspace(_NextPreconditionerLevel->_FineSol, out);
    _SolveProlongationTimer.Stop();

    _SolveMiscTimer.Start();
    _FineMdagMOp.Op(out, fineTmp);
    fineTmp                                = in - fineTmp;
    auto r                                 = norm2(fineTmp);
    auto residualAfterCoarseGridCorrection = std::sqrt(r / inputNorm);
    _SolveMiscTimer.Stop();

    _SolveSmootherTimer.Start();
    fineGMRES(_FineSmootherMdagMOp, in, out);
    _SolveSmootherTimer.Stop();

    _SolveMiscTimer.Start();
    _FineMdagMOp.Op(out, fineTmp);
    fineTmp                        = in - fineTmp;
    r                              = norm2(fineTmp);
    auto residualAfterPostSmoother = std::sqrt(r / inputNorm);

    std::cout << GridLogMGrid(_CurrentLevel) << "V-cycle: Input norm = " << std::sqrt(inputNorm)
              << " Coarse residual = " << residualAfterCoarseGridCorrection << " Post-Smoother residual = " << residualAfterPostSmoother
              << std::endl;
    _SolveMiscTimer.Stop();

    _SolveTotalTimer.Stop();
  }

  void kCycle(FineVector const &in, FineVector &out) {

    _SolveTotalTimer.Start();

    _SolveMiscTimer.Start();
    RealD inputNorm = norm2(in);

    _NextPreconditionerLevel->_FineSol = zero;

    FineVector fineTmp(in._grid);

    auto smootherMaxIter = _MultiGridParams.smootherMaxOuterIter[_CurrentLevel] * _MultiGridParams.smootherMaxInnerIter[_CurrentLevel];
    auto kCycleMaxIter   = _MultiGridParams.kCycleMaxOuterIter[_CurrentLevel] * _MultiGridParams.kCycleMaxInnerIter[_CurrentLevel];

    GeneralisedMinimalResidual<FineVector> fineGMRES(_MultiGridParams.smootherTol[_CurrentLevel],
                                                     smootherMaxIter,
                                                     _MultiGridParams.smootherMaxInnerIter[_CurrentLevel],
                                                     false);
    FlexibleGeneralisedMinimalResidual<CoarseVector> coarseFGMRES(_MultiGridParams.kCycleTol[_CurrentLevel],
                                                                  kCycleMaxIter,
                                                                  *_NextPreconditionerLevel,
                                                                  _MultiGridParams.kCycleMaxInnerIter[_CurrentLevel],
                                                                  false);
    _SolveMiscTimer.Stop();

    _SolveRestrictionTimer.Start();
    _Aggregates.ProjectToSubspace(_NextPreconditionerLevel->_FineSrc, in);
    _SolveRestrictionTimer.Stop();

    _SolveNextLevelTimer.Start();
    coarseFGMRES(_NextPreconditionerLevel->_FineMdagMOp, _NextPreconditionerLevel->_FineSrc, _NextPreconditionerLevel->_FineSol);
    _SolveNextLevelTimer.Stop();

    _SolveProlongationTimer.Start();
    _Aggregates.PromoteFromSubspace(_NextPreconditionerLevel->_FineSol, out);
    _SolveProlongationTimer.Stop();

    _SolveMiscTimer.Start();
    _FineMdagMOp.Op(out, fineTmp);
    fineTmp                                = in - fineTmp;
    auto r                                 = norm2(fineTmp);
    auto residualAfterCoarseGridCorrection = std::sqrt(r / inputNorm);
    _SolveMiscTimer.Stop();

    _SolveSmootherTimer.Start();
    fineGMRES(_FineSmootherMdagMOp, in, out);
    _SolveSmootherTimer.Stop();

    _SolveMiscTimer.Start();
    _FineMdagMOp.Op(out, fineTmp);
    fineTmp                        = in - fineTmp;
    r                              = norm2(fineTmp);
    auto residualAfterPostSmoother = std::sqrt(r / inputNorm);

    std::cout << GridLogMGrid(_CurrentLevel) << "K-cycle: Input norm = " << std::sqrt(inputNorm)
              << " Coarse residual = " << residualAfterCoarseGridCorrection << " Post-Smoother residual = " << residualAfterPostSmoother
              << std::endl;
    _SolveMiscTimer.Stop();

    _SolveTotalTimer.Stop();
  }

  void runChecks(RealD tolerance) {

    std::cout << GridLogMGrid(_CurrentLevel) << "Running MG correctness checks" << std::endl;

    std::vector<FineVector>   fineTmps(7, _LevelInfo.Grids[_CurrentLevel]);
    std::vector<CoarseVector> coarseTmps(4, _LevelInfo.Grids[_NextCoarserLevel]);

    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "MG correctness check: 0 == (M - (Mdiag + Σ_μ Mdir_μ)) * v" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_CurrentLevel], fineTmps[0]);

    _FineMdagMOp.Op(fineTmps[0], fineTmps[1]);     //     M * v
    _FineMdagMOp.OpDiag(fineTmps[0], fineTmps[2]); // Mdiag * v

    fineTmps[4] = zero;
    for(int dir = 0; dir < 4; dir++) { //       Σ_μ Mdir_μ * v
      for(auto disp : {+1, -1}) {
        _FineMdagMOp.OpDir(fineTmps[0], fineTmps[3], dir, disp);
        fineTmps[4] = fineTmps[4] + fineTmps[3];
      }
    }

    fineTmps[5] = fineTmps[2] + fineTmps[4]; // (Mdiag + Σ_μ Mdir_μ) * v

    fineTmps[6]    = fineTmps[1] - fineTmps[5];
    auto deviation = std::sqrt(norm2(fineTmps[6]) / norm2(fineTmps[1]));

    std::cout << GridLogMGrid(_CurrentLevel) << "norm2(M * v)                    = " << norm2(fineTmps[1]) << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "norm2(Mdiag * v)                = " << norm2(fineTmps[2]) << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "norm2(Σ_μ Mdir_μ * v)           = " << norm2(fineTmps[4]) << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "norm2((Mdiag + Σ_μ Mdir_μ) * v) = " << norm2(fineTmps[5]) << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "relative deviation              = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "MG correctness check: 0 == (1 - P R) v" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;

    for(auto i = 0; i < _Aggregates.Subspace().size(); ++i) {
      _Aggregates.ProjectToSubspace(coarseTmps[0], _Aggregates.Subspace()[i]); //   R v_i
      _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]);             // P R v_i

      fineTmps[1] = _Aggregates.Subspace()[i] - fineTmps[0]; // v_i - P R v_i
      deviation   = std::sqrt(norm2(fineTmps[1]) / norm2(_Aggregates.Subspace()[i]));

      std::cout << GridLogMGrid(_CurrentLevel) << "Vector " << i << ": norm2(v_i) = " << norm2(_Aggregates.Subspace()[i])
                << " | norm2(R v_i) = " << norm2(coarseTmps[0]) << " | norm2(P R v_i) = " << norm2(fineTmps[0])
                << " | relative deviation = " << deviation;

      if(deviation > tolerance) {
        std::cout << " > " << tolerance << " -> check failed" << std::endl;
        abort();
      } else {
        std::cout << " < " << tolerance << " -> check passed" << std::endl;
      }
    }

    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "MG correctness check: 0 == (1 - R P) v_c" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextCoarserLevel], coarseTmps[0]);

    _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]); //   P v_c
    _Aggregates.ProjectToSubspace(coarseTmps[1], fineTmps[0]);   // R P v_c

    coarseTmps[2] = coarseTmps[0] - coarseTmps[1]; // v_c - R P v_c
    deviation     = std::sqrt(norm2(coarseTmps[2]) / norm2(coarseTmps[0]));

    std::cout << GridLogMGrid(_CurrentLevel) << "norm2(v_c) = " << norm2(coarseTmps[0])
              << " | norm2(R P v_c) = " << norm2(coarseTmps[1]) << " | norm2(P v_c) = " << norm2(fineTmps[0])
              << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "MG correctness check: 0 == (R D P - D_c) v_c" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextCoarserLevel], coarseTmps[0]);

    _Aggregates.PromoteFromSubspace(coarseTmps[0], fineTmps[0]); //     P v_c
    _FineMdagMOp.Op(fineTmps[0], fineTmps[1]);                   //   D P v_c
    _Aggregates.ProjectToSubspace(coarseTmps[1], fineTmps[1]);   // R D P v_c

    _NextPreconditionerLevel->_FineMdagMOp.Op(coarseTmps[0], coarseTmps[2]); // D_c v_c

    coarseTmps[3] = coarseTmps[1] - coarseTmps[2]; // R D P v_c - D_c v_c
    deviation     = std::sqrt(norm2(coarseTmps[3]) / norm2(coarseTmps[1]));

    std::cout << GridLogMGrid(_CurrentLevel) << "norm2(R D P v_c) = " << norm2(coarseTmps[1])
              << " | norm2(D_c v_c) = " << norm2(coarseTmps[2]) << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "MG correctness check: 0 == |(Im(v_c^dag D_c^dag D_c v_c)|" << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "**************************************************" << std::endl;

    random(_LevelInfo.PRNGs[_NextCoarserLevel], coarseTmps[0]);

    _NextPreconditionerLevel->_FineMdagMOp.Op(coarseTmps[0], coarseTmps[1]);    //         D_c v_c
    _NextPreconditionerLevel->_FineMdagMOp.AdjOp(coarseTmps[1], coarseTmps[2]); // D_c^dag D_c v_c

    auto dot  = innerProduct(coarseTmps[0], coarseTmps[2]); //v_c^dag D_c^dag D_c v_c
    deviation = std::abs(imag(dot)) / std::abs(real(dot));

    std::cout << GridLogMGrid(_CurrentLevel) << "Re(v_c^dag D_c^dag D_c v_c) = " << real(dot)
              << " | Im(v_c^dag D_c^dag D_c v_c) = " << imag(dot) << " | relative deviation = " << deviation;

    if(deviation > tolerance) {
      std::cout << " > " << tolerance << " -> check failed" << std::endl;
      abort();
    } else {
      std::cout << " < " << tolerance << " -> check passed" << std::endl;
    }

    _NextPreconditionerLevel->runChecks(tolerance);
  }

  void reportTimings() {

    auto totalSetupTime = _InitialSetupTotalTimer.Elapsed() + _IterativeSetupTotalTimer.Elapsed();
    auto totalTime = totalSetupTime +  _SolveTotalTimer.Elapsed();

    // clang-format off
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Sum   total                            " <<                totalTime                               << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup total                            " <<                totalSetupTime                          << std::endl;

    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial total                    " <<                      _InitialSetupTotalTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial create subspace          " <<             _InitialSetupCreateSubspaceTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial copy subspace to tmp     " <<   _InitialSetupCopySubspaceToTmpVectorsTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial orthogonalise subspace   " <<      _InitialSetupOrthogonaliseSubspaceTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial project subspace down    " <<            _InitialSetupProjectSubspaceTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial project chiral           " <<       _InitialSetupProjectToChiralitiesTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial coarsen operator         " <<            _InitialSetupCoarsenOperatorTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup initial next level               " <<                  _InitialSetupNextLevelTimer.Elapsed() << std::endl;

    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative total                  " <<                    _IterativeSetupTotalTimer.Elapsed() << std::endl;

    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative orthog test vectors    " << _IterativeSetupOrthogonaliseTestVectorsTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative operator               " <<                 _IterativeSetupOperatorTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative update test vector     " <<         _IterativeSetupUpdateTestVectorTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative copy tmp to subspace   " <<      _IterativeSetupCopyTmpVectorsToSubspace.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative coarsen operator       " <<          _IterativeSetupCoarsenOperatorTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Setup iterative next level             " <<                _IterativeSetupNextLevelTimer.Elapsed() << std::endl;

    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve total                            " <<                             _SolveTotalTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve restriction                      " <<                       _SolveRestrictionTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve prolongation                     " <<                      _SolveProlongationTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve smoother                         " <<                          _SolveSmootherTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve misc                             " <<                              _SolveMiscTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve next level                       " <<                         _SolveNextLevelTimer.Elapsed() << std::endl;
    // clang-format on

    _NextPreconditionerLevel->reportTimings();
  }

  void resetInitialSetupTimers() {

    _InitialSetupTotalTimer.Reset();
    _InitialSetupCreateSubspaceTimer.Reset();
    _InitialSetupCopySubspaceToTmpVectorsTimer.Reset();
    _InitialSetupOrthogonaliseSubspaceTimer.Reset();
    _InitialSetupProjectSubspaceTimer.Reset();
    _InitialSetupProjectToChiralitiesTimer.Reset();
    _InitialSetupCoarsenOperatorTimer.Reset();
    _InitialSetupNextLevelTimer.Reset();
  }

  void resetIterativeSetupTimers() {

    _IterativeSetupTotalTimer.Reset();
    _IterativeSetupOrthogonaliseTestVectorsTimer.Reset();
    _IterativeSetupOperatorTimer.Reset();
    _IterativeSetupUpdateTestVectorTimer.Reset();
    _IterativeSetupCopyTmpVectorsToSubspace.Reset();
    _IterativeSetupCoarsenOperatorTimer.Reset();
    _IterativeSetupNextLevelTimer.Reset();
  }

  void resetSolveTimers() {

    _SolveTotalTimer.Reset();
    _SolveRestrictionTimer.Reset();
    _SolveProlongationTimer.Reset();
    _SolveSmootherTimer.Reset();
    _SolveMiscTimer.Reset();
    _SolveNextLevelTimer.Reset();

    _NextPreconditionerLevel->resetSolveTimers();
  }

  void resetTimers() {

    resetInitialSetupTimers();
    resetIterativeSetupTimers();
    resetSolveTimers();

    _NextPreconditionerLevel->resetTimers();
  }

  template<int ncl = nCoarserLevels, typename std::enable_if<(ncl >= 2), int>::type = 0>
  void projectSubspaceDownwardsIfNecessary() {
    std::cout << GridLogMGrid(_CurrentLevel) << "Projecting test vectors to next coarser level" << std::endl;
    for(int n = 0; n < nB; n++) {
      _Aggregates.ProjectToSubspace(_NextPreconditionerLevel->_Aggregates.Subspace()[n], _TmpTestVectors[n]);
    }
  }

  template<int ncl = nCoarserLevels, typename std::enable_if<(ncl <= 1), int>::type = 0>
  void projectSubspaceDownwardsIfNecessary() {
    std::cout << GridLogMGrid(_CurrentLevel) << "NOT projecting test vectors to next coarser level" << std::endl;
  }

  void copySubspaceToTmpVectors() {

    for(int n = 0; n < nB; n++) {
#if defined(USE_TWOSPIN_COARSENING)
      _TmpTestVectors[n] = _Aggregates.Subspace()[n];
#else
      _TmpTestVectors[n] = _Aggregates.Subspace()[n] + _Aggregates.Subspace()[n + nB];
#endif
      std::cout << GridLogMGrid(_CurrentLevel)
                << "Copied subspace vector " << n << " to tmp vectors. "
                << "norm2(vec[" << n << "]) = " << norm2(_TmpTestVectors[n]) << std::endl;
    }
  }

  void copyTmpVectorsToSubspace() {

    for(int n = 0; n < nB; n++) _Aggregates.Subspace()[n] = _TmpTestVectors[n];
    _Aggregates.DoChiralDoubling();

    for(int n = 0; n < nB; n++) {
      std::cout << GridLogMGrid(_CurrentLevel)
                << "Copied tmp vector " << n << " to subspace vectors. "
                << "norm2(vec[" << n << "]) = " << norm2(_Aggregates.Subspace()[n])
#if !defined(USE_TWOSPIN_COARSENING)
                << " norm2(vec[" << n + nB << "]) = " << norm2(_Aggregates.Subspace()[n + nB])
#endif
                << std::endl;
    }
  }

  void updateTestVector(int n) {

    _NextPreconditionerLevel->updateTestVector(n);

    auto scale         = std::pow(norm2(_FineSol), -0.5);
    _TmpTestVectors[n] = scale * _FineSol;
  }

  void recreateSubspacesAndCoarseOperators() {

    _IterativeSetupCopyTmpVectorsToSubspace.Start();
    copyTmpVectorsToSubspace();
    _IterativeSetupCopyTmpVectorsToSubspace.Stop();

    _IterativeSetupCoarsenOperatorTimer.Start();
    _CoarseMatrix.CoarsenOperator(_LevelInfo.Grids[_CurrentLevel],
                                  _FineMdagMOp,
                                  _Aggregates); // reconstruct D_c, this automatically orthogonalizes the aggregates again
    _IterativeSetupCoarsenOperatorTimer.Stop();

    std::cout << GridLogMGrid(_CurrentLevel) << "Recreated intergrid operators and coarse operator" << std::endl;

    _NextPreconditionerLevel->recreateSubspacesAndCoarseOperators();
  }
};

// Specialization for the coarsest level
template<class Fobj, class CComplex, int nBasis, class Matrix>
class MultiGridPreconditioner<Fobj, CComplex, nBasis, 0, Matrix> : public MultiGridPreconditionerBase<Lattice<Fobj>> {
public:
  /////////////////////////////////////////////
  // Type Definitions
  /////////////////////////////////////////////

  typedef Matrix        FineDiracMatrix;
  typedef Lattice<Fobj> FineVector;

  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

  int _CurrentLevel;

  MultiGridParams &_MultiGridParams;
  LevelInfo &      _LevelInfo;

  FineDiracMatrix &_FineMatrix;
  FineDiracMatrix &_SmootherMatrix;

  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineMdagMOp;
  MdagMLinearOperator<FineDiracMatrix, FineVector> _FineSmootherMdagMOp;

  FineVector _FineSrc;
  FineVector _FineSol;

  bool _StartSubspaceFromRandom;

  GridStopWatch _SolveTotalTimer;
  GridStopWatch _SolveSmootherTimer;
  GridStopWatch _SolveMiscTimer;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

  MultiGridPreconditioner(MultiGridParams &mgParams, LevelInfo &LvlInfo, FineDiracMatrix &FineMat, FineDiracMatrix &SmootherMat)
    : _CurrentLevel(mgParams.nLevels - (0 + 1))
    , _MultiGridParams(mgParams)
    , _LevelInfo(LvlInfo)
    , _FineMatrix(FineMat)
    , _SmootherMatrix(SmootherMat)
    , _FineMdagMOp(_FineMatrix)
    , _FineSmootherMdagMOp(_SmootherMatrix)
    , _FineSrc(_LevelInfo.Grids[_CurrentLevel])
    , _FineSol(_LevelInfo.Grids[_CurrentLevel])
    , _StartSubspaceFromRandom(_MultiGridParams.startSubspaceFromRandom) { // Value not important since this level doesn't have a subspace

    std::cout << GridLogMGrid(_CurrentLevel) << "Will not be doing any subspace generation since this is the coarsest level" << std::endl;

    resetTimers();
  }

  void initialSetup() {}

  void iterativeSetup() {}

  void updateTestVector(int n) {}

  void recreateSubspacesAndCoarseOperators() {}

  virtual void operator()(FineVector const &in, FineVector &out) {

    _SolveTotalTimer.Start();

    _SolveMiscTimer.Start();
    conformable(_LevelInfo.Grids[_CurrentLevel], in._grid);
    conformable(in, out);

    auto coarseSolverMaxIter = _MultiGridParams.coarseSolverMaxOuterIter * _MultiGridParams.coarseSolverMaxInnerIter;

    // On the coarsest level we only have what I above call the fine level, no coarse one
    GeneralisedMinimalResidual<FineVector> fineGMRES(
      _MultiGridParams.coarseSolverTol, coarseSolverMaxIter, _MultiGridParams.coarseSolverMaxInnerIter, false);
    SchurRedBlackDiagMooeeNonHermSolve<FineVector> fineRBGMRES(fineGMRES);
    _SolveMiscTimer.Stop();

    _SolveSmootherTimer.Start();
    if(_MultiGridParams.coarseSolverUseRBPrec)
      fineRBGMRES(_FineMatrix, in, out);
    else
      fineGMRES(_FineMdagMOp, in, out);
    _SolveSmootherTimer.Stop();

    _SolveTotalTimer.Stop();
  }

  void runChecks(RealD tolerance) {}

  void reportTimings() {

    // clang-format off
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve total                            " <<    _SolveTotalTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve smoother                         " << _SolveSmootherTimer.Elapsed() << std::endl;
    std::cout << GridLogMGrid(_CurrentLevel) << "Time elapsed: Solve misc                             " <<     _SolveMiscTimer.Elapsed() << std::endl;
    // clang-format on
  }

  void resetSolveTimers() {

    _SolveTotalTimer.Reset();
    _SolveSmootherTimer.Reset();
    _SolveMiscTimer.Reset();
  }

  void resetTimers() {

    resetSolveTimers();
  }
};

template<class Fobj, class CComplex, int nBasis, int nLevels, class Matrix>
using NLevelMGPreconditioner = MultiGridPreconditioner<Fobj, CComplex, nBasis, nLevels - 1, Matrix>;

template<class Fobj, class CComplex, int nBasis, class Matrix>
std::unique_ptr<MultiGridPreconditionerBase<Lattice<Fobj>>>
createMGInstance(MultiGridParams &mgParams, LevelInfo &levelInfo, Matrix &FineMat, Matrix &SmootherMat) {

#define CASE_FOR_N_LEVELS(nLevels)                                                                                     \
  case nLevels:                                                                                                        \
    return std::unique_ptr<NLevelMGPreconditioner<Fobj, CComplex, nBasis, nLevels, Matrix>>(                           \
      new NLevelMGPreconditioner<Fobj, CComplex, nBasis, nLevels, Matrix>(mgParams, levelInfo, FineMat, SmootherMat)); \
    break;

  switch(mgParams.nLevels) {
    CASE_FOR_N_LEVELS(2);
    CASE_FOR_N_LEVELS(3);
    CASE_FOR_N_LEVELS(4);
    default:
      std::cout << GridLogError << "We currently only support nLevels ∈ {2, 3, 4}" << std::endl;
      exit(EXIT_FAILURE);
      break;
  }
#undef CASE_FOR_N_LEVELS
}

}
#endif
