/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/solver/Aggregation.h

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

#pragma once

#include "../core/MiscHelpers.h"

NAMESPACE_BEGIN(Grid);

struct SubspaceParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(SubspaceParams,
                                  int,         npreortho,
                                  int,         npostortho,
                                  std::string, vectorType,
                                  bool,        solverUseEo,
                                  double,      solverTol,
                                  int,         solverMaxIter,
                                  int,         solverRestartLength,
                                  std::string, solverName);
  // constructor with default values
  SubspaceParams()
    : npreortho(0)
    , npostortho(1)
    , vectorType("test")
    , solverUseEo(true)
    , solverTol(1e-6)
    , solverMaxIter(1000)
    , solverRestartLength(20)
    , solverName("cg")
  {}
};


void checkParameterValidity(const SubspaceParams& params) {
  assert(MiscHelpers::element_of(params.vectorType, {"null", "test"}));
  assert(MiscHelpers::element_of(params.solverName, {"cg", "gmres", "bicgstab"}));
}


template<class Field>
void CreateSubspace(GridParallelRNG&           rng,
                    LinearOperatorBase<Field>& hermop,
                    std::vector<Field>&        basis,
                    const SubspaceParams&      params) {
  CreateSubspaceSimple(rng, hermop, basis, params);
}


template<class Field>
void orthogonalize(Field& w, const std::vector<Field>& basis, int n) {
  assert(n <= basis.size());
  for(int i = 0; i < n; i++) {
    auto ip = innerProduct(basis[i], w);
    w       = w - ip * basis[i];
  }
}


template<class Field>
void orthonormalize(std::vector<Field>& basis, int n) {
  assert(n <= basis.size());
  for(int i = 0; i < n; i++) {
    auto& v  = basis[i];
    auto  v2 = norm2(v);
    v        = v * std::pow(v2, -0.5);
    orthogonalize(v, basis, i);
  }
}


template<class Field>
void CreateSubspaceSimple(GridParallelRNG&           rng,
                          LinearOperatorBase<Field>& op,
                          std::vector<Field>&        basis,
                          const SubspaceParams&      params) {
  assert(basis.size() % 2 == 0);
  int nbasis = basis.size();
  int nb     = nbasis / 2;

  GridBase* grid = basis[0].Grid();

  // randomize vectors
  for(int b = 0; b < nb; b++) { gaussian(rng, basis[b]); }
  std::cout << GridLogMessage << "Done randomizing basis vectors" << std::endl;

  // pre-orthonormalize globally
  for(int n = 0; n < params.npreortho; n++) { orthonormalize(basis, nb); }
  std::cout << GridLogMessage << "Done pre-orthonormalizing basis vectors" << std::endl;

  // determine solver
  GeneralisedMinimalResidual<Field> solver(params.solverTol, params.solverMaxIter, params.solverRestartLength, false);
  std::cout << GridLogMessage << "Done setting up solver" << std::endl;

  // find near-null vectors
  Field src(grid); Field psi(grid);
  for(int b = 0; b < nb; b++) {
    auto& v = basis[b];
    if(params.vectorType == "test") {
      psi = Zero();
      src = v;
    } else if(params.vectorType == "null") {
      src = Zero();
      psi = v;
    } else {
      assert(0 && "Wrong vector type");
    }
    solver(op, src, psi);
    v = psi;
    std::cout << GridLogMessage << "Done finding near-null vector " << b << std::endl;
  }

  // post-orthonormalize globally
  for(int n = 0; n < params.npostortho; n++) { orthonormalize(basis, nb); }
  std::cout << GridLogMessage << "Done post-orthonormalizing basis vectors" << std::endl;
}

NAMESPACE_END(Grid);
