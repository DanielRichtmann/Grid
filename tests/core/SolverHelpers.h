/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/SolverHelpers.h

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

#include "MiscHelpers.h"

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(SolverHelpers);

template<class Matrix, class Field>
class NonHermitianSolverChoice : public LinearFunction<Field> {
private:
  NonHermitianLinearOperator<Matrix, Field> op;
  std::unique_ptr<OperatorFunction<Field>>  slv;

public:
  Matrix&     mat;
  RealD       tolerance;
  int         maxIter;
  int         restartLength;
  bool        useRB;
  std::string type;

  NonHermitianSolverChoice(Matrix&     _mat,
                           RealD       _tolerance,
                           int         _maxIter,
                           int         _restartLength,
                           bool        _useRB,
                           std::string _type)
    : mat(_mat)
    , op(_mat)
    , tolerance(_tolerance)
    , maxIter(_maxIter)
    , restartLength(_restartLength)
    , useRB(_useRB)
    , type(_type) {
    assert(MiscHelpers::element_of(type, {"mr", "gmres", "bicgstab"}));
    if(type == "mr") {
      slv =
        std::unique_ptr<OperatorFunction<Field>>(new MinimalResidual<Field>(tolerance, maxIter, 1.0, false));
    } else if(type == "gmres") {
      slv = std::unique_ptr<OperatorFunction<Field>>(
        new GeneralisedMinimalResidual<Field>(tolerance, maxIter, restartLength, false));
      // } else if(type == "fgmres") {
      //   slv = std::unique_ptr<OperatorFunction<Field>>(new FlexibleGeneralisedMinimalResidual<Field>(
      //     tolerance, maxIter, prec, restartLength, false));
    } else if(type == "bicgstab") {
      slv = std::unique_ptr<OperatorFunction<Field>>(new BiCGSTAB<Field>(tolerance, maxIter, false));
    }
  }

  void operator()(const Field& in, Field& out) {
    if(useRB) {
      std::cout << "NonHermitianSolverChoice begin   RB: " << norm2(in) << " " << norm2(out) << std::endl;
      NonHermitianSchurRedBlackDiagMooeeSolve<Field> slv_rb(*slv, false, true);
      slv_rb(mat, in, out);
      std::cout << "NonHermitianSolverChoice end     RB: " << norm2(in) << " " << norm2(out) << std::endl;
    } else {
      std::cout << "NonHermitianSolverChoice begin NORB: " << norm2(in) << " " << norm2(out) << std::endl;
      (*slv)(op, in, out);
      std::cout << "NonHermitianSolverChoice end   NORB: " << norm2(in) << " " << norm2(out) << std::endl;
    }
  }
};

NAMESPACE_END(SolverHelpers);
NAMESPACE_END(Grid);
