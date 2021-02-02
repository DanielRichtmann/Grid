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
class SolverChoice : public LinearFunction<Field> {
private:
  std::unique_ptr<OperatorFunction<Field>> slv;

public:
  Matrix& mat;
  RealD tolerance;
  int maxIter;
  int restartLength;
  std::string type;
  // LinearFunction<Field>& prec;


  SolverChoice(Matrix&                _mat,
               RealD                  _tolerance,
               int                    _maxIter,
               int                    _restartLength,
               std::string            _type)
               // LinearFunction<Field>& _prec = TrivialPrecon<Field>())
    : mat(_mat)
    , tolerance(_tolerance)
    , maxIter(_maxIter)
    , restartLength(_restartLength)
    , type(_type)
    // , prec(_prec)
  {
    assert(MiscHelpers::element_of(type, {"mr", "gmres", "bicgstab"}));
    if(type == "mr") {
      slv = std::unique_ptr<OperatorFunction<Field>>(
        new MinimalResidual<Field>(tolerance, maxIter, 1.0, false));
    } else if(type == "gmres") {
      slv = std::unique_ptr<OperatorFunction<Field>>(
        new GeneralisedMinimalResidual<Field>(tolerance, maxIter, restartLength, false));
    // } else if(type == "fgmres") {
    //   slv = std::unique_ptr<OperatorFunction<Field>>(new FlexibleGeneralisedMinimalResidual<Field>(
    //     tolerance, maxIter, prec, restartLength, false));
    } else if(type == "bicgstab") {
      slv = std::unique_ptr<OperatorFunction<Field>>(
        new BiCGSTAB<Field>(tolerance, maxIter, false));
    }
  }

  void operator()(const Field& in, Field& out) {
    std::cout << "SolverChoice: " << norm2(in) << " " << norm2(out) << std::endl;
    (*slv)(mat, in, out);
    std::cout << "SolverChoice: " << norm2(in) << " " << norm2(out) << std::endl;
  }
};

NAMESPACE_END(SolverHelpers);
NAMESPACE_END(Grid);
