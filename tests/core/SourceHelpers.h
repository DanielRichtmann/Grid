/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/SourceHelpers.h

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
NAMESPACE_BEGIN(SourceHelpers);

struct SourceParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(SourceParams,
                                  std::string,      type,
                                  std::vector<int>, coor);
  SourceParams()
    : type("point")
    , coor({0,0,0,2})
  {}
};


void checkParameterValidity(SourceParams const& params) {
  assert(MiscHelpers::element_of(params.type, {"point", "guassian", "random", "bernoulli"}));
  assert(params.coor.size() == Nd);
}


template<typename Field>
void createSource(GridParallelRNG& rng, const SourceParams& params, Field& src) {
  checkParameterValidity(params);
  if(params.type == "point") {
    src = Zero();
    typename Field::scalar_object srcSite; srcSite = 1.;
    pokeSite(srcSite, src, params.coor);
    std::cout << GridLogMessage << "Created point source at " << params.coor << std::endl;
  } else if(params.type == "gaussian") {
    gaussian(rng, src);
    std::cout << GridLogMessage << "Created gaussian source" << std::endl;
  } else if(params.type == "random") {
    random(rng, src);
    std::cout << GridLogMessage << "Created random source" << std::endl;
  } else if(params.type == "bernoulli") {
    bernoulli(rng, src);
    std::cout << GridLogMessage << "Created bernoulli source" << std::endl;
  }
}

NAMESPACE_END(SourceHelpers);
NAMESPACE_END(Grid);
