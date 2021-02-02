/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/MiscHelpers.h

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

NAMESPACE_BEGIN(Grid);
NAMESPACE_BEGIN(MiscHelpers);

template<class T>
bool element_of(const T& needle, const std::vector<T>& haystack) {
  return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
}


RealD kappaFromMass(RealD mass)  { return 0.5 / (Nd + mass); }
RealD massFromKappa(RealD kappa) { return 1. / (2. * kappa) - Nd; }


bool gridsCompatible(GridBase* foo, GridBase* bar) {
  assert(Nd == 4);
  bool ret = true;

  ret = ret && foo->_fdimensions[0] == bar->_fdimensions[0];
  ret = ret && foo->_fdimensions[1] == bar->_fdimensions[1];
  ret = ret && foo->_fdimensions[2] == bar->_fdimensions[2];
  ret = ret && foo->_fdimensions[3] == bar->_fdimensions[3];

  ret = ret && foo->_gdimensions[0] == bar->_gdimensions[0];
  ret = ret && foo->_gdimensions[1] == bar->_gdimensions[1];
  ret = ret && foo->_gdimensions[2] == bar->_gdimensions[2];
  ret = ret && foo->_gdimensions[3] == bar->_gdimensions[3];

  ret = ret && foo->_ldimensions[0] == bar->_ldimensions[0];
  ret = ret && foo->_ldimensions[1] == bar->_ldimensions[1];
  ret = ret && foo->_ldimensions[2] == bar->_ldimensions[2];
  ret = ret && foo->_ldimensions[3] == bar->_ldimensions[3];

  ret = ret && foo->_rdimensions[0] == bar->_rdimensions[0];
  ret = ret && foo->_rdimensions[1] == bar->_rdimensions[1];
  ret = ret && foo->_rdimensions[2] == bar->_rdimensions[2];
  ret = ret && foo->_rdimensions[3] == bar->_rdimensions[3];

  ret = ret && foo->_isCheckerBoarded == bar->_isCheckerBoarded;

  return ret;
}

NAMESPACE_END(MiscHelpers);
NAMESPACE_END(Grid);
