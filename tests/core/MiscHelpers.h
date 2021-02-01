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

NAMESPACE_END(MiscHelpers);
NAMESPACE_END(Grid);
