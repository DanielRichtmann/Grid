/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/solver/convert_lime_to_nersc.cc

    Copyright (C) 2015-2020

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

#include <Grid/Grid.h>
#include <Test_multigrid_common.h>

using namespace std;
using namespace Grid;

int main(int argc, char** argv) {

  Grid_init(&argc, &argv);

  GridCartesian* Grid = SpaceTimeGrid::makeFourDimGrid(
    GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());

  LatticeGaugeField Umu(Grid);

  assert(GridCmdOptionExists(argv, argv + argc, "--in"));
  assert(GridCmdOptionExists(argv, argv + argc, "--out"));

  {
    std::string config_lime = GridCmdOptionPayload(argv, argv + argc, "--in");
    assert(config_lime.length() != 0);
    FieldMetaData header;
    IldgReader    reader;
    reader.open(config_lime);
    reader.readConfiguration(Umu, header);
    reader.close();
  }

  {
    std::string config_nersc = GridCmdOptionPayload(argv, argv + argc, "--out");
    assert(config_nersc.length() != 0);
    FieldMetaData header;
    NerscIO::writeConfiguration(Umu, config_nersc, 0, 0);
  }

  Grid_finalize();
}
