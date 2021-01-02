/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_create_grid.cc

    Copyright (C) 2015 - 2020

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

#include <Grid/Grid.h>

using namespace Grid;


#define grid_message std::cout << GridLogMessage
#define grid_printf(...) \
{\
  char _buf[1024];\
  sprintf(_buf, __VA_ARGS__);\
  std::cout << GridLogMessage << _buf;\
}


int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate mpi = GridDefaultMpi();
  Coordinate simd_d = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate simd_f = GridDefaultSimd(Nd, vComplexF::Nsimd());

  grid_message << "Latt   = " << latt << std::endl;
  grid_message << "MPI    = " << mpi << std::endl;
  grid_message << "Simd_d = " << simd_d << std::endl;
  grid_message << "Simd_f = " << simd_f << std::endl;

  GridCartesian*         UGrid_d   = SpaceTimeGrid::makeFourDimGrid(latt, simd_d, mpi);
  grid_message << "Succesfully created a full grid in double precision. Decomposition:" << std::endl;
  UGrid_d->show_decomposition();

  GridRedBlackCartesian* UrbGrid_d = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_d);
  grid_message << "Succesfully created a rb grid in double precision. Decomposition:" << std::endl;
  UrbGrid_d->show_decomposition();

  GridCartesian*         UGrid_f   = SpaceTimeGrid::makeFourDimGrid(latt, simd_f, mpi);
  grid_message << "Succesfully created a full grid in single precision. Decomposition:" << std::endl;
  UGrid_f->show_decomposition();

  GridRedBlackCartesian* UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  grid_message << "Succesfully created a rb grid in single precision. Decomposition:" << std::endl;
  UrbGrid_f->show_decomposition();

  Grid_finalize();
}
