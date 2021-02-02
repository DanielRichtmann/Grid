/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/ReadHelpers.h

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
NAMESPACE_BEGIN(ReadHelpers);

struct ConfigParams : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(ConfigParams,
                                  RealD, kappa,
                                  RealD, csw,
                                  std::string, config,
                                  std::string, filetype,
                                  std::vector<RealD>, boundary_phases);
  ConfigParams()
    : kappa(0.1)
    , csw(1.0)
    , config("./dummy_path")
    , filetype("random")
    , boundary_phases({1.0, 1.0, 1.0, -1.0})
  {}
};


void checkParameterValidity(ConfigParams const &params) {
  assert(params.kappa > 0.0);
  assert(params.csw >= 0.0);
  assert(MiscHelpers::element_of(params.filetype, {"random", "nersc", "openqcd"}));
}


void readLattSize(const std::string& filetype, const std::string& config, Coordinate& fdimensions) {
  assert(config.length() != 0);
  assert(Nd == 4);
  if(filetype == "nersc") {
    std::map<std::string,std::string> fields;
    std::string ln;
    std::ifstream f(config);
    getline(f,ln); removeWhitespace(ln); assert(ln == "BEGIN_HEADER");
    do {
      getline(f,ln); removeWhitespace(ln);
      int i = ln.find("=");
      if(i>0) {
        auto k=ln.substr(0,i); removeWhitespace(k);
        auto v=ln.substr(i+1); removeWhitespace(v);
        fields[k] = v;
      }
    } while(ln != "END_HEADER");
    fdimensions[0] = atoi(fields["DIMENSION_1"].c_str());
    fdimensions[1] = atoi(fields["DIMENSION_2"].c_str());
    fdimensions[2] = atoi(fields["DIMENSION_3"].c_str());
    fdimensions[3] = atoi(fields["DIMENSION_4"].c_str());
  } else if(filetype == "openqcd") {
    FILE* f = fopen(config.c_str(),"rb");
    if(!f) assert(0 && "Openqcd: File not readable");
    OpenQcdHeader header;
    if(fread(&header,sizeof(header),1,f)!=1) assert(0 && "Openqcd: File header wrong");
#define CHECK_EXTENT(L) assert(L > 1 && L <= 10000);
    CHECK_EXTENT(header.Nx);
    CHECK_EXTENT(header.Ny);
    CHECK_EXTENT(header.Nz);
    CHECK_EXTENT(header.Nt);
#undef CHECK_EXTENT
    assert(header.plaq >= -100.0 && header.plaq <= 100.0);
    fdimensions[0] = header.Nx;
    fdimensions[1] = header.Ny;
    fdimensions[2] = header.Nz;
    fdimensions[3] = header.Nt;
  }
}


GridCartesian* gridFromFile(const std::string& filetype, const std::string& config) {
  Coordinate fdimensions = GridDefaultLatt();
  readLattSize(filetype, config, fdimensions);
  return SpaceTimeGrid::makeFourDimGrid(
    fdimensions, GridDefaultSimd(4, vComplexD::Nsimd()), GridDefaultMpi());
}


void readConfiguration(const std::string& filetype, const std::string& config, LatticeGaugeFieldD& Umu) {
  GridCartesian* grid = gridFromFile(filetype, config);
  // NOTE: cannot do conformable since this requires pointer is same
  assert(MiscHelpers::gridsCompatible(grid, Umu.Grid()));
  FieldMetaData header;
  if(filetype == "nersc") {
    NerscIO::readConfiguration(Umu, header, config);
  } else if(filetype == "openqcd") {
    OpenQcdIO::readConfiguration(Umu, header, config);
  } else {
    std::cout << GridLogMessage << "Did not read anything because filetype is " << filetype << std::endl;
  }
}

NAMESPACE_END(ReadHelpers);
NAMESPACE_END(Grid);
