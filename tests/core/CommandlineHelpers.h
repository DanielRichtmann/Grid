/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/CommandlineHelpers.h

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
NAMESPACE_BEGIN(CommandlineHelpers);

int readInt(int* argc, char*** argv, std::string&& option, int defaultValue) {
  std::string arg;
  int         ret = defaultValue;
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionInt(arg, ret);
  }
  return ret;
}


std::vector<int> readIvec(int*                    argc,
                          char***                 argv,
                          std::string&&           option,
                          const std::vector<int>& defaultValue) {
  std::string      arg;
  std::vector<int> ret(defaultValue);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    GridCmdOptionIntVector(arg, ret);
  }
  return ret;
}


template<class Reader,
         class Params,
         typename std::enable_if<std::is_base_of<Serializable, Params>::value, void>::type* = nullptr>
Params readParameterFile(int* argc, char*** argv, std::string&& option, const Params& defaultValue) {
  std::string arg;
  Params      ret(defaultValue);
  if(GridCmdOptionExists(*argv, *argv + *argc, option)) {
    arg = GridCmdOptionPayload(*argv, *argv + *argc, option);
    assert(arg.length() != 0);
    Reader reader(arg);
    read(reader, "Params", ret);
    std::cout << GridLogMessage << "Read in params from file " << arg << std::endl;
  } else {
    std::cout << GridLogMessage << "Using default params since nothing passed on command line" << arg << std::endl;
  }
  std::cout << ret << std::endl;
  checkParameterValidity(ret);
  return ret;
}


NAMESPACE_END(CommandlineHelpers);
NAMESPACE_END(Grid);
