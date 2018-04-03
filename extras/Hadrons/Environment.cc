/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid 

Source file: extras/Hadrons/Environment.cc

Copyright (C) 2015-2018

Author: Antonin Portelli <antonin.portelli@me.com>

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

#include <Grid/Hadrons/Environment.hpp>
#include <Grid/Hadrons/Module.hpp>
#include <Grid/Hadrons/ModuleFactory.hpp>

using namespace Grid;
using namespace QCD;
using namespace Hadrons;

#define ERROR_NO_ADDRESS(address)\
HADRON_ERROR(Definition, "no object with address " + std::to_string(address));

/******************************************************************************
 *                       Environment implementation                           *
 ******************************************************************************/
// constructor /////////////////////////////////////////////////////////////////
Environment::Environment(void)
{
    dim_ = GridDefaultLatt();
    nd_  = dim_.size();
    grid4d_.reset(SpaceTimeGrid::makeFourDimGrid(
        dim_, GridDefaultSimd(nd_, vComplex::Nsimd()),
        GridDefaultMpi()));
    gridRb4d_.reset(SpaceTimeGrid::makeFourDimRedBlackGrid(grid4d_.get()));
    auto loc = getGrid()->LocalDimensions();
    locVol_ = 1;
    for (unsigned int d = 0; d < loc.size(); ++d)
    {
        locVol_ *= loc[d];
    }
    rng4d_.reset(new GridParallelRNG(grid4d_.get()));
}

// grids ///////////////////////////////////////////////////////////////////////
void Environment::createGrid(const unsigned int Ls)
{
    if ((Ls > 1) and (grid5d_.find(Ls) == grid5d_.end()))
    {
        auto g = getGrid();
        
        grid5d_[Ls].reset(SpaceTimeGrid::makeFiveDimGrid(Ls, g));
        gridRb5d_[Ls].reset(SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, g));
    }
}

void Environment::createCoarseGrid(const std::vector<int> &blockSize, 
                                   const unsigned int Ls)
{
    int              nd      = getNd();
    std::vector<int> fineDim = getDim(), coarseDim;
    unsigned int     cLs;
    auto             key4d = blockSize, key5d = blockSize;

    createGrid(Ls);
    coarseDim.resize(nd);
    for (int d = 0; d < coarseDim.size(); d++)
    {
        coarseDim[d] = fineDim[d]/blockSize[d];
        if (coarseDim[d]*blockSize[d] != fineDim[d])
        {
            HADRON_ERROR(Size, "Fine dimension " + std::to_string(d) 
                         + " (" + std::to_string(fineDim[d]) 
                         + ") not divisible by coarse dimension ("
                         + std::to_string(coarseDim[d]) + ")"); 
        }
    }
    if (blockSize.size() > nd)
    {
        cLs = Ls/blockSize[nd];
        if (cLs*blockSize[nd] != Ls)
        {
            HADRON_ERROR(Size, "Fine Ls (" + std::to_string(Ls) 
                         + ") not divisible by coarse Ls ("
                         + std::to_string(cLs) + ")");
        }
        key4d.resize(nd);
        key5d.push_back(Ls);
    }
    gridCoarse4d_[key4d].reset(
        SpaceTimeGrid::makeFourDimGrid(coarseDim, 
            GridDefaultSimd(nd, vComplex::Nsimd()), GridDefaultMpi()));
    if (Ls > 1)
    {
        gridCoarse5d_[key5d].reset(
            SpaceTimeGrid::makeFiveDimGrid(cLs, gridCoarse4d_[key4d].get()));
    }
}

GridCartesian * Environment::getGrid(const unsigned int Ls) const
{
    try
    {
        if (Ls == 1)
        {
            return grid4d_.get();
        }
        else
        {
            return grid5d_.at(Ls).get();
        }
    }
    catch(std::out_of_range &)
    {
        HADRON_ERROR(Definition, "no grid with Ls= " + std::to_string(Ls));
    }
}

GridRedBlackCartesian * Environment::getRbGrid(const unsigned int Ls) const
{
    try
    {
        if (Ls == 1)
        {
            return gridRb4d_.get();
        }
        else
        {
            return gridRb5d_.at(Ls).get();
        }
    }
    catch(std::out_of_range &)
    {
        HADRON_ERROR(Definition, "no red-black grid with Ls= " + std::to_string(Ls));
    }
}

GridCartesian * Environment::getCoarseGrid(
    const std::vector<int> &blockSize, const unsigned int Ls) const
{
    auto key = blockSize;

    try
    {
        if (Ls == 1)
        {
            key.resize(getNd());
            return gridCoarse4d_.at(key).get();
        }
        else
        {
            key.push_back(Ls);
            return gridCoarse5d_.at(key).get();
        }
    }
    catch(std::out_of_range &)
    {
        HADRON_ERROR(Definition, "no coarse grid with Ls= " + std::to_string(Ls));
    }
}

unsigned int Environment::getNd(void) const
{
    return nd_;
}

std::vector<int> Environment::getDim(void) const
{
    return dim_;
}

int Environment::getDim(const unsigned int mu) const
{
    return dim_[mu];
}

unsigned long int Environment::getLocalVolume(void) const
{
    return locVol_;
}

// random number generator /////////////////////////////////////////////////////
void Environment::setSeed(const std::vector<int> &seed)
{
    rng4d_->SeedFixedIntegers(seed);
}

GridParallelRNG * Environment::get4dRng(void) const
{
    return rng4d_.get();
}

// general memory management ///////////////////////////////////////////////////
void Environment::addObject(const std::string name, const int moduleAddress)
{
    if (!hasObject(name))
    {
        ObjInfo info;
        
        info.name   = name;
        info.module = moduleAddress;
        info.data   = nullptr;
        object_.push_back(std::move(info));
        objectAddress_[name] = static_cast<unsigned int>(object_.size() - 1);
    }
    else
    {
        HADRON_ERROR(Definition, "object '" + name + "' already exists");
    }
}

void Environment::setObjectModule(const unsigned int objAddress,
                                  const int modAddress)
{
    object_[objAddress].module = modAddress;
}

unsigned int Environment::getMaxAddress(void) const
{
    return object_.size();
}

unsigned int Environment::getObjectAddress(const std::string name) const
{
    if (hasObject(name))
    {
        return objectAddress_.at(name);
    }
    else
    {
        HADRON_ERROR(Definition, "no object with name '" + name + "'");
    }
}

std::string Environment::getObjectName(const unsigned int address) const
{
    if (hasObject(address))
    {
        return object_[address].name;
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

std::string Environment::getObjectType(const unsigned int address) const
{
    if (hasObject(address))
    {
        if (object_[address].type)
        {
            return typeName(object_[address].type);
        }
        else
        {
            return "<no type>";
        }
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

std::string Environment::getObjectType(const std::string name) const
{
    return getObjectType(getObjectAddress(name));
}

Environment::Size Environment::getObjectSize(const unsigned int address) const
{
    if (hasObject(address))
    {
        return object_[address].size;
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

Environment::Size Environment::getObjectSize(const std::string name) const
{
    return getObjectSize(getObjectAddress(name));
}

Environment::Storage Environment::getObjectStorage(const unsigned int address) const
{
    if (hasObject(address))
    {
        return object_[address].storage;
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

Environment::Storage Environment::getObjectStorage(const std::string name) const
{
    return getObjectStorage(getObjectAddress(name));
}

int Environment::getObjectModule(const unsigned int address) const
{
    if (hasObject(address))
    {
        return object_[address].module;
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

int Environment::getObjectModule(const std::string name) const
{
    return getObjectModule(getObjectAddress(name));
}

unsigned int Environment::getObjectLs(const unsigned int address) const
{
    if (hasCreatedObject(address))
    {
        return object_[address].Ls;
    }
    else
    {
        ERROR_NO_ADDRESS(address);
    }
}

unsigned int Environment::getObjectLs(const std::string name) const
{
    return getObjectLs(getObjectAddress(name));
}

bool Environment::hasObject(const unsigned int address) const
{
    return (address < object_.size());
}

bool Environment::hasObject(const std::string name) const
{
    auto it = objectAddress_.find(name);
    
    return ((it != objectAddress_.end()) and hasObject(it->second));
}

bool Environment::hasCreatedObject(const unsigned int address) const
{
    if (hasObject(address))
    {
        return (object_[address].data != nullptr);
    }
    else
    {
        return false;
    }
}

bool Environment::hasCreatedObject(const std::string name) const
{
    if (hasObject(name))
    {
        return hasCreatedObject(getObjectAddress(name));
    }
    else
    {
        return false;
    }
}

bool Environment::isObject5d(const unsigned int address) const
{
    return (getObjectLs(address) > 1);
}

bool Environment::isObject5d(const std::string name) const
{
    return (getObjectLs(name) > 1);
}

Environment::Size Environment::getTotalSize(void) const
{
    Environment::Size size = 0;
    
    for (auto &o: object_)
    {
        size += o.size;
    }
    
    return size;
}

void Environment::freeObject(const unsigned int address)
{
    if (hasCreatedObject(address))
    {
        LOG(Message) << "Destroying object '" << object_[address].name
                     << "'" << std::endl;
    }
    object_[address].size = 0;
    object_[address].type = nullptr;
    object_[address].data.reset(nullptr);
}

void Environment::freeObject(const std::string name)
{
    freeObject(getObjectAddress(name));
}

void Environment::freeAll(void)
{
    for (unsigned int i = 0; i < object_.size(); ++i)
    {
        freeObject(i);
    }
}

void Environment::protectObjects(const bool protect)
{
    protect_ = protect;
}

bool Environment::objectsProtected(void) const
{
    return protect_;
}

// print environment content ///////////////////////////////////////////////////
void Environment::printContent(void) const
{
    LOG(Debug) << "Objects: " << std::endl;
    for (unsigned int i = 0; i < object_.size(); ++i)
    {
        LOG(Debug) << std::setw(4) << i << ": "
                   << getObjectName(i) << " ("
                   << sizeString(getObjectSize(i)) << ")" << std::endl;
    }
}
