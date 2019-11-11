/*************************************************************************************
 
 Grid physics library, www.github.com/paboyle/Grid
 
 Source file: Hadrons/Modules/MDistil/PerambFromSolve.hpp
 
 Copyright (C) 2019
 
 Author: Felix Erben <ferben@ed.ac.uk>
 Author: Michael Marshall <Michael.Marshall@ed.ac.uk>
 
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

#ifndef Hadrons_MDistil_PerambFromSolve_hpp_
#define Hadrons_MDistil_PerambFromSolve_hpp_

#include <Hadrons/Modules/MDistil/DistilCommon.hpp>

BEGIN_HADRONS_NAMESPACE
BEGIN_MODULE_NAMESPACE(MDistil)

/******************************************************************************
 *                         PerambFromSolve 

  This module computes a perambulator from an already completed solve. 
  Optionally, the number of eigenvectors used in the perambulator and the 
  parameter LI can be chosen to be lower than the ones in the solve, allowing 
  for a study of the signal with different values of nvec.
 ******************************************************************************/

class PerambFromSolvePar: Serializable
{
public:
    GRID_SERIALIZABLE_CLASS_MEMBERS(PerambFromSolvePar,
                                    std::string, eigenPack,
                                    std::string, PerambFileName,
                                    std::string, solve,
                                    std::string, nvec_reduced,
                                    std::string, LI_reduced,
                                    MDistil::DistilParameters, DistilPar);
};

template <typename FImpl>
class TPerambFromSolve: public Module<PerambFromSolvePar>
{
public:
    FERM_TYPE_ALIASES(FImpl,);
    // constructor
    TPerambFromSolve(const std::string name);
    // destructor
    virtual ~TPerambFromSolve(void);
    // dependency relation
    virtual std::vector<std::string> getInput(void);
    virtual std::vector<std::string> getOutput(void);
    // setup
    virtual void setup(void);
    // execution
    virtual void execute(void);
protected:
    GridCartesian * grid3d; // Owned by me, so I must delete it
    GridCartesian * grid4d;
protected:
    virtual void Cleanup(void);
    
};

MODULE_REGISTER_TMP(PerambFromSolve, TPerambFromSolve<FIMPL>, MDistil);

/******************************************************************************
 *                 TPerambFromSolve implementation                             *
 ******************************************************************************/
// constructor /////////////////////////////////////////////////////////////////
template <typename FImpl>
TPerambFromSolve<FImpl>::TPerambFromSolve(const std::string name)
:grid3d{nullptr}, grid4d{nullptr}, Module<PerambFromSolvePar>(name)
{}
//destructor
template <typename FImpl>
TPerambFromSolve<FImpl>::~TPerambFromSolve(void)
{
    Cleanup();
};

// dependencies/products ///////////////////////////////////////////////////////
template <typename FImpl>
std::vector<std::string> TPerambFromSolve<FImpl>::getInput(void)
{
    return std::vector<std::string>{ par().solve, par().eigenPack };
}

template <typename FImpl>
std::vector<std::string> TPerambFromSolve<FImpl>::getOutput(void)
{
    return std::vector<std::string>{ getName() };
}

// setup ///////////////////////////////////////////////////////////////////////
template <typename FImpl>
void TPerambFromSolve<FImpl>::setup(void)
{
    Cleanup();
    const int Nt{env().getDim(Tdir)}; 
    const int nvec{par().DistilPar.nvec}; 
    const int nnoise{par().DistilPar.nnoise}; 
    const int LI{par().DistilPar.LI}; 
    const int TI{par().DistilPar.TI}; 
    const int SI{par().DistilPar.SI}; 
    const bool full_tdil{ TI == Nt }; 
    const int Nt_inv{ full_tdil ? 1 : TI };
    const int nvec_reduced{ par().nvec_reduced.empty() ? nvec:std::stoi(par().nvec_reduced)};
    const int LI_reduced{ par().LI_reduced.empty() ? LI:std::stoi(par().LI_reduced)};
    grid4d = env().getGrid();
    grid3d = MakeLowerDimGrid(grid4d);
    envCreate(PerambTensor, getName(), 1, Nt,nvec_reduced,LI_reduced,nnoise,Nt_inv,SI);
    envCreate(NoiseTensor, getName() + "_noise", 1, nnoise, Nt, nvec, Ns );
    envTmp(LatticeColourVector, "result3d_nospin",1,LatticeColourVector(grid3d));
    envTmp(LatticeColourVector, "evec3d",1,LatticeColourVector(grid3d));
    envTmpLat(LatticeColourVector, "result4d_nospin");
}

template <typename FImpl>
void TPerambFromSolve<FImpl>::Cleanup(void)
{
    if (grid3d != nullptr)
    {
        delete grid3d;
        grid3d = nullptr;
    }
    grid4d = nullptr;
}

// execution ///////////////////////////////////////////////////////////////////
template <typename FImpl>
void TPerambFromSolve<FImpl>::execute(void)
{
    GridCartesian * grid4d = env().getGrid();
    const int Ntlocal{grid4d->LocalDimensions()[3]};
    const int Ntfirst{grid4d->LocalStarts()[3]};
    const int Nt{env().getDim(Tdir)}; 
    const int nvec{par().DistilPar.nvec}; 
    const int nnoise{par().DistilPar.nnoise}; 
    const int TI{par().DistilPar.TI}; 
    const int LI{par().DistilPar.LI}; 
    const int SI{par().DistilPar.SI}; 
    const bool full_tdil{ TI == Nt }; 
    const int Nt_inv{ full_tdil ? 1 : TI };
    const int nvec_reduced{ par().nvec_reduced.empty() ? nvec:std::stoi(par().nvec_reduced)};
    const int LI_reduced{ par().LI_reduced.empty() ? LI:std::stoi(par().LI_reduced)};
    auto &perambulator  = envGet(PerambTensor, getName());
    auto &solve         = envGet(std::vector<FermionField>, par().solve);
    auto &epack         = envGet(Grid::Hadrons::EigenPack<LatticeColourVector>, par().eigenPack);
    
    envGetTmp(LatticeColourVector, result4d_nospin);
    envGetTmp(LatticeColourVector, result3d_nospin);
    envGetTmp(LatticeColourVector, evec3d);
    
    for (int inoise = 0; inoise < nnoise; inoise++)
    {
        for (int dk = 0; dk < LI_reduced; dk++)
       	{
            for (int dt = 0; dt < Nt_inv; dt++)
	    {
                for (int ds = 0; ds < SI; ds++)
	       	{
                    for (int is = 0; is < Ns; is++)
		    {
                        result4d_nospin = peekSpin(solve[inoise+nnoise*(dk+LI*(dt+Nt_inv*ds))],is);
                        for (int t = Ntfirst; t < Ntfirst + Ntlocal; t++)
		       	{
                            ExtractSliceLocal(result3d_nospin,result4d_nospin,0,t-Ntfirst,Tdir);
                            for (int ivec = 0; ivec < nvec_reduced; ivec++)
			    {
                                ExtractSliceLocal(evec3d,epack.evec[ivec],0,t-Ntfirst,Tdir);
                                pokeSpin(perambulator.tensor(t, ivec, dk, inoise,dt,ds),static_cast<Complex>(innerProduct(evec3d, result3d_nospin)),is);
                                LOG(Message) <<  "perambulator(t, ivec, dk, inoise,dt,ds)(is) = (" << t << "," << ivec << "," << dk << "," << inoise << "," << dt << "," << ds << ")(" << is << ") = " <<  perambulator.tensor(t, ivec, dk, inoise,dt,ds)()(is)() << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    if(grid4d->IsBoss())
    {
        std::string sPerambName{par().PerambFileName};
        sPerambName.append( "." );
        sPerambName.append( std::to_string(vm().getTrajectory()));
        perambulator.write(sPerambName.c_str());
    }
}

END_MODULE_NAMESPACE
END_HADRONS_NAMESPACE
#endif // Hadrons_MDistil_PerambFromSolve_hpp_
