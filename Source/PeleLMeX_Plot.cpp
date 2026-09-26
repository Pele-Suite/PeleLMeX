#include <PeleLMeX.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_AsyncOut.H>
#include <AMReX_buildInfo.H>
#ifdef PELE_USE_CMAKE
#include <PeleGitHashes.H>
#endif
#include "PelePhysics.H"
#include <PltFileManager.H>
#include <AMReX_ParmParse.H>
#include <PeleLMeX_BCfill.H>
#include <AMReX_FillPatchUtil.H>
#include <memory>
#include <fstream>
#include <sundials/sundials_version.h>
#ifdef AMREX_USE_HIP
#include <hip/hip_version.h>
#endif
#ifdef AMREX_USE_HYPRE
#include <HYPRE_config.h>
#endif
#ifdef PELE_USE_KLU
#include <klu.h>
#endif
#ifdef PELE_USE_MAGMA
#include <magma_v2.h>
#endif
#ifdef AMREX_USE_EB
#include <AMReX_EBInterpolater.H>
#endif

#ifdef PELE_USE_SPRAY
#include "SprayParticles.H"
#endif
#ifdef PELE_USE_SOOT
#include "SootModel.H"
#endif
#ifdef PELE_USE_RADIATION
#include "PeleLMRad.H"
#endif

namespace m2c = pele::physics::utilities::mks2cgs;
namespace c2m = pele::physics::utilities::cgs2mks;

namespace {
const std::string level_prefix{"Level_"};
}

void
GotoNextLine(std::istream& is)
{
  constexpr std::streamsize bl_ignore_max{100000};
  is.ignore(bl_ignore_max, '\n');
}

namespace {

// Write the AMReX "standard" plotfile Header, then append the trailer
// the AMReX ParaView/VisIt reader recognises: one extra vector field of
// AMREX_SPACEDIM components holding the per-node displacement from Xi-
// space to physical space.  Mirrors ERF's WriteGenericPlotfileHeader-
// WithTerrain (Source/IO/ERF_Plotfile.cpp) but built on top of AMReX's
// public WriteGenericPlotfileHeader so we don't duplicate the standard
// header body.
void
writePlotfileHeaderWithMapping(
  std::ostream& HeaderFile,
  int nlevels,
  const amrex::Vector<amrex::BoxArray>& bArrays,
  const amrex::Vector<std::string>& varnames,
  const amrex::Vector<amrex::Geometry>& geom,
  amrex::Real time,
  const amrex::Vector<int>& level_steps,
  const amrex::Vector<amrex::IntVect>& ref_ratio,
  const std::string& versionName,
  const std::string& levelPrefix,
  const std::string& mfPrefix,
  const std::string& mfNodalPrefix)
{
  amrex::WriteGenericPlotfileHeader(
    HeaderFile, nlevels, bArrays, varnames, geom, time, level_steps, ref_ratio,
    versionName, levelPrefix, mfPrefix);

  // Magic trailer: one extra vector field of AMREX_SPACEDIM components
  // named nu_{x,y,z}.  The AMReX ParaView/VisIt plugin treats
  // this as a node-centred displacement and renders the solution on
  // (Xi + nu) rather than on Xi, yielding physical-space visualization
  // automatically.
  HeaderFile << 1 << '\n';
  HeaderFile << AMREX_SPACEDIM << '\n';
  HeaderFile << "nu_x" << '\n';
#if (AMREX_SPACEDIM >= 2)
  HeaderFile << "nu_y" << '\n';
#endif
#if (AMREX_SPACEDIM == 3)
  HeaderFile << "nu_z" << '\n';
#endif
  for (int lev = 0; lev < nlevels; ++lev) {
    HeaderFile << amrex::MultiFabHeaderPath(lev, levelPrefix, mfNodalPrefix)
               << '\n';
  }
}

// Write a multi-level plotfile in the AMReX native format, extended with
// a per-level nodal-displacement MultiFab that tags the plotfile as
// mesh-mapped.  The cell-centered data is written under MultiFab prefix
// "Cell" and the nodal displacement under "Nu_nd"; the extra handshake
// lines (see writePlotfileHeaderWithMapping) go at the end of Header.
void
writePlotfileWithMapping(
  const std::string& plotfilename,
  int nlevels,
  const amrex::Vector<const amrex::MultiFab*>& mf_cc,
  const amrex::Vector<const amrex::MultiFab*>& mf_nd,
  const amrex::Vector<std::string>& varnames,
  const amrex::Vector<amrex::Geometry>& geom,
  amrex::Real time,
  const amrex::Vector<int>& level_steps,
  const amrex::Vector<amrex::IntVect>& ref_ratio,
  const std::string& versionName = "HyperCLaw-V1.1")
{
  BL_PROFILE("PeleLMeX::writePlotfileWithMapping()");

  AMREX_ALWAYS_ASSERT(nlevels <= mf_cc.size());
  AMREX_ALWAYS_ASSERT(nlevels <= mf_nd.size());
  AMREX_ALWAYS_ASSERT(nlevels <= ref_ratio.size() + 1);
  AMREX_ALWAYS_ASSERT(nlevels <= level_steps.size());
  AMREX_ALWAYS_ASSERT(mf_cc[0]->nComp() == static_cast<int>(varnames.size()));
  AMREX_ALWAYS_ASSERT(mf_nd[0]->nComp() >= AMREX_SPACEDIM);

  const std::string levelPrefix{"Level_"};
  const std::string mfPrefix{"Cell"};
  const std::string mfNodalPrefix{"Nu_nd"};

  const bool callBarrier = false;
  amrex::PreBuildDirectorHierarchy(
    plotfilename, levelPrefix, nlevels, callBarrier);
  amrex::ParallelDescriptor::Barrier();

  // Header is written by the highest-rank process, matching ERF / AMReX
  // convention.
  if (
    amrex::ParallelDescriptor::MyProc() ==
    amrex::ParallelDescriptor::NProcs() - 1) {
    amrex::Vector<amrex::BoxArray> bArrays(nlevels);
    for (int lev = 0; lev < nlevels; ++lev) {
      bArrays[lev] = mf_cc[lev]->boxArray();
    }
    auto f = [=]() {
      amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
      const std::string HeaderFileName(plotfilename + "/Header");
      std::ofstream HeaderFile;
      HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
      HeaderFile.open(
        HeaderFileName.c_str(),
        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      if (!HeaderFile.good()) {
        amrex::FileOpenFailed(HeaderFileName);
      }
      writePlotfileHeaderWithMapping(
        HeaderFile, nlevels, bArrays, varnames, geom, time, level_steps,
        ref_ratio, versionName, levelPrefix, mfPrefix, mfNodalPrefix);
    };
    if (amrex::AsyncOut::UseAsyncOut()) {
      amrex::AsyncOut::Submit(std::move(f));
    } else {
      f();
    }
  }

  for (int lev = 0; lev < nlevels; ++lev) {
    if (amrex::AsyncOut::UseAsyncOut()) {
      amrex::VisMF::AsyncWrite(
        *mf_cc[lev],
        amrex::MultiFabFileFullPrefix(lev, plotfilename, levelPrefix, mfPrefix),
        true);
      amrex::VisMF::AsyncWrite(
        *mf_nd[lev],
        amrex::MultiFabFileFullPrefix(
          lev, plotfilename, levelPrefix, mfNodalPrefix),
        true);
    } else {
      amrex::VisMF::Write(
        *mf_cc[lev], amrex::MultiFabFileFullPrefix(
                       lev, plotfilename, levelPrefix, mfPrefix));
      amrex::VisMF::Write(
        *mf_nd[lev], amrex::MultiFabFileFullPrefix(
                       lev, plotfilename, levelPrefix, mfNodalPrefix));
    }
  }

  if (amrex::AsyncOut::UseAsyncOut()) {
    amrex::AsyncOut::Finalize();
  }
}

} // anonymous namespace

void
PeleLM::WriteDebugPlotFile(
  const amrex::Vector<const amrex::MultiFab*>& a_MF, const std::string& pltname)
{
  const int nComp = a_MF[0]->nComp();
  amrex::Vector<std::string> names(nComp);
  for (int n = 0; n < nComp; ++n) {
    names[n] = "comp" + std::to_string(n);
  }
  amrex::Vector<int> istep(finest_level + 1, m_nstep);
#ifdef AMREX_USE_HDF5
  if (m_write_hdf5_pltfile) {
    amrex::WriteMultiLevelPlotfileHDF5(
      pltname, finest_level + 1, a_MF, names, Geom(), m_cur_time, istep,
      refRatio());
  } else
#endif
  {
    amrex::WriteMultiLevelPlotfile(
      pltname, finest_level + 1, a_MF, names, Geom(), m_cur_time, istep,
      refRatio());
  }
}

void
PeleLM::WritePlotFile()
{
  BL_PROFILE("PeleLMeX::WritePlotFile()");

  const std::string& plotfilename =
    amrex::Concatenate(m_plot_file, m_nstep, m_ioDigits);

  if (m_verbose != 0) {
    amrex::Print() << "\n Writing plotfile: " << plotfilename << "\n";
  }

  //----------------------------------------------------------------
  // Delete plotfiles if present and requested (and have same name)
  if (m_plot_overwrite) {
    if (amrex::ParallelContext::IOProcessorSub()) {
      if (amrex::FileExists(plotfilename)) {
        amrex::FileSystem::RemoveAll(plotfilename);
      }
    }
  }

  amrex::VisMF::SetNOutFiles(m_nfiles);

  //----------------------------------------------------------------
  // Average down the state
  averageDownState(AmrNewTime);
  if (m_nAux > 0) {
    averageDownAux(AmrNewTime);
  }
  // Get consistent reaction data across level
  if ((m_do_react != 0) && (m_skipInstantRR == 0) && (m_plot_react != 0)) {
    averageDownReaction();
  }

  //----------------------------------------------------------------
  // Number of components
  int ncomp = 0;

  // State
  if (m_incompressible != 0) {
    // Velocity + pressure gradients
    ncomp = 2 * AMREX_SPACEDIM;
  } else {
    // State + pressure gradients
    if (m_plot_grad_p != 0) {
      ncomp = NVAR + AMREX_SPACEDIM;
    } else {
      ncomp = NVAR;
    }
    // Make the plot lighter by dropping species by default
    if (m_plotStateSpec == 0) {
      ncomp -= NUM_SPECIES;
    }
    if (m_has_divu != 0) {
      ncomp += 1;
    }
  }

  ncomp += m_nAux;

  // Reactions
  if ((m_do_react != 0) && (m_skipInstantRR == 0) && (m_plot_react != 0)) {
    // Cons Rate
    ncomp += nCompIR();
    // FunctCall
    ncomp += 1;
    // Extras:
    if (m_plotHeatRelease != 0) {
      ncomp += 1;
    }
  }

#ifdef AMREX_USE_EB
  // Include volume fraction in plotfile
  ncomp += 1;
#endif

#ifdef PELE_USE_RADIATION
  if (do_rad_solve) {
    ncomp += 3;
  }
#endif

  // Derive
  int deriveEntryCount = 0;
  for (int ivar = 0; ivar < m_derivePlotVarCount; ++ivar) {
    const PeleLMDeriveRec* rec = derive_lst.get(m_derivePlotVars[ivar]);
    deriveEntryCount += rec->numDerive();
  }
  ncomp += deriveEntryCount;
#ifdef PELE_USE_SPRAY
  if (do_spray_particles) {
    ncomp += SprayParticleContainer::NumDeriveVars();
    if (SprayParticleContainer::plot_spray_src) {
      ncomp += AMREX_SPACEDIM + 2 + SPRAY_FUEL_NUM;
    }
  }
#endif

#ifdef PELE_USE_PLASMA
  if (m_do_extraEFdiags) {
    ncomp += NUM_IONS * AMREX_SPACEDIM;
  }
#endif

  if (m_do_les && m_plot_les) {
    // Compute turbulent viscosity from current flow field at AmrNewTime
    calcTurbViscosity(AmrNewTime);
    ncomp += 1;
  }

  if (m_plot_extSource) {
    ncomp += NVAR;
  }

  //----------------------------------------------------------------
  // Plot MultiFabs
  amrex::Vector<amrex::MultiFab> mf_plt;
  mf_plt.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    mf_plt.emplace_back(
      grids[lev], dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
  }

  //----------------------------------------------------------------
  // Components names
  amrex::Vector<std::string> names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    names, &(eos_parms.host_parm()));

  amrex::Vector<std::string> plt_VarsName;
  AMREX_D_TERM(plt_VarsName.push_back("x_velocity");
               , plt_VarsName.push_back("y_velocity");
               , plt_VarsName.push_back("z_velocity"));
  if (m_incompressible == 0) {
    plt_VarsName.push_back("density");
    if (m_plotStateSpec != 0) {
      for (int n = 0; n < NUM_SPECIES; ++n) {
        plt_VarsName.push_back("rho.Y(" + names[n] + ")");
      }
    }
    plt_VarsName.push_back("rhoh");
    plt_VarsName.push_back("temp");
    plt_VarsName.push_back("RhoRT");
#ifdef PELE_USE_PLASMA
    plt_VarsName.push_back("nE");
    plt_VarsName.push_back("phiV");
#endif
#ifdef PELE_USE_SOOT
    for (int mom = 0; mom < NUMSOOTVAR; ++mom) {
      const std::string sootname = soot_model->sootVariableName(mom);
      plt_VarsName.push_back(sootname);
    }
#endif
#ifdef PELE_USE_RADIATION
    if (do_rad_solve) {
      plt_VarsName.push_back("rad.G");
      plt_VarsName.push_back("rad.kappa");
      plt_VarsName.push_back("rad.emis");
    }
#endif
    if (m_has_divu != 0) {
      plt_VarsName.push_back("divu");
    }
  }

  if (m_plot_grad_p != 0) {
    AMREX_D_TERM(plt_VarsName.push_back("gradpx");
                 , plt_VarsName.push_back("gradpy");
                 , plt_VarsName.push_back("gradpz"));
  }

  for (int n = 0; n < m_nAux; ++n) {
    plt_VarsName.push_back(m_aux_names[n]);
  }

  if ((m_do_react != 0) && (m_skipInstantRR == 0) && (m_plot_react != 0)) {
    for (int n = 0; n < NUM_SPECIES; ++n) {
      plt_VarsName.push_back("I_R(" + names[n] + ")");
    }
#ifdef PELE_USE_PLASMA
    plt_VarsName.push_back("I_R(nE)");
#endif
    plt_VarsName.push_back("FunctCall");
    // Extras:
    if (m_plotHeatRelease != 0) {
      plt_VarsName.push_back("HeatRelease");
    }
  }

#ifdef AMREX_USE_EB
  plt_VarsName.push_back("volFrac");
#endif

  for (int ivar = 0; ivar < m_derivePlotVarCount; ++ivar) {
    const PeleLMDeriveRec* rec = derive_lst.get(m_derivePlotVars[ivar]);
    for (int dvar = 0; dvar < rec->numDerive(); ++dvar) {
      plt_VarsName.push_back(rec->variableName(dvar));
    }
  }
#ifdef PELE_USE_SPRAY
  if (SprayParticleContainer::NumDeriveVars() > 0) {
    // We need virtual particles for the lower levels
    setupVirtualParticles(0);
    for (const auto& spray_derive_name :
         SprayParticleContainer::DeriveVarNames()) {
      plt_VarsName.push_back(spray_derive_name);
    }
  }
  if (do_spray_particles && SprayParticleContainer::plot_spray_src) {
    plt_VarsName.push_back("spray_mass_src");
    plt_VarsName.push_back("spray_energy_src");
    AMREX_D_TERM(plt_VarsName.push_back("spray_momentumX_src");
                 , plt_VarsName.push_back("spray_momentumY_src");
                 , plt_VarsName.push_back("spray_momentumZ_src"));
    for (const auto& spray_fuel_name :
         SprayParticleContainer::m_sprayDepNames) {
      plt_VarsName.push_back("spray_" + spray_fuel_name + "_src");
    }
  }
#endif

#ifdef PELE_USE_PLASMA
  if (m_do_extraEFdiags) {
    for (int ivar = 0; ivar < NUM_IONS; ++ivar) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const std::string dir = (idim == 0) ? "X" : ((idim == 1) ? "Y" : "Z");
        plt_VarsName.push_back(
          "DriftFlux_" + names[NUM_SPECIES - NUM_IONS + ivar] + "_" + dir);
      }
    }
  }
#endif

  if (m_do_les && m_plot_les) {
    plt_VarsName.push_back("viscturb");
  }

#if NUM_ODE > 0
  for (int n = 0; n < NUM_ODE; ++n) {
    plt_VarsName.push_back(m_ode_names[n]);
  }
#endif

  // External source terms
  if (m_plot_extSource) {
    for (int ivar = 0; ivar < NVAR; ++ivar) {
      plt_VarsName.push_back("extsource_" + stateVariableName(ivar));
    }
  }

  //----------------------------------------------------------------
  // Fill the plot MultiFabs
  for (int lev = 0; lev <= finest_level; ++lev) {
    int cnt = 0;
    if (m_incompressible != 0) {
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->state, 0, cnt, AMREX_SPACEDIM, 0);
      cnt += AMREX_SPACEDIM;
    } else {
      // Velocity and density
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->state, 0, cnt, AMREX_SPACEDIM + 1,
        0);
      cnt += AMREX_SPACEDIM + 1;
      // Species only if requested
      if (m_plotStateSpec != 0) {
        amrex::MultiFab::Copy(
          mf_plt[lev], m_leveldata_new[lev]->state, FIRSTSPEC, cnt, NUM_SPECIES,
          0);
        cnt += NUM_SPECIES;
      }
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->state, RHOH, cnt, 3, 0);
      cnt += 3;
#ifdef PELE_USE_PLASMA
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->state, NE, cnt, 2, 0);
      cnt += 2;
#endif
#ifdef PELE_USE_SOOT
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->state, FIRSTSOOT, cnt, NUMSOOTVAR,
        0);
      cnt += NUMSOOTVAR;
#endif
#ifdef PELE_USE_RADIATION
      if (do_rad_solve) {
        amrex::MultiFab::Copy(mf_plt[lev], rad_model->G()[lev], 0, cnt, 1, 0);
        cnt += 1;
        amrex::MultiFab::Copy(
          mf_plt[lev], rad_model->kappa()[lev], 0, cnt, 1, 0);
        cnt += 1;
        amrex::MultiFab::Copy(
          mf_plt[lev], rad_model->emis()[lev], 0, cnt, 1, 0);
        cnt += 1;
      }
#endif
      if (m_has_divu != 0) {
        amrex::MultiFab::Copy(
          mf_plt[lev], m_leveldata_new[lev]->divu, 0, cnt, 1, 0);
        cnt += 1;
      }
    }
    if (m_plot_grad_p != 0) {
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->gp, 0, cnt, AMREX_SPACEDIM, 0);
      cnt += AMREX_SPACEDIM;
    }

    if (m_nAux > 0) {
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldata_new[lev]->auxiliaries, 0, cnt, m_nAux, 0);
      cnt += m_nAux;
    }

    if ((m_do_react != 0) && (m_skipInstantRR == 0) && (m_plot_react != 0)) {
      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldatareact[lev]->I_R, 0, cnt, nCompIR(), 0);
      cnt += nCompIR();

      amrex::MultiFab::Copy(
        mf_plt[lev], m_leveldatareact[lev]->functC, 0, cnt, 1, 0);
      cnt += 1;

      if (m_plotHeatRelease != 0) {
        std::unique_ptr<amrex::MultiFab> mf;
        mf = std::make_unique<amrex::MultiFab>(grids[lev], dmap[lev], 1, 0);
        getHeatRelease(lev, mf.get());
        amrex::MultiFab::Copy(mf_plt[lev], *mf, 0, cnt, 1, 0);
        cnt += 1;
      }
    }

#ifdef AMREX_USE_EB
    amrex::MultiFab::Copy(
      mf_plt[lev], EBFactory(lev).getVolFrac(), 0, cnt, 1, 0);
    cnt += 1;
#endif

    for (int ivar = 0; ivar < m_derivePlotVarCount; ++ivar) {
      std::unique_ptr<amrex::MultiFab> mf;
      mf = derive(m_derivePlotVars[ivar], m_cur_time, lev, 0);
      amrex::MultiFab::Copy(mf_plt[lev], *mf, 0, cnt, mf->nComp(), 0);
      cnt += mf->nComp();
    }
#ifdef PELE_USE_SPRAY
    if (SprayParticleContainer::NumDeriveVars() > 0) {
      const int num_spray_derive = SprayParticleContainer::NumDeriveVars();
      mf_plt[lev].setVal(0., cnt, num_spray_derive);
      SprayPC->computeDerivedVars(mf_plt[lev], lev, cnt);
      if (lev < finest_level) {
        amrex::MultiFab tmp_plt(
          grids[lev], dmap[lev], num_spray_derive, 0, amrex::MFInfo(),
          Factory(lev));
        tmp_plt.setVal(0.);
        VirtPC->computeDerivedVars(tmp_plt, lev, 0);
        amrex::MultiFab::Add(mf_plt[lev], tmp_plt, 0, cnt, num_spray_derive, 0);
      }
      cnt += num_spray_derive;
    }
    if (do_spray_particles && SprayParticleContainer::plot_spray_src) {
      SprayComps scomps = SprayParticleContainer::getSprayComps();
      amrex::MultiFab::Copy(
        mf_plt[lev], *m_spraysource[lev], scomps.rhoSrcIndx, cnt++, 1, 0);
      amrex::MultiFab::Copy(
        mf_plt[lev], *m_spraysource[lev], scomps.engSrcIndx, cnt++, 1, 0);
      amrex::MultiFab::Copy(
        mf_plt[lev], *m_spraysource[lev], scomps.momSrcIndx, cnt,
        AMREX_SPACEDIM, 0);
      cnt += AMREX_SPACEDIM;
      for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
        amrex::MultiFab::Copy(
          mf_plt[lev], *m_spraysource[lev], scomps.specSrcIndx + spf, cnt++, 1,
          0);
      }
    }
#endif
#ifdef PELE_USE_PLASMA
    if (m_do_extraEFdiags) {
      amrex::MultiFab::Copy(
        mf_plt[lev], *m_ionsFluxes[lev], 0, cnt, m_ionsFluxes[lev]->nComp(), 0);
      cnt += m_ionsFluxes[lev]->nComp();
    }
#endif

    if (m_do_les && m_plot_les) {
      constexpr amrex::Real fact =
        0.5 / static_cast<amrex::Real>(AMREX_SPACEDIM);
      auto const& plot_arr = mf_plt[lev].arrays();
      AMREX_D_TERM(
        auto const& mut_arr_x =
          m_leveldata_new[lev]->visc_turb_fc[0].const_arrays();
        , auto const& mut_arr_y =
            m_leveldata_new[lev]->visc_turb_fc[1].const_arrays();
        , auto const& mut_arr_z =
            m_leveldata_new[lev]->visc_turb_fc[2].const_arrays();)
      // interpolate turbulent viscosity from faces to centers
      amrex::ParallelFor(
        mf_plt[lev], [plot_arr, cnt, mut_arr_x, mut_arr_y
#if (AMREX_SPACEDIM == 3)
                      ,
                      mut_arr_z
#endif
      ] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          plot_arr[box_no](i, j, k, cnt) =
            fact *
            (AMREX_D_TERM(
              mut_arr_x[box_no](i, j, k) + mut_arr_x[box_no](i + 1, j, k),
              +mut_arr_y[box_no](i, j, k) + mut_arr_y[box_no](i, j + 1, k),
              +mut_arr_z[box_no](i, j, k) + mut_arr_z[box_no](i, j, k + 1)));
        });
      amrex::Gpu::streamSynchronize();
      cnt += 1;
    }

#if NUM_ODE > 0
    amrex::MultiFab::Copy(
      mf_plt[lev], m_leveldata_new[lev]->state, FIRSTODE, cnt, NUM_ODE, 0);
    cnt += NUM_ODE;
#endif

    if (m_plot_extSource) {
      amrex::MultiFab::Copy(mf_plt[lev], *m_extSource[lev], 0, cnt, NVAR, 0);
    }

#ifdef AMREX_USE_EB
    if (m_plot_zeroEBcovered != 0) {
      EB_set_covered(mf_plt[lev], 0.0);
    }
#endif
  }

  // No SubCycling, all levels the same step.
  amrex::Vector<int> istep(finest_level + 1, m_nstep);

#ifdef AMREX_USE_HDF5
  if (m_write_hdf5_pltfile) {
    // HDF5 plotfiles have no equivalent of the native plotfile's
    // amrexvec_nu_* curvilinear-grid handshake.  When mesh mapping is
    // active we still write the HDF5 file (so data archiving and
    // parallel-I/O performance aren't lost), but it will render in
    // Xi-space in ParaView/VisIt.  Warn once per run so the user is
    // not surprised, and point at the native-plotfile path for
    // physical-space visualization.
    static bool hdf5_mesh_mapping_warned = false;
    if (
      m_mesh_mapping && (m_plot_mesh_mapping != 0) &&
      !hdf5_mesh_mapping_warned) {
      amrex::Print()
        << "\n  [PeleLMeX] NOTE: HDF5 plotfile output is active together "
           "with mesh mapping.\n"
        << "    The AMReX HDF5 reader does not implement the "
           "amrexvec_nu_* curvilinear\n"
        << "    handshake used by the native plotfile reader, so HDF5 "
           "plotfiles will\n"
        << "    render in Xi-space (not physical space) in "
           "ParaView/VisIt.  Options:\n"
        << "      (a) switch to native plotfiles "
           "(peleLM.use_hdf5_plt = 0) if you\n"
        << "          need curvilinear visualization;\n"
        << "      (b) set peleLM.plot_mesh_mapping = 0 to silence this "
           "message and keep\n"
        << "          HDF5 output in Xi-space.\n\n";
      hdf5_mesh_mapping_warned = true;
    }
    amrex::WriteMultiLevelPlotfileHDF5(
      plotfilename, finest_level + 1, GetVecOfConstPtrs(mf_plt), plt_VarsName,
      Geom(), m_cur_time, istep, refRatio());
  } else
#endif
  {
    if (m_mesh_mapping && (m_plot_mesh_mapping != 0)) {
      // Build per-level nodal-displacement MultiFabs (Xi -> physical)
      // and emit the plotfile with the ERF-style "amrexvec_nu_*"
      // handshake so ParaView/VisIt render on the curvilinear grid.
      const int ncomp_nd = AMREX_SPACEDIM;
      amrex::Vector<amrex::MultiFab> mf_nd(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        amrex::BoxArray nba(grids[lev]);
        nba.surroundingNodes();
        mf_nd[lev].define(nba, dmap[lev], ncomp_nd, 0);
        m_mesh_map->fill_nodal_displacement(lev, Geom(lev), mf_nd[lev]);
      }
      writePlotfileWithMapping(
        plotfilename, finest_level + 1, GetVecOfConstPtrs(mf_plt),
        GetVecOfConstPtrs(mf_nd), plt_VarsName, Geom(), m_cur_time, istep,
        refRatio());
    } else {
      amrex::WriteMultiLevelPlotfile(
        plotfilename, finest_level + 1, GetVecOfConstPtrs(mf_plt), plt_VarsName,
        Geom(), m_cur_time, istep, refRatio());
    }
  }

#ifdef PELE_USE_SPRAY
  if (do_spray_particles) {
    constexpr bool is_spraycheck = false;
    for (int lev = 0; lev <= finest_level; ++lev) {
      SprayPC->SprayParticleIO(lev, is_spraycheck, plotfilename);
      // Remove virtual particles that were made for derived variables
      removeVirtualParticles(lev);
    }
  }
#endif
}

void
PeleLM::WriteHeader(const std::string& name, const bool is_checkpoint) const
{
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::string HeaderFileName(name + "/Header");
    amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::IO_Buffer_Size);
    std::ofstream HeaderFile;

    HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    HeaderFile.open(
      HeaderFileName.c_str(),
      std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (!HeaderFile.good()) {
      amrex::FileOpenFailed(HeaderFileName);
    }

    HeaderFile.precision(17);
    if (is_checkpoint) {
      HeaderFile << "Checkpoint version: 1\n";
    } else {
      HeaderFile << "HyperCLaw-V1.1\n";
    }

    HeaderFile << finest_level << "\n";

    HeaderFile << m_nstep << "\n";

#ifdef AMREX_USE_EB
    HeaderFile << m_EB_generate_max_level << "\n";
#endif

    HeaderFile << m_cur_time << "\n";
    HeaderFile << m_dt << "\n";
    HeaderFile << m_prev_dt << "\n";

    // Geometry
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      HeaderFile << Geom(0).ProbLo(i) << ' ';
    }
    HeaderFile << '\n';

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      HeaderFile << Geom(0).ProbHi(i) << ' ';
    }
    HeaderFile << '\n';

    // BoxArray
    for (int lev = 0; lev <= finest_level; ++lev) {
      boxArray(lev).writeOn(HeaderFile);
      HeaderFile << '\n';
    }

    // Ambient pressure and typvals
    HeaderFile << m_pNew << "\n";
    for (amrex::Real typical_value : typical_values) {
      HeaderFile << typical_value << "\n";
    }

    // Optional recycling-plane block. When inlet-from-plane is enabled and
    // recycling_active, emit the recycling block, including per-level status.
    // "RecyclingPlanePresent: 0" means the feature was enabled but no recycling
    // samples were available for that level yet; older checkpoints without
    // the marker remain readable because the restart code treats the absence
    // of the marker as "inactive".
    // Only fluctuations mode carries restart state (the running mean); in
    // full mode the injection buffer reseeds from the first post-restart
    // snapshot and nothing needs to be checkpointed.
    const bool recycling_active =
      (m_use_inlet_from_plane != 0) &&
      (m_recycling_mode == RecyclingMode::Fluctuations) &&
      m_inlet_recycling.initialized &&
      static_cast<int>(m_inlet_recycling.mean_src.size()) >= finest_level + 1;
    if ((m_use_inlet_from_plane != 0) && recycling_active) {
      HeaderFile << "RecyclingPlane: " << "\n";
      HeaderFile << (recycling_active ? m_inlet_recycling.n_samples : 0)
                 << "\n";
      for (int lev = 0; lev <= finest_level; ++lev) {
        const bool planePresent =
          recycling_active && (m_inlet_recycling.mean_src[lev] != nullptr);
        HeaderFile << "RecyclingPlanePresent: " << (planePresent ? 1 : 0)
                   << "\n";
        if (planePresent) {
          m_inlet_recycling.mean_src[lev]->boxArray().writeOn(HeaderFile);
          HeaderFile << "\n";
        }
      }
    }
  }
}

void
PeleLM::WriteCheckPointFile()
{
  BL_PROFILE("PeleLMeX::WriteCheckPointFile()");

  const std::string& checkpointname =
    amrex::Concatenate(m_check_file, m_nstep, m_ioDigits);

  if (m_verbose != 0) {
    amrex::Print() << "\n Writing checkpoint file: " << checkpointname << "\n";
  }

  //----------------------------------------------------------------
  // Delete checkfiles if present and requested (and have same name)
  if (m_check_overwrite) {
    if (amrex::ParallelContext::IOProcessorSub()) {
      if (amrex::FileExists(checkpointname)) {
        amrex::FileSystem::RemoveAll(checkpointname);
      }
    }
  }

  amrex::VisMF::SetNOutFiles(m_nfiles);

  amrex::PreBuildDirectorHierarchy(
    checkpointname, level_prefix, finest_level + 1, true);

  constexpr bool is_checkpoint = true;
  WriteHeader(checkpointname, is_checkpoint);
  WriteJobInfo(checkpointname);

  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::VisMF::Write(
      m_leveldata_new[lev]->state,
      amrex::MultiFabFileFullPrefix(
        lev, checkpointname, level_prefix, "state"));

    amrex::VisMF::Write(
      m_leveldata_new[lev]->gp, amrex::MultiFabFileFullPrefix(
                                  lev, checkpointname, level_prefix, "gradp"));

    amrex::VisMF::Write(
      m_leveldata_new[lev]->press,
      amrex::MultiFabFileFullPrefix(lev, checkpointname, level_prefix, "p"));

    if (m_nAux > 0) {
      amrex::VisMF::Write(
        m_leveldata_new[lev]->auxiliaries,
        amrex::MultiFabFileFullPrefix(
          lev, checkpointname, level_prefix, "aux"));
    }

    if (m_incompressible == 0) {
      if (m_has_divu != 0) {
        amrex::VisMF::Write(
          m_leveldata_new[lev]->divu,
          amrex::MultiFabFileFullPrefix(
            lev, checkpointname, level_prefix, "divU"));
      }

      if (m_do_react != 0) {
        amrex::VisMF::Write(
          m_leveldatareact[lev]->I_R,
          amrex::MultiFabFileFullPrefix(
            lev, checkpointname, level_prefix, "I_R"));
      }
    }

    // Per-level recycling-plane running mean (only when active and seeded;
    // fluctuations mode only, matching the header block in WriteHeader).
    if (
      (m_use_inlet_from_plane != 0) &&
      (m_recycling_mode == RecyclingMode::Fluctuations) &&
      m_inlet_recycling.initialized &&
      lev < static_cast<int>(m_inlet_recycling.mean_src.size()) &&
      m_inlet_recycling.mean_src[lev]) {
      amrex::VisMF::Write(
        *m_inlet_recycling.mean_src[lev],
        amrex::MultiFabFileFullPrefix(
          lev, checkpointname, level_prefix, "recycle_mean"));
    }
  }
#ifdef PELE_USE_SPRAY
  if (do_spray_particles) {
    constexpr bool is_spraycheck = true;
    for (int lev = 0; lev <= finest_level; ++lev) {
      SprayPC->SprayParticleIO(lev, is_spraycheck, checkpointname);
    }
  }
#endif
}

void
PeleLM::ReadCheckPointFile()
{
  BL_PROFILE("PeleLMeX::ReadCheckPointFile()");

  amrex::Print() << "Restarting from checkpoint " << m_restart_chkfile << "\n";

  amrex::Real prob_lo[AMREX_SPACEDIM];
  amrex::Real prob_hi[AMREX_SPACEDIM];

  /***************************************************************************
  ** Load header: set up problem domain (including BoxArray)                 *
  **              allocate PeleLMeX memory (PeleLM::AllocateArrays)            *
  **              (by calling MakeNewLevelFromScratch)                       *
  ****************************************************************************/

  std::string File(m_restart_chkfile + "/Header");

  amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

  amrex::Vector<char> fileCharPtr;
  amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
  std::string fileCharPtrString(fileCharPtr.dataPtr());
  std::istringstream is(fileCharPtrString, std::istringstream::in);

  std::string line, word;

  // Start reading from checkpoint file

  // Title line
  std::getline(is, line);

  // Finest level
  int chk_finest_level = 0;
  is >> chk_finest_level;
  GotoNextLine(is);
  finest_level = amrex::min(chk_finest_level, max_level);

  // Step count
  is >> m_nstep;
  GotoNextLine(is);

#ifdef AMREX_USE_EB
  // Finest level at which EB was generated
  // actually used independently, so just skip ...
  std::getline(is, line);

  // ... but to be backward compatible, if we get a float,
  // let's assume it's m_cur_time
  if (line.find('.') != std::string::npos) {
    m_cur_time = std::stod(line);
  } else {
    // Skip line and read current time
    is >> m_cur_time;
    GotoNextLine(is);
  }
#else

  // Current time
  is >> m_cur_time;
  GotoNextLine(is);
#endif

  // Time step size
  is >> m_dt;
  GotoNextLine(is);

  is >> m_prev_dt;
  GotoNextLine(is);

  // Low coordinates of domain bounding box
  std::getline(is, line);
  {
    std::istringstream lis(line);
    int i = 0;
    while (lis >> word) {
      prob_lo[i++] = std::stod(word);
    }
  }

  // High coordinates of domain bounding box
  std::getline(is, line);
  {
    std::istringstream lis(line);
    int i = 0;
    while (lis >> word) {
      prob_hi[i++] = std::stod(word);
    }
  }

  // Set up problem domain
  amrex::RealBox rb(prob_lo, prob_hi);
  amrex::Geometry::ResetDefaultProbDomain(rb);
  for (int lev = 0; lev <= max_level; ++lev) {
    SetGeometry(
      lev,
      amrex::Geometry(
        Geom(lev).Domain(), rb, Geom(lev).CoordInt(), Geom(lev).isPeriodic()));
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    // read in level 'lev' BoxArray from Header
    amrex::BoxArray ba;
    ba.readFrom(is);
    GotoNextLine(is);

    // Create distribution mapping
    amrex::DistributionMapping dm{ba, amrex::ParallelDescriptor::NProcs()};
    MakeNewLevelFromScratch(lev, m_cur_time, ba, dm);
  }

  for (int lev = finest_level + 1; lev <= chk_finest_level; ++lev) {
    // read dummy level 'lev' BoxArray if restarting with reduced levels
    amrex::BoxArray ba;
    ba.readFrom(is);
  }

  // deal with typval and P_amb
  is >> m_pNew;
  GotoNextLine(is);
  m_pOld = m_pNew;
  for (amrex::Real& typical_value : typical_values) {
    is >> typical_value;
    GotoNextLine(is);
  }

  // Optional recycling-plane block. Pre-existing checkpoints don't have this
  // marker; we leave the state un-restored (the running mean will reseed
  // from the next sample), however we must be careful to safely rewind the
  // stream if the marker is not found
  bool have_recycling_chk = false;
  int chk_recycling_n_samples = 0;
  amrex::Vector<amrex::BoxArray> chk_recycling_ba;
  {
    std::streampos pos = is.tellg();
    std::string marker;
    if ((is >> marker) && marker == "RecyclingPlane:") {
      GotoNextLine(is);
      is >> chk_recycling_n_samples;
      GotoNextLine(is);
      chk_recycling_ba.resize(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        std::string level_marker;
        int level_present;
        is >> level_marker >> level_present;
        GotoNextLine(is);
        AMREX_ASSERT(level_marker == "RecyclingPlanePresent:");
        if (level_present != 0) {
          chk_recycling_ba[lev].readFrom(is);
          GotoNextLine(is);
        }
      }
      have_recycling_chk = true;
    } else {
      is.clear();    // Clear EOF flags if they were set
      is.seekg(pos); // Only rewind if RecyclingPlane data NOT found
    }
  }

  /***************************************************************************
   * Load fluid data                                                         *
   ***************************************************************************/

  // Load the field data
  for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef PELE_USE_PLASMA
    if (!m_restart_nonEF) {
      amrex::VisMF::Read(
        m_leveldata_new[lev]->state,
        amrex::MultiFabFileFullPrefix(
          lev, m_restart_chkfile, level_prefix, "state"));
    } else {
      // The chk state is 2 component shorter since phiV and nE aren't in it
      amrex::MultiFab stateTemp(grids[lev], dmap[lev], NVAR - 2, m_nGrowState);
      amrex::VisMF::Read(
        stateTemp, amrex::MultiFabFileFullPrefix(
                     lev, m_restart_chkfile, level_prefix, "state"));
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->state, stateTemp, 0, 0, NVAR - 2, m_nGrowState);
    }
#else
    amrex::VisMF::Read(
      m_leveldata_new[lev]->state,
      amrex::MultiFabFileFullPrefix(
        lev, m_restart_chkfile, level_prefix, "state"));
#endif

    amrex::VisMF::Read(
      m_leveldata_new[lev]->gp,
      amrex::MultiFabFileFullPrefix(
        lev, m_restart_chkfile, level_prefix, "gradp"));

    amrex::VisMF::Read(
      m_leveldata_new[lev]->press,
      amrex::MultiFabFileFullPrefix(lev, m_restart_chkfile, level_prefix, "p"));
    if (m_nAux > 0) {
      amrex::VisMF::Read(
        m_leveldata_new[lev]->auxiliaries,
        amrex::MultiFabFileFullPrefix(
          lev, m_restart_chkfile, level_prefix, "aux"));
    }

    if (m_incompressible == 0) {
      if (m_has_divu != 0) {
        amrex::VisMF::Read(
          m_leveldata_new[lev]->divu,
          amrex::MultiFabFileFullPrefix(
            lev, m_restart_chkfile, level_prefix, "divU"));
      }

#ifdef PELE_USE_PLASMA
      if (!m_restart_nonEF) {
        if (m_do_react) {
          amrex::VisMF::Read(
            m_leveldatareact[lev]->I_R,
            amrex::MultiFabFileFullPrefix(
              lev, m_restart_chkfile, level_prefix, "I_R"));
        }
      } else {
        // I_R for non-EF simulation is one component shorted, need to account
        // for that.
        if (m_do_react) {
          amrex::MultiFab I_Rtemp(grids[lev], dmap[lev], NUM_SPECIES, 0);
          amrex::VisMF::Read(
            I_Rtemp, amrex::MultiFabFileFullPrefix(
                       lev, m_restart_chkfile, level_prefix, "I_R"));
          amrex::MultiFab::Copy(
            m_leveldatareact[lev]->I_R, I_Rtemp, 0, 0, NUM_SPECIES, 0);
        }

        // Initialize nE & phiV
        m_leveldata_new[lev]->state.setVal(0.0, NE, 2, m_nGrowState);
      }
#else
      if (m_do_react != 0) {
        amrex::VisMF::Read(
          m_leveldatareact[lev]->I_R,
          amrex::MultiFabFileFullPrefix(
            lev, m_restart_chkfile, level_prefix, "I_R"));
      }
#endif
    }
  }

  // Restore the recycling-plane running mean if both the checkpoint had it
  // and recycling is enabled in this run. We deliberately ParallelCopy from a
  // temporary MultiFab (built with the on-disk BoxArray) into the rebuilt
  // slab (which uses the current run's maxGridSize), so a maxGridSize change
  // between the original run and the restart is handled transparently.
  const bool recycling_fluct_active =
    (m_use_inlet_from_plane != 0) &&
    (m_recycling_mode == RecyclingMode::Fluctuations);
  if (have_recycling_chk && recycling_fluct_active) {
    buildRecyclingPlaneStorage();
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (
        lev < static_cast<int>(chk_recycling_ba.size()) &&
        (!chk_recycling_ba[lev].empty()) &&
        lev < static_cast<int>(m_inlet_recycling.mean_src.size()) &&
        m_inlet_recycling.mean_src[lev]) {
        amrex::DistributionMapping chk_dm{
          chk_recycling_ba[lev], amrex::ParallelDescriptor::NProcs()};
        amrex::MultiFab chk_mean(
          chk_recycling_ba[lev], chk_dm, AMREX_SPACEDIM, 0);
        amrex::VisMF::Read(
          chk_mean, amrex::MultiFabFileFullPrefix(
                      lev, m_restart_chkfile, level_prefix, "recycle_mean"));
        m_inlet_recycling.mean_src[lev]->ParallelCopy(
          chk_mean, 0, 0, AMREX_SPACEDIM);
      }
    }
    m_inlet_recycling.initialized = true;
    m_inlet_recycling.n_samples = chk_recycling_n_samples;
    if (m_inlet_recycling.n_samples <= m_inlet_plane_warmup_steps) {
      amrex::Print()
        << "WARNING: Current setting for inlet_plane_warmup_steps is greater "
        << "than or equal to checkpointed n_samples. Injected fluctuations "
        << "will remain disabled until additional snapshots increase "
        << "n_samples beyond the warmup threshold.\n";
    }
    // Storage was just rebuilt from current-run geometry; the regrid hooks
    // may set this back to 1 later, which is fine.
    m_recycling_needs_rebuild = false;
  } else if (have_recycling_chk && (m_use_inlet_from_plane == 0)) {
    amrex::Print()
      << "WARNING: Restart checkpoint contains RecyclingPlane data, but "
      << "peleLM.use_inlet_from_plane = 0 in this run. Ignoring checkpointed "
      << "recycling-plane running mean.\n";
  } else if (have_recycling_chk) {
    // use_inlet_from_plane != 0 but recycling_mode = full: the mean belongs
    // to fluctuations mode and full mode needs no restored state (the
    // injection buffer reseeds from the first post-restart snapshot).
    amrex::Print()
      << "WARNING: peleLM.recycling_mode = full; ignoring the checkpointed "
      << "recycling-plane running mean.\n";
  } else if ((!have_recycling_chk) && recycling_fluct_active) {
    amrex::Print()
      << "WARNING: peleLM.use_inlet_from_plane != 0 in this run, but the "
      << "restart checkpoint does not contain RecyclingPlane data. "
      << "Recycling-plane running mean will not be restored from checkpoint.\n";
  }

  if (m_verbose != 0) {
    amrex::Print() << "Restart complete \n";
  }
}

void
PeleLM::initLevelDataFromPlt(int a_lev, const std::string& a_dataPltFile)
{
  if (m_incompressible != 0) {
    amrex::Abort(
      " initializing data from a pltfile only available for low-Mach "
      "simulations");
  }
  if (m_nAux > 0) {
    amrex::Warning(
      " restarting from plotfile with auxiliaries not currently "
      "implemented, and will not be captured");
  }
  if (m_verbose > 0) {
    amrex::Print() << " initData on level " << a_lev << " from pltfile "
                   << a_dataPltFile << "\n";
    if (pltfileSource == "LM") {
      amrex::Print() << " Assuming pltfile was generated in LM/LMeX \n";
    } else if (pltfileSource == "C") {
      amrex::Print() << " Assuming pltfile was generated in PeleC \n";
    }
  }

  // Use PelePhysics PltFileManager – reuse the shared instance if it was
  // pre-created in initData() to avoid re-reading the plotfile header for
  // every AMR level, which can cause OOM for large plotfiles.
  std::unique_ptr<pele::physics::pltfilemanager::PltFileManager> localPltData;
  pele::physics::pltfilemanager::PltFileManager* pltDataPtr = nullptr;
  if (m_pltDataManager) {
    pltDataPtr = m_pltDataManager.get();
  } else {
    localPltData =
      std::make_unique<pele::physics::pltfilemanager::PltFileManager>(
        a_dataPltFile);
    pltDataPtr = localPltData.get();
  }
  auto& pltData = *pltDataPtr;
  amrex::Vector<std::string> plt_vars = pltData.getVariableList();

  // Track whether we coarsened the base level (for skip_plt_fill logic later)
  bool did_coarsen_base = false;

  // Handle 2x coarsening with peleLM.initDataPlt_coarsen
  if (m_initDataPlt_coarsen) {
    if (a_lev == 0) {
      m_level_shift = computeLevelShift(pltData);

      if (m_level_shift > 0) {
        amrex::Print() << "  peleLM.initDataPlt_coarsen: Coarsening active\n";
        amrex::Print() << "  Plotfile level K will become new level K+1\n";

        // Verify we have enough plotfile levels
        if (finest_level - 1 > pltData.getNlev() - 1) {
          amrex::Abort(
            "peleLM.initDataPlt_coarsen: Insufficient plotfile levels.");
        }
      }
    }

    // Coarsen from plotfile level 0 to new coarser level 0
    if (a_lev == 0 && m_level_shift > 0) {
      coarsenLevelFromPlt(
        a_lev, pltData, amrex::IntVect(AMREX_D_DECL(2, 2, 2)));
      did_coarsen_base = true;
    }
    // Intermediate levels don't have corresponding plotfile data - interpolate
    else if (a_lev > 0 && a_lev < m_level_shift) {
      if (m_verbose > 0) {
        amrex::Print() << "  Level " << a_lev << ": Interpolating from level "
                       << (a_lev - 1) << "\n";
      }

      auto* ldata_p = getLevelDataPtr(a_lev, AmrNewTime);
      fillcoarsepatch_state(a_lev, m_cur_time, ldata_p->state, 0);
      return;
    }
  }

  if (m_do_reset_time == 0) {
    m_cur_time = pltData.getTime();
    m_nstep = pltData.getNsteps();

    // Plotfiles don't contain dt/prev_dt values, but we need to initialize
    // them to avoid negative dt calculation
    if (m_fixed_dt > 0.0) {
      m_dt = m_fixed_dt;
      m_prev_dt = m_fixed_dt;
    } else {
      // Use large value so CFL calculation isn't artificially constrained
      m_dt = m_max_dt;
      m_prev_dt = m_max_dt;
    }
  }

  // Find required data in pltfile
  amrex::Vector<std::string> spec_names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    spec_names, &(eos_parms.host_parm()));
  int idT = -1, idV = -1, idY = -1, nSpecPlt = 0;
#ifdef PELE_USE_PLASMA
  int inE = -1, iPhiV = -1;
#endif
#ifdef PELE_USE_SOOT
  int inSoot = -1;
#endif
  for (int i = 0; i < plt_vars.size(); ++i) {
    std::string firstChars = plt_vars[i].substr(0, 2);

    if (pltfileSource == "LM") {
      if (plt_vars[i] == "temp") {
        idT = i;
      }
    } else if (pltfileSource == "C") {
      if (plt_vars[i] == "Temp") {
        idT = i;
      }
    }
    if (plt_vars[i] == "x_velocity") {
      idV = i;
    }
    if (firstChars == "Y(" && idY < 0) { // species might not be ordered in the
                                         // order of the current mech.
      idY = i;
    }
    if (firstChars == "Y(") {
      nSpecPlt += 1;
    }
#ifdef PELE_USE_PLASMA
    if (plt_vars[i] == "nE")
      inE = i;
    if (plt_vars[i] == "phiV")
      iPhiV = i;
#endif
#ifdef PELE_USE_SOOT
    if (plt_vars[i] == "soot_N") {
      inSoot = i;
    }
#endif
  }
  if (idY < 0) {
    amrex::Abort("Couldn't find species mass fractions in pltfile");
  } else if (idT < 0) {
    amrex::Abort("Couldn't find temperature in pltfile");
  }
  if (m_verbose > 0) {
    amrex::Print() << " " << nSpecPlt
                   << " species found in pltfile, starting with "
                   << plt_vars[idY] << "\n";
  }

  // Get level data
  auto* ldata_p = getLevelDataPtr(a_lev, AmrNewTime);

  // Determine if we need to fill from plotfile
  bool skip_plt_fill = false;
  int source_plt_level = a_lev;

  if (m_initDataPlt_coarsen && m_level_shift > 0) {
    if (did_coarsen_base) {
      skip_plt_fill = true;
    } else {
      source_plt_level =
        a_lev - m_level_shift; // Read from corresponding plotfile level
      if (m_verbose > 0) {
        amrex::Print() << "  Level " << a_lev
                       << ": Reading from plotfile level " << source_plt_level
                       << "\n";
      }
    }
  }

  if (!skip_plt_fill) {
    // Velocity
    pltData.fillPatchFromPlt(
      source_plt_level, geom[a_lev], idV, VELX, AMREX_SPACEDIM, ldata_p->state);

    // Temperature
    pltData.fillPatchFromPlt(
      source_plt_level, geom[a_lev], idT, TEMP, 1, ldata_p->state);

    // Species
    // Hold the species in temporary MF before copying to level data
    // in case the number of species differs.
    amrex::MultiFab speciesPlt(grids[a_lev], dmap[a_lev], nSpecPlt, 0);
    pltData.fillPatchFromPlt(
      source_plt_level, geom[a_lev], idY, 0, nSpecPlt, speciesPlt);
    for (int i = 0; i < NUM_SPECIES; ++i) {
      std::string specName = "Y(" + spec_names[i] + ")";
      std::string specString = specName;
      if (!m_initDataPlt_specname_map.empty()) {
        specString = m_initDataPlt_specname_map[i];
      }
      int foundSpec = 0;
      for (int iplt = 0; iplt < nSpecPlt; ++iplt) {
        if (specString == plt_vars[idY + iplt]) {
          amrex::MultiFab::Copy(
            ldata_p->state, speciesPlt, iplt, FIRSTSPEC + i, 1, 0);
          foundSpec = 1;
          if (m_verbose > 0) {
            amrex::Print() << "Loading species " << specName
                           << " from plotfile species " << specString << "\n";
          }
        }
      }
      if (foundSpec == 0) {
        for (int iplt = 0; iplt < plt_vars.size(); ++iplt) {
          if (specString == plt_vars[iplt]) {
            foundSpec = 1;
            if (m_verbose > 0) {
              amrex::Print() << "Loading species " << specName
                             << " from plotfile entry " << specString << "\n";
            }
            pltData.fillPatchFromPlt(
              source_plt_level, geom[a_lev], iplt, FIRSTSPEC + i, 1,
              ldata_p->state);
          }
        }
      }
      if (foundSpec == 0) {
        ldata_p->state.setVal(0.0, FIRSTSPEC + i, 1);
        if (m_verbose > 0) {
          amrex::Print() << "For species " << specName << " entry "
                         << specString
                         << " not found in plot file, setting to 0\n";
        }
      }
    }

    // Converting units when pltfile is coming from PeleC solution
    if (pltfileSource == "C") {
      amrex::Print() << " Converting CGS to MKS units... \n";
      auto const& state_ma = ldata_p->state.arrays();
      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
        [state_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> vel(state_ma[box_no], VELX);
          vel(i, j, k, n) *= 0.01;
        });
      amrex::Gpu::streamSynchronize();
    }

#ifdef PELE_USE_PLASMA
    // nE
    pltData.fillPatchFromPlt(
      source_plt_level, geom[a_lev], inE, NE, 1, ldata_p->state);
    // phiV
    pltData.fillPatchFromPlt(
      source_plt_level, geom[a_lev], iPhiV, PHIV, 1, ldata_p->state);
#endif
#ifdef PELE_USE_SOOT
    if (do_soot_solve) {
      if (inSoot >= 0) {
        pltData.fillPatchFromPlt(
          source_plt_level, geom[a_lev], inSoot, FIRSTSOOT, NUMSOOTVAR,
          ldata_p->state);
        if (pltfileSource == "C") {
          SootConst sc;
          amrex::Real* momV = sc.MomOrderV.data();
          amrex::Real* momS = sc.MomOrderS.data();
          auto const& state_ma = ldata_p->state.arrays();
          amrex::Real soot_exp[NUM_SOOT_MOMENTS] = {0.0};
          for (int n = 0; n < NUM_SOOT_MOMENTS; ++n) {
            soot_exp[n] = 3. - (3. * momV[n] + 2. * momS[n]);
          }
          amrex::ParallelFor(
            ldata_p->state, [state_ma, soot_exp] AMREX_GPU_DEVICE(
                              int box_no, int i, int j, int k) noexcept {
              amrex::Array4<amrex::Real> soot(state_ma[box_no], FIRSTSOOT);
              for (int n = 0; n < NUM_SOOT_MOMENTS; ++n) {
                soot(i, j, k, n) *= std::pow(100., soot_exp[n]);
              }
              soot(i, j, k, NUMSOOTVAR - 1) *= 1.E6;
            });
          amrex::Gpu::streamSynchronize();
        }
      } else {
        SootData* const sd = soot_model->getSootData();
        amrex::Real moments[NUM_SOOT_MOMENTS + 1];
        sd->initialSmallMomVals(moments);
        auto const& state_ma = ldata_p->state.arrays();
        amrex::ParallelFor(
          ldata_p->state, amrex::IntVect(0), NUM_SOOT_MOMENTS + 1,
          [state_ma, moments] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            amrex::Array4<amrex::Real> soot(state_ma[box_no], FIRSTSOOT);
            soot(i, j, k, n) = moments[n];
          });
      }
    }
#endif
  } // end if (!skip_plt_fill)

  // Pressure and pressure gradients to zero
  ldata_p->press.setVal(0.0);
  ldata_p->gp.setVal(0.0);

  ProbParm const* lprobparm = prob_parm_d;

  // If m_do_patch_flow_variables is set as true, call user-defined function to
  // patch flow variables
  if (m_do_patch_flow_variables) {
    ProblemSpecificFunctions::patchFlowVariables(
      geom[a_lev], *lprobparm, ldata_p->state);
  }

  // Enforce rho and rhoH consistent with temperature and mixture
  // The above handles species mapping (to some extent), but nothing enforce
  // sum of Ys = 1 -> use N2 in the following if N2 is present
  auto const& state_ma = ldata_p->state.arrays();
  auto const* leosparm = eos_parms.device_parm();
  const amrex::Real P_cgs = m2c::P(lprobparm->P_mean);

  amrex::ParallelFor(
    ldata_p->state, [state_ma, P_cgs, eosparm = leosparm] AMREX_GPU_DEVICE(
                      int box_no, int i, int j, int k) noexcept {
      auto eos = pele::physics::PhysicsType::eos(eosparm);
      amrex::Array4<amrex::Real> rho(state_ma[box_no], DENSITY);
      amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
      amrex::Array4<amrex::Real> rhoH(state_ma[box_no], RHOH);
      amrex::Array4<amrex::Real> temp(state_ma[box_no], TEMP);
      amrex::Real massfrac[NUM_SPECIES] = {0.0};
      amrex::Real sumYs = 0.0;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        massfrac[n] = rhoY(i, j, k, n);
#ifdef N2_ID
        if (n != N2_ID) {
          sumYs += massfrac[n];
        }
#endif
      }
#ifdef N2_ID
      massfrac[N2_ID] = 1.0 - sumYs;
#endif
      // Get density
      amrex::Real rho_cgs = 0.0;
      eos.PYT2R(P_cgs, massfrac, temp(i, j, k), rho_cgs);
      rho(i, j, k) = c2m::Rho(rho_cgs);

      // Get enthalpy
      amrex::Real h_cgs = 0.0;
      eos.TY2H(temp(i, j, k), massfrac, h_cgs);
      rhoH(i, j, k) = c2m::H(h_cgs) * rho(i, j, k);

      // Fill rhoYs
      for (int n = 0; n < NUM_SPECIES; ++n) {
        rhoY(i, j, k, n) = massfrac[n] * rho(i, j, k);
      }
    });
  amrex::Gpu::streamSynchronize();
  // Initialize thermodynamic pressure
  setThermoPress(a_lev, AmrNewTime);
  if (m_has_divu != 0) {
    ldata_p->divu.setVal(0.0);
  }
}

int
PeleLM::computeLevelShift(
  pele::physics::pltfilemanager::PltFileManager& pltData)
{
  amrex::IntVect plt_domain_size = pltData.getGeom(0).Domain().size();
  amrex::IntVect new_domain_size = geom[0].Domain().size();

  // No coarsening if domains match
  if (plt_domain_size == new_domain_size) {
    return 0;
  }

  // Validate 2x coarsening in all directions
  if (plt_domain_size != new_domain_size * 2) {
    amrex::Abort(
      "peleLM.initDataPlt_coarsen: Only 2x coarsening is supported. "
      "Plotfile domain must be exactly 2x larger in each direction.");
  }

  if (m_verbose > 0) {
    amrex::Print() << "  peleLM.initDataPlt_coarsen: 2x coarsening detected\n";
    amrex::Print() << "  Plotfile level 0: " << plt_domain_size << "\n";
    amrex::Print() << "  New level 0: " << new_domain_size << "\n";
  }

  return 1; // level_shift = 1 for 2x coarsening
}

void
PeleLM::coarsenLevelFromPlt(
  const int a_lev,
  pele::physics::pltfilemanager::PltFileManager& pltData,
  const amrex::IntVect& coarsen_ratio)
{
  BL_PROFILE("PeleLMeX::coarsenLevelFromPlt()");

  if (m_verbose > 0) {
    amrex::Print() << "  Coarsening level " << a_lev
                   << " from plotfile level 0 "
                   << "with ratio " << coarsen_ratio << "\n";
  }

#ifdef AMREX_USE_EB
  if (!EBFactory(a_lev).isAllRegular()) {
    amrex::Abort(
      "peleLM.initDataPlt_coarsen: EB geometry not supported. "
      "This feature currently only works for non-EB cases.");
  }
#endif

  if (m_mesh_mapping) {
    amrex::Abort(
      "peleLM.initDataPlt_coarsen: Mesh mapping not supported. "
      "This feature currently only works without mesh mapping.");
  }

  // Create temporary fine MultiFab matching plotfile level 0 resolution
  amrex::BoxArray fineBA = amrex::refine(grids[a_lev], coarsen_ratio);
  amrex::DistributionMapping fineDM(fineBA);
  amrex::MultiFab fineData(fineBA, fineDM, NVAR, 0);

  // Initialize all components to avoid propagating garbage in uninitialized
  // fields (e.g., RHORT, plasma/soot fields) when averaging down
  fineData.setVal(0.0);

  // Fill fine data from plotfile level 0 using standard fillPatchFromPlt
  // This bypasses the coarsening restriction in PltFileManager
  amrex::Vector<std::string> spec_names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    spec_names, &(eos_parms.host_parm()));
  amrex::Vector<std::string> plt_vars = pltData.getVariableList();

  // Find required variable indices in plotfile
  int idT = -1, idV = -1, idY = -1;
  for (int i = 0; i < plt_vars.size(); ++i) {
    if (
      (pltfileSource == "LM" && plt_vars[i] == "temp") ||
      (pltfileSource == "C" && plt_vars[i] == "Temp")) {
      idT = i;
    }
    if (plt_vars[i] == "x_velocity") {
      idV = i;
    }
    std::string firstChars = plt_vars[i].substr(0, 2);
    if (firstChars == "Y(" && idY < 0) {
      idY = i;
    }
  }

  if (idT < 0 || idV < 0 || idY < 0) {
    amrex::Abort(
      "coarsenLevelFromPlt: Could not find required variables in plotfile");
  }

  // Create fine geometry for interpolation
  amrex::Geometry fineGeom = amrex::refine(geom[a_lev], coarsen_ratio);

  // Fill velocity
  pltData.fillPatchFromPlt(0, fineGeom, idV, VELX, AMREX_SPACEDIM, fineData);

  // Fill temperature
  pltData.fillPatchFromPlt(0, fineGeom, idT, TEMP, 1, fineData);

  // Fill species
  int nSpecPlt = 0;
  for (const auto& var : plt_vars) {
    if (var.substr(0, 2) == "Y(") {
      nSpecPlt++;
    }
  }

  amrex::MultiFab fineSpecies(fineBA, fineDM, nSpecPlt, 0);
  pltData.fillPatchFromPlt(0, fineGeom, idY, 0, nSpecPlt, fineSpecies);

  // Map species to current mechanism
  for (int i = 0; i < NUM_SPECIES; ++i) {
    std::string specName = "Y(" + spec_names[i] + ")";
    std::string specString = specName;
    if (!m_initDataPlt_specname_map.empty()) {
      specString = m_initDataPlt_specname_map[i];
    }

    int foundSpec = 0;
    for (int iplt = 0; iplt < nSpecPlt; ++iplt) {
      if (specString == plt_vars[idY + iplt]) {
        amrex::MultiFab::Copy(fineData, fineSpecies, iplt, FIRSTSPEC + i, 1, 0);
        foundSpec = 1;
        break;
      }
    }

    if (foundSpec == 0) {
      // Try searching full variable list
      for (int iplt = 0; iplt < plt_vars.size(); ++iplt) {
        if (specString == plt_vars[iplt]) {
          amrex::MultiFab tmpSpec(fineBA, fineDM, 1, 0);
          pltData.fillPatchFromPlt(0, fineGeom, iplt, 0, 1, tmpSpec);
          amrex::MultiFab::Copy(fineData, tmpSpec, 0, FIRSTSPEC + i, 1, 0);
          foundSpec = 1;
          break;
        }
      }
    }

    if (foundSpec == 0) {
      fineData.setVal(0.0, FIRSTSPEC + i, 1, 0);
      if (m_verbose > 0) {
        amrex::Print() << "  Species " << specName
                       << " not found, setting to 0\n";
      }
    }
  }

  // Get level data pointer
  auto* ldata_p = getLevelDataPtr(a_lev, AmrNewTime);

  // Apply conservative averaging to coarsen data
  // Average down all state components
  amrex::average_down(fineData, ldata_p->state, 0, NVAR, coarsen_ratio);

  if (m_verbose > 1) {
    amrex::Print() << "  Conservative coarsening complete for level " << a_lev
                   << "\n";
  }
}

void
PeleLM::addLevelVelocityDataFromPlt(int a_lev, const std::string& a_velPltFile)
{
  if (m_verbose > 0) {
    amrex::Print() << " init velocity data on level " << a_lev
                   << " from pltfile " << a_velPltFile << "\n";
  }

  // Use PelePhysics PltFileManager – reuse the shared instance if it was
  // pre-created in initData() to avoid re-reading the plotfile header for
  // every AMR level.
  std::unique_ptr<pele::physics::pltfilemanager::PltFileManager>
    localVelPltData;
  pele::physics::pltfilemanager::PltFileManager* pltDataPtr = nullptr;
  if (m_velPltDataManager) {
    pltDataPtr = m_velPltDataManager.get();
  } else {
    localVelPltData =
      std::make_unique<pele::physics::pltfilemanager::PltFileManager>(
        a_velPltFile);
    pltDataPtr = localVelPltData.get();
  }
  auto& pltData = *pltDataPtr;
  amrex::Vector<std::string> plt_vars = pltData.getVariableList();

  // do some compatibility checks
  if (pltData.getNlev() < a_lev) {
    amrex::Abort("USE_VELOCITY: not enough levels in plotfile");
  }
  if (pltData.getGeom(a_lev).Domain() != geom[a_lev].Domain()) {
    amrex::Abort("USE_VELOCITY: problem domains do not match");
  }

  // find velocity in the plotfile
  int idXvel = -1;
  for (int i = 0; i < plt_vars.size(); ++i) {
    if (plt_vars[i] == "x_velocity") {
      idXvel = i;
    }
  }
  if (idXvel == -1) {
    amrex::Abort(
      "Could not find velocity fields in supplied velocity_plotfile");
  }

  // Get level data
  auto* ldata_p = getLevelDataPtr(a_lev, AmrNewTime);

  // load data from plot file
  amrex::BoxArray tmpVelBA(ldata_p->state.boxArray());
  amrex::DistributionMapping tmpVelDM(tmpVelBA);
  int nGrow0(0), sComp0(0);
  amrex::MultiFab tmpVel(tmpVelBA, tmpVelDM, AMREX_SPACEDIM, nGrow0);
  pltData.fillPatchFromPlt(
    a_lev, geom[a_lev], idXvel, sComp0, AMREX_SPACEDIM, tmpVel);
  // scale the velocity
  tmpVel.mult(m_velocity_plotfile_scale);
  amrex::MultiFab::Add(ldata_p->state, tmpVel, 0, VELX, AMREX_SPACEDIM, 0);
}

void
PeleLM::WriteJobInfo(const std::string& path) const
{
  if (amrex::ParallelDescriptor::IOProcessor()) {
    // job_info file with details about the run
    std::ofstream jobInfoFile;
    std::string FullPathJobInfoFile = path;

    FullPathJobInfoFile += "/PeleLMeX_job_info";
    jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

    // job information
    jobInfoFile << PrettyLine;
    jobInfoFile << " PeleLMeX Job Information\n";
    jobInfoFile << PrettyLine;

    jobInfoFile << "number of MPI processes: "
                << amrex::ParallelDescriptor::NProcs() << "\n";
#ifdef AMREX_USE_OMP
    jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif
#ifdef AMREX_USE_GPU
    jobInfoFile << "number of GPUs:          "
                << amrex::Gpu::Device::numDevicesUsed() << "\n";
    jobInfoFile << "GPU model:               "
                << amrex::Gpu::Device::deviceName() << "\n";
#endif

    jobInfoFile << "\n\n";

    // build information
    jobInfoFile << PrettyLine;
    jobInfoFile << " Build Information\n";
    jobInfoFile << PrettyLine;

    jobInfoFile << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
    jobInfoFile << "build machine: " << amrex::buildInfoGetBuildMachine()
                << "\n";
    jobInfoFile << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
    jobInfoFile << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

    jobInfoFile << "\n";

    jobInfoFile << "COMP:          " << amrex::buildInfoGetComp() << "\n";
    jobInfoFile << "COMP version:  " << amrex::buildInfoGetCompVersion()
                << "\n";
    jobInfoFile << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
    jobInfoFile << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

    // Compiler and library versions, from the headers this file was compiled
    // against. Version macros missing in a library's headers give "unknown".
    jobInfoFile << "Compiled with: ";
#if defined(__VERSION__)
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    jobInfoFile << "GCC " << __VERSION__;
#else
    jobInfoFile << __VERSION__;
#endif
#else
    jobInfoFile << "unknown";
#endif
    jobInfoFile << "\n";

#ifdef AMREX_USE_CUDA
    jobInfoFile << "CUDA:          ";
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
  defined(__CUDACC_VER_BUILD__)
    jobInfoFile << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "."
                << __CUDACC_VER_BUILD__;
#elif defined(CUDART_VERSION)
    jobInfoFile << CUDART_VERSION;
#else
    jobInfoFile << "unknown";
#endif
    jobInfoFile << "\n";
#endif
#ifdef AMREX_USE_HIP
    jobInfoFile << "HIP:           ";
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && \
  defined(HIP_VERSION_PATCH)
    jobInfoFile << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
                << HIP_VERSION_PATCH;
#else
    jobInfoFile << "unknown";
#endif
    jobInfoFile << "\n";
#endif

#ifdef AMREX_USE_MPI
    // Version of the MPI library actually loaded, first line only
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING];
    int mpi_version_len = 0;
    MPI_Get_library_version(mpi_version, &mpi_version_len);
    const std::string mpi(mpi_version);
    jobInfoFile << "MPI library:   "
                << amrex::trim(mpi.substr(0, mpi.find('\n'))) << "\n";
#endif

    // SUNDIALS_GIT_VERSION is the commit the installed library was built from
    char sundials_lib[64];
    SUNDIALSGetVersion(sundials_lib, sizeof(sundials_lib));
    jobInfoFile << "SUNDIALS:      " << SUNDIALS_VERSION;
#ifdef SUNDIALS_GIT_VERSION
    if (strlen(SUNDIALS_GIT_VERSION) > 0) {
      jobInfoFile << " (git " << SUNDIALS_GIT_VERSION << ")";
    }
#endif
    jobInfoFile << ", loaded library " << sundials_lib << "\n";

#ifdef PELE_USE_KLU
    jobInfoFile << "SuiteSparse:   ";
#if defined(SUITESPARSE_MAIN_VERSION) && defined(SUITESPARSE_SUB_VERSION) && \
  defined(SUITESPARSE_SUBSUB_VERSION)
    jobInfoFile << SUITESPARSE_MAIN_VERSION << "." << SUITESPARSE_SUB_VERSION
                << "." << SUITESPARSE_SUBSUB_VERSION;
#else
    jobInfoFile << "unknown";
#endif
#if defined(KLU_MAIN_VERSION) && defined(KLU_SUB_VERSION) && \
  defined(KLU_SUBSUB_VERSION)
    jobInfoFile << " (KLU " << KLU_MAIN_VERSION << "." << KLU_SUB_VERSION << "."
                << KLU_SUBSUB_VERSION << ")";
#endif
    jobInfoFile << "\n";
#endif
#ifdef PELE_USE_MAGMA
    jobInfoFile << "MAGMA:         ";
#if defined(MAGMA_VERSION_MAJOR) && defined(MAGMA_VERSION_MINOR) && \
  defined(MAGMA_VERSION_MICRO)
    jobInfoFile << MAGMA_VERSION_MAJOR << "." << MAGMA_VERSION_MINOR << "."
                << MAGMA_VERSION_MICRO;
#else
    jobInfoFile << "unknown";
#endif
    jobInfoFile << "\n";
#endif
#ifdef AMREX_USE_HYPRE
    jobInfoFile << "HYPRE:         ";
#if defined(HYPRE_RELEASE_VERSION)
    jobInfoFile << HYPRE_RELEASE_VERSION;
#else
    jobInfoFile << "unknown";
#endif
#ifdef HYPRE_DEVELOP_STRING
    jobInfoFile << " (" << HYPRE_DEVELOP_STRING << ")";
#endif
    jobInfoFile << "\n";
#endif

    jobInfoFile << "\n";

#ifdef PELE_USE_CMAKE
    // Same hashes as the start-up banner, from PeleGitHashes.H
    const char* githash1 = PeleBuildInfo::PeleLMeX_git_hash.c_str();
    const char* githash2 = PeleBuildInfo::AMReX_git_hash.c_str();
    const char* githash3 = PeleBuildInfo::PelePhysics_git_hash.c_str();
    const char* githash4 = PeleBuildInfo::AMReXHydro_git_hash.c_str();
    const char* githash6 = "";
#else
    const char* githash1 = amrex::buildInfoGetGitHash(1);
    const char* githash2 = amrex::buildInfoGetGitHash(2);
    const char* githash3 = amrex::buildInfoGetGitHash(3);
    const char* githash4 = amrex::buildInfoGetGitHash(4);
    // Entry 5 is the SUNDIALS source (start-up banner); SUNDIALS is reported
    // above with the version of the installed library
    const char* githash6 = amrex::buildInfoGetGitHash(6);
#endif
    const char* buildgithash = amrex::buildInfoGetBuildGitHash();

    if (strlen(githash1) > 0) {
      jobInfoFile << "PeleLMeX     git describe: " << githash1 << "\n";
    }
    if (strlen(githash2) > 0) {
      jobInfoFile << "AMReX        git describe: " << githash2 << "\n";
    }
    if (strlen(githash3) > 0) {
      jobInfoFile << "PelePhysics  git describe: " << githash3 << "\n";
    }
    if (strlen(githash4) > 0) {
      jobInfoFile << "AMREX-Hydro  git describe: " << githash4 << "\n";
    }
    if (strlen(githash6) > 0) {
      jobInfoFile << "Mechanism    git describe: " << githash6 << "\n";
    }
    if (strlen(buildgithash) > 0) {
      jobInfoFile << "Case         git describe: " << buildgithash << "\n";
    }

    jobInfoFile << "\n";

    jobInfoFile << "EOS model:       "
                << pele::physics::PhysicsType::eos_type::identifier() << "\n";
    jobInfoFile << "Transport model: "
                << pele::physics::PhysicsType::transport_type::identifier()
                << "\n";

    // build modules (GNU make only)
    for (int i = 1; i <= amrex::buildInfoGetNumModules(); ++i) {
      jobInfoFile << amrex::buildInfoGetModuleName(i) << ": "
                  << amrex::buildInfoGetModuleVal(i) << "\n";
    }

    jobInfoFile << "\n\n";

    // grid information
    jobInfoFile << PrettyLine;
    jobInfoFile << " Grid Information\n";
    jobInfoFile << PrettyLine;

    for (int lev = 0; lev <= finest_level; ++lev) {
      jobInfoFile << " level: " << lev << "\n";
      jobInfoFile << "   number of boxes = " << grids[lev].size() << "\n";
      jobInfoFile << "   maximum zones   = ";
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        jobInfoFile << geom[lev].Domain().length(idim) << " ";
      }
      jobInfoFile << "\n\n";
    }

    jobInfoFile << "\n\n";

    // runtime parameters
    jobInfoFile << PrettyLine;
    jobInfoFile << " Inputs File Parameters\n";
    jobInfoFile << PrettyLine;

    amrex::ParmParse::dumpTable(jobInfoFile, true);

    jobInfoFile.close();
  }
}

void
PeleLM::WriteMLMGResidual(
  amrex::MLMG& a_mlmg,
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs,
  const std::string& a_solver_name,
  int a_step)
{
  // Get the linear operator
  auto& linop = a_mlmg.getLinOp();
  int nlevs = linop.NAMRLevels();
  int ncomp_residual = linop.getNComp();

  // Create MultiFabs to hold the residual
  amrex::Vector<amrex::MultiFab> residual(nlevs);
  amrex::Vector<amrex::MultiFab*> residual_ptrs(nlevs);

  for (int lev = 0; lev < nlevs; ++lev) {
    // Use the same BoxArray and DistributionMap as the solution
    const auto& ba = a_sol[lev]->boxArray();
    const auto& dm = a_sol[lev]->DistributionMap();
    int ngrow = 0;
    residual[lev].define(ba, dm, ncomp_residual, ngrow);
    residual_ptrs[lev] = &residual[lev];
  }

  // Classify solver type
  bool is_tensor_diff =
    (a_solver_name.find("vel_diffusion") != std::string::npos);
  bool is_species =
    (a_solver_name.find("species_diffusion") != std::string::npos);
  bool is_temp =
    (a_solver_name.find("temperature_diffusion") != std::string::npos);
  bool is_proj = (a_solver_name.find("projection") != std::string::npos);

  // Compute the residual: r = b - A*x
  if (!is_tensor_diff) {
    // Use compResidual which handles refluxing and average-down for AMR
    a_mlmg.compResidual(residual_ptrs, a_sol, a_rhs);
  } else {
    // Use solutionResidual computation for tensor diffusion as MLEBTensorOp
    // and MLTensorOp don't implement update() which is needed by compResidual
    amrex::Print()
      << "  NOTE: Using linop.solutionResidual because compResidual"
      << " is not available for tensor diffusion\n";
    if (nlevs > 1) {
      amrex::Print() << "  WARNING: Multi-level residual computed without "
                        "refluxing or average-down.\n";
    }

    for (int lev = 0; lev < nlevs; ++lev) {
      const amrex::MultiFab* crse_bcdata = (lev > 0) ? a_sol[lev - 1] : nullptr;
      linop.solutionResidual(
        lev, residual[lev], *a_sol[lev], *a_rhs[lev], crse_bcdata);
    }
  }

  // Get state data
  auto state_data = GetVecOfConstPtrs(getStateVect(PeleLM::AmrNewTime));
  amrex::Vector<std::string> var_names;
  int ncomp_total = ncomp_residual;
  bool has_eb = false;

  AMREX_ALWAYS_ASSERT(!state_data.empty());
  ncomp_total += state_data[0]->nComp();

#ifdef AMREX_USE_EB
  has_eb = !EBFactory(0).isAllRegular();
  if (has_eb) {
    // Include volFrac
    ncomp_total += 1;
  }
#endif

  for (int n = 0; n < state_data[0]->nComp(); ++n) {
    var_names.push_back(stateVariableName(n));
  }

  // Add volFrac if EB
  if (has_eb) {
    var_names.push_back("volFrac");
  }

  if (is_species) {
    // Get species names
    amrex::Vector<std::string> spec_names;
    pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
      spec_names, &(PeleLM::eos_parms.host_parm()));
    for (int n = 0; n < ncomp_residual; ++n) {
      var_names.push_back("mlmg_residual_" + spec_names[n]);
    }
  } else if (is_tensor_diff) {
    // Get velocity component indices
    for (int n = 0; n < ncomp_residual; ++n) {
      var_names.push_back("mlmg_residual_" + std::to_string(n));
    }
  } else if (is_proj) {
    var_names.push_back("mlmg_residual_" + a_solver_name);
  } else if (is_temp) {
    var_names.push_back("mlmg_residual_temp");
  } else {
    for (int n = 0; n < ncomp_residual; ++n) {
      var_names.push_back("mlmg_residual_comp_" + std::to_string(n));
    }
  }

  // Create combined MultiFabs with state + volFrac + residual
  amrex::Vector<amrex::MultiFab> combined_data(nlevs);
  amrex::Vector<const amrex::MultiFab*> combined_ptrs(nlevs);

  for (int lev = 0; lev < nlevs; ++lev) {
    const auto& ba = a_sol[lev]->boxArray();
    const auto& dm = a_sol[lev]->DistributionMap();
    combined_data[lev].define(ba, dm, ncomp_total, 0);

    int comp_offset = 0;

    // State data
    amrex::MultiFab::Copy(
      combined_data[lev], *state_data[lev], 0, comp_offset,
      state_data[lev]->nComp(), 0);
    comp_offset += state_data[lev]->nComp();

    // volFrac
#ifdef AMREX_USE_EB
    if (has_eb) {
      const auto& vfrac = EBFactory(lev).getVolFrac();
      amrex::MultiFab::Copy(combined_data[lev], vfrac, 0, comp_offset, 1, 0);
      comp_offset += 1;
    }
#endif

    amrex::MultiFab::Copy(
      combined_data[lev], residual[lev], 0, comp_offset, ncomp_residual, 0);
    combined_ptrs[lev] = &combined_data[lev];
  }

  // Extract directory from plot_file_root if it contains a path
  std::string plot_dir;
  std::string plot_base = m_plot_file;
  size_t last_slash = m_plot_file.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    plot_dir = m_plot_file.substr(0, last_slash + 1);
    plot_base = m_plot_file.substr(last_slash + 1);
  }

  // Get iteration count from MLMG
  int num_iters = a_mlmg.getNumIters();

  // Create plotfile name: <plot_dir>/pltMLMGResidual_<solver>_<step>_<iters>
  std::string residual_name = "pltMLMGResidual_" + a_solver_name + "_";
  int ioDigits = m_ioDigits;
  std::string plotfile_name =
    plot_dir + amrex::Concatenate(residual_name, a_step, ioDigits);
  plotfile_name += "_" + amrex::Concatenate("", num_iters, 2);

  amrex::Vector<int> level_steps(nlevs, a_step);

  amrex::WriteMultiLevelPlotfile(
    plotfile_name, nlevs, combined_ptrs, var_names, Geom(),
    0.0, // time
    level_steps, refRatio());
}
