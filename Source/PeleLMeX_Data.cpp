#include <PeleLMeX.H>

PeleLM::LevelData::LevelData(
  amrex::BoxArray const& ba,
  amrex::DistributionMapping const& dm,
  amrex::FabFactory<amrex::FArrayBox> const& factory,
  const int a_incompressible,
  const int a_has_divu,
  const int a_nAux,
  const int a_nGrowState,
  const int a_use_soret,
  const int a_do_les)
{
  if (a_incompressible != 0) {
    state.define(
      ba, dm, AMREX_SPACEDIM, a_nGrowState, amrex::MFInfo(), factory);
  } else {
    state.define(ba, dm, NVAR, a_nGrowState, amrex::MFInfo(), factory);
  }
  gp.define(ba, dm, AMREX_SPACEDIM, 0, amrex::MFInfo(), factory);
  press.define(
    amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, 1, 1,
    amrex::MFInfo(), factory);
  visc_cc.define(ba, dm, 1, 1, amrex::MFInfo(), factory);
  if (a_do_les != 0) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      visc_turb_fc[i].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(i)), dm, 1, 0,
        amrex::MFInfo(), factory);
      if (a_incompressible == 0) {
        lambda_turb_fc[i].define(
          amrex::convert(ba, amrex::IntVect::TheDimensionVector(i)), dm, 1, 0,
          amrex::MFInfo(), factory);
      }
    }
  }
  if (a_incompressible == 0) {
    if (a_has_divu != 0) {
      divu.define(ba, dm, 1, 1, amrex::MFInfo(), factory);
    }
    if (a_use_soret != 0) {
      diff_cc.define(ba, dm, 2 * NUM_SPECIES + 2, 1, amrex::MFInfo(), factory);
    } else {
      diff_cc.define(ba, dm, NUM_SPECIES + 2, 1, amrex::MFInfo(), factory);
    }

#ifdef PELE_USE_PLASMA
    diffE_cc.define(ba, dm, 1, 1, amrex::MFInfo(), factory);
    mobE_cc.define(ba, dm, 1, 1, amrex::MFInfo(), factory);
    mob_cc.define(ba, dm, NUM_IONS, 1, amrex::MFInfo(), factory);
#endif
  }
  if (a_nAux > 0) {
    auxiliaries.define(ba, dm, a_nAux, a_nGrowState, amrex::MFInfo(), factory);
    diff_aux_cc.define(ba, dm, a_nAux, 1, amrex::MFInfo(), factory);
  }
}

PeleLM::LevelDataReact::LevelDataReact(
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm,
  const amrex::FabFactory<amrex::FArrayBox>& factory)
{
#ifdef PELE_USE_PLASMA
  constexpr int IRsize = NUM_SPECIES + 1;
#else
  constexpr int IRsize = NUM_SPECIES;
#endif
  I_R.define(ba, dm, IRsize, 0, amrex::MFInfo(), factory);
  functC.define(ba, dm, 1, 0, amrex::MFInfo(), factory);
}

#ifdef PELE_USE_PLASMA
PeleLM::LevelDataNLSolve::LevelDataNLSolve(
  amrex::BoxArray const& ba,
  amrex::DistributionMapping const& dm,
  amrex::FabFactory<amrex::FArrayBox> const& factory,
  const int a_nGrow)
{
  nlState.define(ba, dm, 2, a_nGrow, amrex::MFInfo(), factory);
  nlResid.define(ba, dm, 2, a_nGrow, amrex::MFInfo(), factory);
  backgroundCharge.define(ba, dm, 1, 0, amrex::MFInfo(), factory);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    const amrex::BoxArray& faceba =
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim));
    gPhiVOld[idim].define(faceba, dm, 1, 0, amrex::MFInfo(), factory);
    uEffnE[idim].define(faceba, dm, 1, 0, amrex::MFInfo(), factory);
    umac[idim].define(faceba, dm, 1, 0, amrex::MFInfo(), factory);
  }
}
#endif

PeleLM::AdvanceDiffData::AdvanceDiffData(
  const int a_finestLevel,
  const amrex::Vector<amrex::BoxArray>& ba,
  const amrex::Vector<amrex::DistributionMapping>& dm,
  const amrex::Vector<std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>>>&
    factory,
  const int nGrowAdv,
  const int a_use_wbar,
  const int a_use_soret,
  const int a_nAux,
  const int is_init,
  const bool a_ext_sources_SDC,
  const int a_nGrowExt)
{
  if (is_init != 0) { // All I need is a container for a single diffusion term
    Dnp1.reserve(a_finestLevel + 1);
    // Define MFs
    for (int lev = 0; lev <= a_finestLevel; ++lev) {
      Dnp1.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 2, nGrowAdv, amrex::MFInfo(),
        *factory[lev]);
    }
    if (a_nAux > 0) {
      Dnp1_aux.reserve(a_finestLevel + 1);
      for (int lev = 0; lev <= a_finestLevel; ++lev) {
        Dnp1_aux.emplace_back(
          ba[lev], dm[lev], a_nAux, nGrowAdv, amrex::MFInfo(), *factory[lev]);
      }
    }
  } else {
    // Reserve/resize Vectors
    Dn.reserve(a_finestLevel + 1);
    Dnp1.reserve(a_finestLevel + 1);
    Dhat.reserve(a_finestLevel + 1);
    if (a_nAux > 0) {
      Dn_aux.reserve(a_finestLevel + 1);
      Dnp1_aux.reserve(a_finestLevel + 1);
      Dhat_aux.reserve(a_finestLevel + 1);
    }
    if (a_use_wbar != 0) {
      Dwbar.reserve(a_finestLevel + 1);
      wbar_fluxes.resize(a_finestLevel + 1);
    }
    if (a_use_soret != 0) {
      DT.reserve(a_finestLevel + 1);
      soret_fluxes.resize(a_finestLevel + 1);
    }

    // Define MFs
    for (int lev = 0; lev <= a_finestLevel; ++lev) {
      Dn.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 2, nGrowAdv, amrex::MFInfo(),
        *factory[lev]);
      Dnp1.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 2, nGrowAdv, amrex::MFInfo(),
        *factory[lev]);
      Dhat.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 2, nGrowAdv, amrex::MFInfo(),
        *factory[lev]);
      if (a_use_wbar != 0) {
        Dwbar.emplace_back(
          ba[lev], dm[lev], NUM_SPECIES, nGrowAdv, amrex::MFInfo(),
          *factory[lev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const amrex::BoxArray& faceba =
            amrex::convert(ba[lev], amrex::IntVect::TheDimensionVector(idim));
          wbar_fluxes[lev][idim].define(
            faceba, dm[lev], NUM_SPECIES, 0, amrex::MFInfo(), *factory[lev]);
        }
      }
      if (a_use_soret != 0) {
        DT.emplace_back(
          ba[lev], dm[lev], NUM_SPECIES, nGrowAdv, amrex::MFInfo(),
          *factory[lev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const amrex::BoxArray& faceba =
            amrex::convert(ba[lev], amrex::IntVect::TheDimensionVector(idim));
          soret_fluxes[lev][idim].define(
            faceba, dm[lev], NUM_SPECIES, 0, amrex::MFInfo(), *factory[lev]);
        }
      }
      if (a_nAux > 0) {
        Dn_aux.emplace_back(
          ba[lev], dm[lev], a_nAux, nGrowAdv, amrex::MFInfo(), *factory[lev]);
        Dnp1_aux.emplace_back(
          ba[lev], dm[lev], a_nAux, nGrowAdv, amrex::MFInfo(), *factory[lev]);
        Dhat_aux.emplace_back(
          ba[lev], dm[lev], a_nAux, nGrowAdv, amrex::MFInfo(), *factory[lev]);
      }
    }
    // ext_sources_SDC: the external sources at t^n and t^{n+1,k}
    if (a_ext_sources_SDC) {
      extSrc_n.reserve(a_finestLevel + 1);
      extSrc_np1.reserve(a_finestLevel + 1);
      for (int lev = 0; lev <= a_finestLevel; ++lev) {
        extSrc_n.emplace_back(
          ba[lev], dm[lev], NVAR, a_nGrowExt, amrex::MFInfo(), *factory[lev]);
        extSrc_np1.emplace_back(
          ba[lev], dm[lev], NVAR, a_nGrowExt, amrex::MFInfo(), *factory[lev]);
      }
    }
  }
}

PeleLM::AdvanceAdvData::AdvanceAdvData(
  const int a_finestLevel,
  const amrex::Vector<amrex::BoxArray>& ba,
  const amrex::Vector<amrex::DistributionMapping>& dm,
  const amrex::Vector<std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>>>&
    factory,
  const int a_incompressible,
  const int a_nAux,
  const int nGrowAdv,
  const int nGrowMAC)
{
  // Resize Vectors
  umac.resize(a_finestLevel + 1);
  AofS.reserve(a_finestLevel + 1);
  if (a_incompressible == 0) {
    chi.reserve(a_finestLevel + 1);
    Forcing.reserve(a_finestLevel + 1);
    mac_divu.reserve(a_finestLevel + 1);
  }
  if (a_nAux > 0) {
    AofS_aux.reserve(a_finestLevel + 1);
    Forcing_aux.reserve(a_finestLevel + 1);
  }
#ifdef PELE_USE_PLASMA
  uDrift.resize(a_finestLevel + 1);
#endif

  // Define MFs
  for (int lev = 0; lev <= a_finestLevel; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::BoxArray& faceba =
        amrex::convert(ba[lev], amrex::IntVect::TheDimensionVector(idim));
      umac[lev][idim].define(
        faceba, dm[lev], 1, nGrowMAC, amrex::MFInfo(), *factory[lev]);
#ifdef PELE_USE_PLASMA
      uDrift[lev][idim].define(
        faceba, dm[lev], NUM_IONS, nGrowMAC, amrex::MFInfo(), *factory[lev]);
#endif
    }
    if (a_incompressible != 0) {
      AofS.emplace_back(
        ba[lev], dm[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), *factory[lev]);
    } else {
      AofS.emplace_back(
        ba[lev], dm[lev], NVAR, 0, amrex::MFInfo(), *factory[lev]);
      chi.emplace_back(ba[lev], dm[lev], 1, 1, amrex::MFInfo(), *factory[lev]);
#ifdef PELE_USE_PLASMA
      Forcing.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 2, nGrowAdv, amrex::MFInfo(),
        *factory[lev]); // Species + TEMP + nE
#else
      Forcing.emplace_back(
        ba[lev], dm[lev], NUM_SPECIES + 1, nGrowAdv, amrex::MFInfo(),
        *factory[lev]); // Species + TEMP
#endif
      mac_divu.emplace_back(
        ba[lev], dm[lev], 1, nGrowAdv, amrex::MFInfo(), *factory[lev]);
    }
    if (a_nAux > 0) {
      AofS_aux.emplace_back(
        ba[lev], dm[lev], a_nAux, 0, amrex::MFInfo(), *factory[lev]);
      Forcing_aux.emplace_back(
        ba[lev], dm[lev], a_nAux, nGrowAdv, amrex::MFInfo(), *factory[lev]);
    }
  }
}

void
PeleLM::copyStateNewToOld(int nGhost)
{
  AMREX_ASSERT(nGhost <= m_nGrowState);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_incompressible != 0) {
      amrex::MultiFab::Copy(
        m_leveldata_old[lev]->state, m_leveldata_new[lev]->state, 0, 0,
        AMREX_SPACEDIM, nGhost);
    } else {
      amrex::MultiFab::Copy(
        m_leveldata_old[lev]->state, m_leveldata_new[lev]->state, 0, 0, NVAR,
        nGhost);
      if (m_has_divu != 0) {
        amrex::MultiFab::Copy(
          m_leveldata_old[lev]->divu, m_leveldata_new[lev]->divu, 0, 0, 1,
          amrex::min(nGhost, 1));
      }
    }
    if (m_nAux > 0) {
      amrex::MultiFab::Copy(
        m_leveldata_old[lev]->auxiliaries, m_leveldata_new[lev]->auxiliaries, 0,
        0, m_nAux, nGhost);
    }
  }
}

void
PeleLM::copyPressNewToOld()
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(
      m_leveldata_old[lev]->press, m_leveldata_new[lev]->press, 0, 0, 1, 1);
    amrex::MultiFab::Copy(
      m_leveldata_old[lev]->gp, m_leveldata_new[lev]->gp, 0, 0, AMREX_SPACEDIM,
      0);
  }
}

void
PeleLM::copyStateOldToNew(int nGhost)
{
  AMREX_ASSERT(nGhost <= m_nGrowState);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_incompressible != 0) {
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->state, m_leveldata_old[lev]->state, 0, 0,
        AMREX_SPACEDIM, nGhost);
    } else {
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->state, m_leveldata_old[lev]->state, 0, 0, NVAR,
        nGhost);
      if (m_has_divu != 0) {
        amrex::MultiFab::Copy(
          m_leveldata_new[lev]->divu, m_leveldata_old[lev]->divu, 0, 0, 1,
          amrex::min(nGhost, 1));
      }
    }
    if (m_nAux > 0) {
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->auxiliaries, m_leveldata_old[lev]->auxiliaries, 0,
        0, m_nAux, nGhost);
    }
  }
}

void
PeleLM::copyTransportOldToNew()
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(
      m_leveldata_new[lev]->visc_cc, m_leveldata_old[lev]->visc_cc, 0, 0, 1, 1);
    if (m_incompressible == 0) {
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->diff_cc, m_leveldata_old[lev]->diff_cc, 0, 0,
        NUM_SPECIES + 2, 1);
#ifdef PELE_USE_PLASMA
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->diffE_cc, m_leveldata_old[lev]->diffE_cc, 0, 0, 1,
        1);
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->mobE_cc, m_leveldata_old[lev]->mobE_cc, 0, 0, 1,
        1);
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->mob_cc, m_leveldata_old[lev]->mob_cc, 0, 0,
        NUM_IONS, 1);
#endif
    }
    if (m_nAux > 0) {
      amrex::MultiFab::Copy(
        m_leveldata_new[lev]->diff_aux_cc, m_leveldata_old[lev]->diff_aux_cc, 0,
        0, m_nAux, 1);
    }
  }
}

void
PeleLM::copyDiffusionOldToNew(const std::unique_ptr<AdvanceDiffData>& diffData)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(
      diffData->Dnp1[lev], diffData->Dn[lev], 0, 0, NUM_SPECIES + 2,
      m_nGrowAdv);
    if (m_nAux > 0) {
      amrex::MultiFab::Copy(
        diffData->Dnp1_aux[lev], diffData->Dn_aux[lev], 0, 0, m_nAux,
        m_nGrowAdv);
    }
  }
}

bool
PeleLM::checkForNaNs()
{
  bool contains_nan = false;
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (
      m_leveldata_new[lev]->state.contains_nan() ||
      m_leveldata_new[lev]->state.contains_inf()) {
      contains_nan = true;
    }
  }
  return contains_nan;
}
