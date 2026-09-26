#include <PeleLMeX.H>
#include <PeleLMeX_BPatch.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

amrex::Real
PeleLM::MFSumMapped(
  const amrex::Vector<const amrex::MultiFab*>& a_MF, const int comp)
{
  if ((!m_mesh_mapping) || (m_mesh_map == nullptr)) {
    return MFSum(a_MF, comp);
  }
  // detJ-weight each level so MFSum (which applies the uniform Xi cell
  // volume and EB vfrac) returns the physical integral sum(field*detJ*..).
  // That is the quantity that telescopes against the Xi-space boundary
  // fluxes in addMassFluxes (d/dt integral == net flux to round-off).
  amrex::Vector<amrex::MultiFab> wgt(finest_level + 1);
  amrex::Vector<const amrex::MultiFab*> wgt_p(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    wgt[lev].define(grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    amrex::MultiFab::Copy(wgt[lev], *a_MF[lev], comp, 0, 1, 0);
    amrex::MultiFab::Multiply(wgt[lev], m_mesh_map->detJ_cc(lev), 0, 0, 1, 0);
    wgt_p[lev] = &wgt[lev];
  }
  return MFSum(wgt_p, 0);
}

void
PeleLM::initTemporals(const PeleLM::TimeStamp a_time)
{
  if ((m_do_temporals == 0) || !((m_nstep + 1) % m_temp_int == 0)) {
    return;
  }

  // Reset mass fluxes integrals on domain boundaries
  if ((m_do_massBalance != 0) && (m_incompressible == 0)) {
    m_massOld = MFSumMapped(GetVecOfConstPtrs(getDensityVect(a_time)), 0);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_domainMassFlux[2 * idim] = 0.0;
      m_domainMassFlux[2 * idim + 1] = 0.0;
    }
  }
  if ((m_do_energyBalance != 0) && (m_incompressible == 0)) {
    m_RhoHOld = MFSumMapped(GetVecOfConstPtrs(getRhoHVect(a_time)), 0);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_domainRhoHFlux[2 * idim] = 0.0;
      m_domainRhoHFlux[2 * idim + 1] = 0.0;
    }
    m_domainRhoHFluxEB = 0.0;
  }

  if ((m_do_speciesBalance != 0) && (m_incompressible == 0)) {
    for (int n = 0; n < NUM_SPECIES; ++n) {
      m_RhoYOld[n] = MFSumMapped(GetVecOfConstPtrs(getSpeciesVect(a_time)), n);
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 2 * idim] = 0.0;
        m_domainRhoYFlux[1 + 2 * n * AMREX_SPACEDIM + 2 * idim] = 0.0;
      }
    }
  }
}

void
PeleLM::massBalance()
{
  // Compute the mass balance on the computational domain
  m_massNew = MFSumMapped(GetVecOfConstPtrs(getDensityVect(AmrNewTime)), 0);
  amrex::Real dmdt = (m_massNew - m_massOld) / m_dt;
  amrex::Real massFluxBalance = AMREX_D_TERM(
    m_domainMassFlux[0] + m_domainMassFlux[1],
    +m_domainMassFlux[2] + m_domainMassFlux[3],
    +m_domainMassFlux[4] + m_domainMassFlux[5]);

  tmpMassFile << m_nstep << "," << m_cur_time // Time info
              << "," << m_massNew             // mass
              << "," << dmdt                  // mass temporal derivative
              << "," << massFluxBalance       // domain boundaries mass fluxes
              << "," << std::abs(dmdt - massFluxBalance) << "\n"; // balance
  tmpMassFile.flush();
}

void
PeleLM::speciesBalancePatch()
{
  tmppatchmfrFile << m_nstep << "," << m_cur_time; // Time info
  for (const auto& m_bPatche : m_bPatches) {
    BPatch::BpatchDataContainer* bphost = m_bPatche->getHostDataPtr();
    for (int i = 0; i < bphost->num_groups; ++i) {
      tmppatchmfrFile << ",";
      amrex::Real tmp = 0.0;
      for (const auto& mechSpeciesIdx : m_bPatche->speciesinGroup[i]) {
        for (int k = 0; k < bphost->num_species; k++) {
          if (bphost->speciesIndex[k] == mechSpeciesIdx) {
            tmp += bphost->speciesFlux[k];
            break;
          }
        }
      }
      tmppatchmfrFile << tmp;
    }
  }
  tmppatchmfrFile << "\n";
  tmppatchmfrFile.flush();
}

void
PeleLM::speciesBalance()
{
  // Compute the species rhoY balance on the computational domain
  amrex::Array<amrex::Real, NUM_SPECIES> dmYdt;
  amrex::Array<amrex::Real, NUM_SPECIES> massYFluxBalance;
  amrex::Array<amrex::Real, NUM_SPECIES> rhoYdots;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    m_RhoYNew[n] =
      MFSumMapped(GetVecOfConstPtrs(getSpeciesVect(AmrNewTime)), n);
    // Reaction source is a volumetric integral too -> detJ-weighted.
    rhoYdots[n] = MFSumMapped(GetVecOfConstPtrs(getIRVect()), n);
    dmYdt[n] = (m_RhoYNew[n] - m_RhoYOld[n]) / m_dt;
    massYFluxBalance[n] = AMREX_D_TERM(
      m_domainRhoYFlux[2 * n * AMREX_SPACEDIM] +
        m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 1],
      +m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 2] +
        m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 3],
      +m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 4] +
        m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + 5]);
  }

  tmpSpecFile << m_nstep << "," << m_cur_time; // Time info
  for (int n = 0; n < NUM_SPECIES; ++n) {
    tmpSpecFile << "," << m_RhoYNew[n]        // mass of Y
                << "," << dmYdt[n]            // mass temporal derivative
                << "," << massYFluxBalance[n] // domain boundaries mass fluxes
                << "," << rhoYdots[n]         // integrated consumption rate
                << ","
                << std::abs(
                     dmYdt[n] - massYFluxBalance[n] - rhoYdots[n]); // balance
  }
  tmpSpecFile << "\n";
  tmpSpecFile.flush();
}

void
PeleLM::addMassFluxes(
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::Geometry& a_geom)
{

  // Do when m_nstep is -1 since m_nstep is increased by one before
  // the writeTemporals
  if (!(m_nstep % m_temp_int == m_temp_int - 1)) {
    return;
  }

  // Get the face areas
  const amrex::Real* dx = a_geom.CellSize();
  amrex::Array<amrex::Real, AMREX_SPACEDIM> area;
#if (AMREX_SPACEDIM == 1)
  area[0] = 1.0;
#elif (AMREX_SPACEDIM == 2)
  area[0] = dx[1];
  area[1] = dx[0];
#else
  area[0] = dx[1] * dx[2];
  area[1] = dx[0] * dx[2];
  area[2] = dx[0] * dx[1];
#endif

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    auto faceDomain =
      amrex::convert(a_geom.Domain(), amrex::IntVect::TheDimensionVector(idim));

    auto const& fma = a_fluxes[idim]->const_arrays();

    amrex::Real sumLo = 0.0;
    amrex::Real sumHi = 0.0;

#if (AMREX_SPACEDIM == 2)
    if (geom[0].IsRZ()) {
      amrex::MultiFab mf_a;
      geom[0].GetFaceArea(mf_a, grids[0], dmap[0], idim, 0);
      auto const& ama = mf_a.const_arrays();
      auto r = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
        amrex::IntVect(0),
        [fma, ama, idim, faceDomain] AMREX_GPU_DEVICE(
          int box_no, int i, int j,
          int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
          amrex::Array4<const amrex::Real> const& flux = fma[box_no];
          amrex::Array4<const amrex::Real> const& area_ar = ama[box_no];

          int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
          // low
          amrex::Real low = 0.0;
          if (idx == faceDomain.smallEnd(idim)) {
            for (int n = 0; n < NUM_SPECIES; ++n) {
              low += flux(i, j, k, n) * area_ar(i, j, k);
            }
          }
          // high
          amrex::Real high = 0.0;
          if (idx == faceDomain.bigEnd(idim)) {
            for (int n = 0; n < NUM_SPECIES; ++n) {
              high += flux(i, j, k, n) * area_ar(i, j, k);
            }
          }
          return {low, high};
        });
      sumLo = amrex::get<0>(r);
      sumHi = amrex::get<1>(r);
    } else
#endif
    {
      auto r = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
        amrex::IntVect(0),
        [fma, idim, faceDomain, area] AMREX_GPU_DEVICE(
          int box_no, int i, int j,
          int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
          amrex::Array4<const amrex::Real> const& flux = fma[box_no];

          int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
          // low
          amrex::Real low = 0.0;
          if (idx == faceDomain.smallEnd(idim)) {
            for (int n = 0; n < NUM_SPECIES; ++n) {
              low += flux(i, j, k, n) * area[idim];
            }
          }
          // high
          amrex::Real high = 0.0;
          if (idx == faceDomain.bigEnd(idim)) {
            for (int n = 0; n < NUM_SPECIES; ++n) {
              high += flux(i, j, k, n) * area[idim];
            }
          }
          return {low, high};
        });
      sumLo = amrex::get<0>(r);
      sumHi = amrex::get<1>(r);
    }
    amrex::ParallelAllReduce::Sum<amrex::Real>(
      {sumLo, sumHi}, amrex::ParallelContext::CommunicatorSub());
    m_domainMassFlux[2 * idim] += sumLo;
    m_domainMassFlux[2 * idim + 1] -= sumHi; // Outflow, negate flux
  }
}

void
PeleLM::addUmacFluxes(
  const std::unique_ptr<AdvanceAdvData>& advData, const amrex::Geometry& a_geom)
{
  // Get the face areas
  const amrex::Real* dx = a_geom.CellSize();
  amrex::Array<amrex::Real, AMREX_SPACEDIM> area;
#if (AMREX_SPACEDIM == 1)
  area[0] = 1.0;
#elif (AMREX_SPACEDIM == 2)
  area[0] = dx[1];
  area[1] = dx[0];
#else
  area[0] = dx[1] * dx[2];
  area[1] = dx[0] * dx[2];
  area[2] = dx[0] * dx[1];
#endif

  // Just use level 0 since we are calling after averaging down
  constexpr int lev = 0;

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    auto faceDomain =
      amrex::convert(a_geom.Domain(), amrex::IntVect::TheDimensionVector(idim));

    auto const& fma = advData->umac[lev][idim].const_arrays();

    amrex::Real sumLo = 0.0;
    amrex::Real sumHi = 0.0;

#if (AMREX_SPACEDIM == 2)
    if (geom[0].IsRZ()) {
      amrex::MultiFab mf_a;
      geom[0].GetFaceArea(mf_a, grids[0], dmap[0], idim, 0);
      auto const& ama = mf_a.const_arrays();
      auto r = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real>{}, advData->umac[lev][idim],
        amrex::IntVect(0),
        [fma, ama, idim, faceDomain] AMREX_GPU_DEVICE(
          int box_no, int i, int j,
          int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
          amrex::Array4<const amrex::Real> const& flux = fma[box_no];
          amrex::Array4<const amrex::Real> const& area_ar = ama[box_no];

          int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
          // low
          amrex::Real low = 0.0;
          if (idx == faceDomain.smallEnd(idim)) {
            low += flux(i, j, k) * area_ar(i, j, k);
          }
          // high
          amrex::Real high = 0.0;
          if (idx == faceDomain.bigEnd(idim)) {
            high += flux(i, j, k) * area_ar(i, j, k);
          }
          return {low, high};
        });
      sumLo = amrex::get<0>(r);
      sumHi = amrex::get<1>(r);
    } else
#endif
    {
      auto r = amrex::ParReduce(
        amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
        amrex::TypeList<amrex::Real, amrex::Real>{}, advData->umac[lev][idim],
        amrex::IntVect(0),
        [fma, idim, faceDomain, area] AMREX_GPU_DEVICE(
          int box_no, int i, int j,
          int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
          amrex::Array4<const amrex::Real> const& flux = fma[box_no];

          int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
          // low
          amrex::Real low = 0.0;
          if (idx == faceDomain.smallEnd(idim)) {
            low += flux(i, j, k) * area[idim];
          }
          // high
          amrex::Real high = 0.0;
          if (idx == faceDomain.bigEnd(idim)) {
            high += flux(i, j, k) * area[idim];
          }
          return {low, high};
        });
      sumLo = amrex::get<0>(r);
      sumHi = amrex::get<1>(r);
    }
    amrex::ParallelAllReduce::Sum<amrex::Real>(
      {sumLo, sumHi}, amrex::ParallelContext::CommunicatorSub());
    m_domainUmacFlux[2 * idim] += sumLo;
    m_domainUmacFlux[2 * idim + 1] -= sumHi; // Outflow, negate flux
  }
}

void
PeleLM::rhoHBalance()
{
  // Compute the enthalpy balance on the computational domain (rho*h)
  m_RhoHNew = MFSumMapped(GetVecOfConstPtrs(getRhoHVect(AmrNewTime)), 0);
  amrex::Real dRhoHdt = (m_RhoHNew - m_RhoHOld) / m_dt;
  amrex::Real rhoHFluxBalance = AMREX_D_TERM(
    m_domainRhoHFlux[0] + m_domainRhoHFlux[1],
    +m_domainRhoHFlux[2] + m_domainRhoHFlux[3],
    +m_domainRhoHFlux[4] + m_domainRhoHFlux[5]);
  rhoHFluxBalance += m_domainRhoHFluxEB; // Isothermal EB conduction

  tmpEnergyFile << m_nstep << "," << m_cur_time // Time info
                << "," << m_RhoHNew             // RhoH
                << "," << dRhoHdt               // RhoH temporal derivative
                << "," << rhoHFluxBalance       // domain boundaries RhoH fluxes
                << "," << std::abs(dRhoHdt - rhoHFluxBalance)
                << "\n"; // balance
  tmpEnergyFile.flush();
}

void
PeleLM::addRhoHFluxes(
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::Geometry& a_geom,
  const amrex::Real& a_factor,
  const int a_nComp)
{
  // Accumulate the rho*h flux components [NUM_SPECIES, NUM_SPECIES+a_nComp)
  // of a_fluxes: a single advective rho*h flux, or the Fourier and
  // differential diffusion enthalpy fluxes from the diffusion solve.

  // Do when m_nstep is -1 since m_nstep is increased by one before
  // the writeTemporals
  if (!(m_nstep % m_temp_int == m_temp_int - 1)) {
    return;
  }

  // Get the face areas
  const amrex::Real* dx = a_geom.CellSize();
  amrex::Array<amrex::Real, AMREX_SPACEDIM> area;
#if (AMREX_SPACEDIM == 1)
  area[0] = 1.0;
#elif (AMREX_SPACEDIM == 2)
  area[0] = dx[1];
  area[1] = dx[0];
#else
  area[0] = dx[1] * dx[2];
  area[1] = dx[0] * dx[2];
  area[2] = dx[0] * dx[1];
#endif

  for (int n = NUM_SPECIES; n < NUM_SPECIES + a_nComp; ++n) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto faceDomain = amrex::convert(
        a_geom.Domain(), amrex::IntVect::TheDimensionVector(idim));

      auto const& fma = a_fluxes[idim]->const_arrays();

      amrex::Real sumLo = 0.0;
      amrex::Real sumHi = 0.0;

#if (AMREX_SPACEDIM == 2)
      if (geom[0].IsRZ()) {
        amrex::MultiFab mf_a;
        geom[0].GetFaceArea(mf_a, grids[0], dmap[0], idim, 0);
        auto const& ama = mf_a.const_arrays();
        auto r = amrex::ParReduce(
          amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
          amrex::IntVect(0),
          [fma, ama, idim, faceDomain, n] AMREX_GPU_DEVICE(
            int box_no, int i, int j,
            int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            amrex::Array4<const amrex::Real> const& flux = fma[box_no];
            amrex::Array4<const amrex::Real> const& area_ar = ama[box_no];

            int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
            // low
            amrex::Real low = 0.0;
            if (idx == faceDomain.smallEnd(idim)) {
              low += flux(i, j, k, n) * area_ar(i, j, k);
            }
            // high
            amrex::Real high = 0.0;
            if (idx == faceDomain.bigEnd(idim)) {
              high += flux(i, j, k, n) * area_ar(i, j, k);
            }
            return {low, high};
          });
        sumLo = amrex::get<0>(r);
        sumHi = amrex::get<1>(r);
      } else
#endif
      {
        auto r = amrex::ParReduce(
          amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
          amrex::IntVect(0),
          [fma, idim, faceDomain, area, n] AMREX_GPU_DEVICE(
            int box_no, int i, int j,
            int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            amrex::Array4<const amrex::Real> const& flux = fma[box_no];

            int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
            // low
            amrex::Real low = 0.0;
            if (idx == faceDomain.smallEnd(idim)) {
              low += flux(i, j, k, n) * area[idim];
            }
            // high
            amrex::Real high = 0.0;
            if (idx == faceDomain.bigEnd(idim)) {
              high += flux(i, j, k, n) * area[idim];
            }
            return {low, high};
          });
        sumLo = amrex::get<0>(r);
        sumHi = amrex::get<1>(r);
      }
      amrex::ParallelAllReduce::Sum<amrex::Real>(
        {sumLo, sumHi}, amrex::ParallelContext::CommunicatorSub());
      m_domainRhoHFlux[2 * idim] += a_factor * sumLo;
      m_domainRhoHFlux[2 * idim + 1] -=
        a_factor * sumHi; // Outflow, negate flux
    }
  }
}

#ifdef AMREX_USE_EB
void
PeleLM::addRhoHFluxesEB(
  const amrex::Vector<amrex::MultiFab*>& a_EBfluxes,
  const amrex::Real& a_factor)
{
  // Accumulate the Fourier enthalpy flux through isothermal EB surfaces
  // on all levels, excluding cells covered by a finer level, using the
  // same sign convention as the domain boundary fluxes (outflow negative)

  // Do when m_nstep is -1 since m_nstep is increased by one before
  // the writeTemporals
  if (!(m_nstep % m_temp_int == m_temp_int - 1)) {
    return;
  }

  amrex::Real sumEB = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& ebfact = EBFactory(lev);
    auto const& flags = ebfact.getMultiEBCellFlagFab();
    auto const& bndryArea = ebfact.getBndryArea();
    const amrex::Real* dx = geom[lev].CellSize();
    const amrex::Real ebArea = AMREX_D_TERM(1.0, *dx[0], *dx[0]);
    const int isFinest = (lev == finest_level) ? 1 : 0;

    amrex::MultiFab ebFluxArea(grids[lev], dmap[lev], 1, 0);
    ebFluxArea.setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ebFluxArea, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      if (flags[mfi].getType(bx) != amrex::FabType::singlevalued) {
        continue;
      }
      auto const& ebflux = a_EBfluxes[lev]->const_array(mfi);
      auto const& barea = bndryArea.const_array(mfi);
      auto const& fa = ebFluxArea.array(mfi);
      auto const& covered = (isFinest != 0)
                              ? amrex::Array4<int const>{}
                              : m_coveredMask[lev]->const_array(mfi);
      amrex::ParallelFor(
        bx, [ebflux, barea, fa, covered, ebArea,
             isFinest] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const amrex::Real mask =
            (isFinest != 0) ? 1.0 : static_cast<amrex::Real>(covered(i, j, k));
          fa(i, j, k) = ebflux(i, j, k) * barea(i, j, k) * ebArea * mask;
        });
    }
    sumEB += ebFluxArea.sum(0, true);
  }
  amrex::ParallelDescriptor::ReduceRealSum(sumEB);
  m_domainRhoHFluxEB -= a_factor * sumEB; // Outflow, negate flux
}
#endif

void
PeleLM::addRhoYFluxes(
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::Geometry& a_geom,
  const amrex::Real& a_factor)
{

  // Do when m_nstep is -1 since m_nstep is increased by one before
  // the writeTemporals
  if (!(m_nstep % m_temp_int == m_temp_int - 1)) {
    return;
  }

  // Get the face areas
  const amrex::Real* dx = a_geom.CellSize();
  amrex::Array<amrex::Real, AMREX_SPACEDIM> area;
#if (AMREX_SPACEDIM == 1)
  area[0] = 1.0;
#elif (AMREX_SPACEDIM == 2)
  area[0] = dx[1];
  area[1] = dx[0];
#else
  area[0] = dx[1] * dx[2];
  area[1] = dx[0] * dx[2];
  area[2] = dx[0] * dx[1];
#endif

  // Outer loop over species
  for (int n = 0; n < NUM_SPECIES; ++n) {
    // Inner loop over dimensions
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto faceDomain = amrex::convert(
        a_geom.Domain(), amrex::IntVect::TheDimensionVector(idim));

      auto const& fma = a_fluxes[idim]->const_arrays();

      amrex::Real sumLo = 0.0;
      amrex::Real sumHi = 0.0;

#if (AMREX_SPACEDIM == 2)
      if (geom[0].IsRZ()) {
        amrex::MultiFab mf_a;
        geom[0].GetFaceArea(mf_a, grids[0], dmap[0], idim, 0);
        auto const& ama = mf_a.const_arrays();
        auto r = amrex::ParReduce(
          amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
          amrex::IntVect(0),
          [fma, ama, idim, faceDomain, n] AMREX_GPU_DEVICE(
            int box_no, int i, int j,
            int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            amrex::Array4<const amrex::Real> const& flux = fma[box_no];
            amrex::Array4<const amrex::Real> const& area_ar = ama[box_no];

            int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
            // low
            amrex::Real low = 0.0;
            if (idx == faceDomain.smallEnd(idim)) {
              low += flux(i, j, k, n) * area_ar(i, j, k);
            }
            // high
            amrex::Real high = 0.0;
            if (idx == faceDomain.bigEnd(idim)) {
              high += flux(i, j, k, n) * area_ar(i, j, k);
            }
            return {low, high};
          });
        sumLo = amrex::get<0>(r);
        sumHi = amrex::get<1>(r);
      } else
#endif
      {
        auto r = amrex::ParReduce(
          amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
          amrex::IntVect(0),
          [fma, idim, faceDomain, area, n] AMREX_GPU_DEVICE(
            int box_no, int i, int j,
            int k) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            amrex::Array4<const amrex::Real> const& flux = fma[box_no];

            int idx = (idim == 0) ? i : ((idim == 1) ? j : k);
            // low
            amrex::Real low = 0.0;
            if (idx == faceDomain.smallEnd(idim)) {
              low += flux(i, j, k, n) * area[idim];
            }
            // high
            amrex::Real high = 0.0;
            if (idx == faceDomain.bigEnd(idim)) {
              high += flux(i, j, k, n) * area[idim];
            }
            return {low, high};
          });
        sumLo = amrex::get<0>(r);
        sumHi = amrex::get<1>(r);
      }
      amrex::ParallelAllReduce::Sum<amrex::Real>(
        {sumLo, sumHi}, amrex::ParallelContext::CommunicatorSub());
      m_domainRhoYFlux[2 * idim + n * 2 * AMREX_SPACEDIM] += a_factor * sumLo;
      m_domainRhoYFlux[2 * idim + n * 2 * AMREX_SPACEDIM + 1] -=
        a_factor * sumHi; // Outflow, negate flux
    }
  }
}

void
PeleLM::initBPatches(
  const amrex::Geometry& a_geom,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>* eosparms_h,
  const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
    eosparms_d)
{
  std::string pele_prefix = "peleLM.bpatch";
  amrex::ParmParse pp(pele_prefix);
  int num_bPatches = 0;
  num_bPatches = pp.countval("patchnames");

  amrex::Vector<std::string> bpatch_name;
  if (num_bPatches > 0) {

    m_bPatches.resize(num_bPatches);
    bpatch_name.resize(num_bPatches);
  }
  for (int n = 0; n < num_bPatches; ++n) {
    pp.get("patchnames", bpatch_name[n], n);
    m_bPatches[n] =
      std::make_unique<BPatch>(bpatch_name[n], a_geom, eosparms_h, eosparms_d);
    if (m_verbose > 0) {
      amrex::Print() << " Initializing boundary patch: " << bpatch_name[n]
                     << "\n";
    }
  }
}

void
PeleLM::addRhoYFluxesPatch(
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::Geometry& a_geom,
  const amrex::Real& a_factor)
{

  if (!(m_nstep % m_temp_int == m_temp_int - 1)) {
    return;
  }

  const auto dx = a_geom.CellSizeArray();
  auto prob_lo = a_geom.ProbLoArray();
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> area;

#if (AMREX_SPACEDIM == 1)
  area[0] = 1.0;
#elif (AMREX_SPACEDIM == 2)
  if (geom[0].IsRZ() && m_bPatches.size() > 0) {
    amrex::Abort("Bpatches not supported in RZ coordinates");
  }
  area[0] = dx[1];
  area[1] = dx[0];
#else
  area[0] = dx[1] * dx[2];
  area[1] = dx[0] * dx[2];
  area[2] = dx[0] * dx[1];
#endif

  // Loop through all patches
  for (const auto& m_bPatche : m_bPatches) {

    BPatch* patch = m_bPatche.get();
    BPatch::BpatchDataContainer const* bpdevice = patch->getDeviceData();
    BPatch::BpatchDataContainer const* bphost = patch->getHostDataPtr();
    const int idim = bphost->m_boundary_dir;

    auto faceDomain =
      amrex::convert(a_geom.Domain(), amrex::IntVect::TheDimensionVector(idim));
    auto const& fma = a_fluxes[idim]->const_arrays();

    // Loop through species specified by user
    for (int m = 0; m < bphost->num_species; ++m) {

      amrex::Real sum_species_flux_global = 0.0;

      {
        auto r = amrex::ParReduce(
          amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real>{}, *a_fluxes[idim],
          amrex::IntVect(0),
          [fma, bpdevice, faceDomain, idim, prob_lo, dx, m,
           area] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
            -> amrex::GpuTuple<amrex::Real, amrex::Real> {
            amrex::Array4<const amrex::Real> const& flux = fma[box_no];
            int idx =
              (bpdevice->m_boundary_dir == 0
                 ? i
                 : (bpdevice->m_boundary_dir == 1 ? j : k));
            int idx_lo_hi =
              (bpdevice->m_boundary_lo_hi == 0 ? faceDomain.smallEnd(idim)
                                               : faceDomain.bigEnd(idim));

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> point_coordinates{
              AMREX_D_DECL(
                prob_lo[0] + (i + 0.5_rt) * dx[0],
                prob_lo[1] + (j + 0.5_rt) * dx[1],
                prob_lo[2] + (k + 0.5_rt) * dx[2])};

            amrex::Real sum_species_flux = 0.0;
            amrex::Real dummy = 0.0;
            const bool ifinside =
              bpdevice->CheckifPointInside(point_coordinates, dx[0]);

            if (idx == idx_lo_hi and ifinside) {
              int species_idx = bpdevice->speciesIndex[m];
              sum_species_flux += flux(i, j, k, species_idx) * area[idim];
            }
            return {sum_species_flux, dummy};
          });
        sum_species_flux_global = amrex::get<0>(r);

        amrex::ParallelAllReduce::Sum<amrex::Real>(
          {sum_species_flux_global}, amrex::ParallelContext::CommunicatorSub());
        bphost->speciesFlux[m] = a_factor * sum_species_flux_global;
      }
    }
  }
}

void
PeleLM::writeTemporals()
{
  if (m_verbose > 1) {
    amrex::Print() << " Writing temporals." << std::endl;
  }

  //----------------------------------------------------------------
  // Mass balance
  if ((m_do_massBalance != 0) && (m_incompressible == 0)) {
    massBalance();
  }

  //----------------------------------------------------------------
  // Energy (rho*h) balance
  if ((m_do_energyBalance != 0) && (m_incompressible == 0)) {
    rhoHBalance();
  }

  //----------------------------------------------------------------
  // Species balance
  if ((m_do_speciesBalance != 0) && (m_incompressible == 0)) {
    speciesBalance();
  }

  // Species balance
  if ((m_do_patch_mfr != 0) && (m_incompressible == 0)) {
    speciesBalancePatch();
  }

  //----------------------------------------------------------------
  // State
  // Get kinetic energy and enstrophy
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> kinEnergy(finest_level + 1);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> enstrophy(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    kinEnergy[lev] = derive("kinetic_energy", m_cur_time, lev, 0);
    enstrophy[lev] = derive("enstrophy", m_cur_time, lev, 0);
  }
  amrex::Real kinenergy_int = MFSum(GetVecOfConstPtrs(kinEnergy), 0);
  amrex::Real enstrophy_int = MFSum(GetVecOfConstPtrs(enstrophy), 0);

  // Combustion
  amrex::Real fuelConsumptionInt = 0.0;
  amrex::Real heatReleaseRateInt = 0.0;

#ifdef USE_MANIFOLD_EOS
  if (!(m_chem_integrator == "ReactorNull")) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      getHeatRelease(lev, kinEnergy[lev].get()); // Reuse kinEnergy container
    }
    heatReleaseRateInt = MFSum(GetVecOfConstPtrs(kinEnergy), 0);
  }
#else
  if (fuelID >= 0 && !(m_chem_integrator == "ReactorNull")) {
    fuelConsumptionInt = MFSum(GetVecOfConstPtrs(getIRVect()), fuelID);
    for (int lev = 0; lev <= finest_level; ++lev) {
      getHeatRelease(lev, kinEnergy[lev].get()); // Reuse kinEnergy container
    }
    heatReleaseRateInt = MFSum(GetVecOfConstPtrs(kinEnergy), 0);
  }
#endif

  tmpStateFile << m_nstep << "," << m_cur_time << "," << m_dt // Time
               << "," << kinenergy_int                        // Kinetic energy
               << "," << enstrophy_int                        // Enstrophy
               << "," << m_pNew             // Thermo. pressure
               << "," << fuelConsumptionInt // Integ fuel burning rate
               << "," << heatReleaseRateInt // Integ heat release rate
               << "\n";
  tmpStateFile.flush();

  // Get min/max for state components
  auto stateMax =
    (m_incompressible) != 0
      ? MLmax(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, AMREX_SPACEDIM)
      : MLmax(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, NVAR);
  auto stateMin =
    (m_incompressible) != 0
      ? MLmin(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, AMREX_SPACEDIM)
      : MLmin(GetVecOfConstPtrs(getStateVect(AmrNewTime)), 0, NVAR);

  tmpExtremasFile << m_nstep << "," << m_cur_time; // Time
  for (int n = 0; n < stateMax.size();
       ++n) { // Min & max of each state variable
    tmpExtremasFile << "," << stateMin[n] << "," << stateMax[n];
  }
  tmpExtremasFile << "\n";
  tmpExtremasFile.flush();

#ifdef PELE_USE_PLASMA
  if (m_do_ionsBalance) {
    ionsBalance();
  }
#endif
}

void
PeleLM::openTempFile()
{
  if (m_do_temporals == 0) {
    return;
  }

  // Create the temporal directory
  amrex::UtilCreateDirectory(m_temporal_dir, 0755);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::string tempFileName = m_temporal_dir + "/tempState";
    tmpStateFile.open(
      tempFileName.c_str(),
      std::ios::out | std::ios::app | std::ios_base::binary);
    tmpStateFile.precision(12);
    tmpStateFile << "iter,time,dt,kinEnergy,enstrophy,pressure,fuelConsumption,"
                    "heatRelease\n";
    if (m_do_massBalance != 0) {
      tempFileName = m_temporal_dir + "/tempMass";
      tmpMassFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmpMassFile.precision(12);
      tmpMassFile << "iter,time,massNew,dmdt,netMassFlux,balance\n";
    }
    if (m_do_energyBalance != 0) {
      tempFileName = m_temporal_dir + "/tempEnergy";
      tmpEnergyFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmpEnergyFile.precision(12);
      tmpEnergyFile << "iter,time,rhoHNew,drhoHdt,netRhoHFlux,balance\n";
    }
    if (m_do_speciesBalance != 0) {
      tempFileName = m_temporal_dir + "/tempSpecies";
      tmpSpecFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmpSpecFile.precision(12);
      tmpSpecFile << "iter,time";
      for (int n = 0; n < NUM_SPECIES; ++n) {
        tmpSpecFile << ",rhoYnew_" << PeleLM::stateVariableName(FIRSTSPEC + n);
        tmpSpecFile << ",drhoYdt_" << PeleLM::stateVariableName(FIRSTSPEC + n);
        tmpSpecFile << ",netFlux_" << PeleLM::stateVariableName(FIRSTSPEC + n);
        tmpSpecFile << ",rxnCons_" << PeleLM::stateVariableName(FIRSTSPEC + n);
        tmpSpecFile << ",balance_" << PeleLM::stateVariableName(FIRSTSPEC + n);
      }
      tmpSpecFile << "\n";
    }
    if (m_do_extremas != 0) {
      tempFileName = m_temporal_dir + "/tempExtremas";
      tmpExtremasFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmpExtremasFile.precision(12);
      tmpExtremasFile << "iter,time";
      for (int n = 0; n < NVAR; ++n) {
        tmpExtremasFile << ",min_" << PeleLM::stateVariableName(n) << ",max_"
                        << PeleLM::stateVariableName(n);
      }
      tmpExtremasFile << "\n";
    }
    if (m_do_patch_mfr != 0) {
      tempFileName = m_temporal_dir + "/temppatchmfr";
      tmppatchmfrFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmppatchmfrFile.precision(12);
      tmppatchmfrFile << "iter,time";
      for (const auto& m_bPatche : m_bPatches) {
        BPatch* patch = m_bPatche.get();
        BPatch::BpatchDataContainer bphost = patch->getHostData();
        for (int i = 0; i < bphost.num_groups; ++i) {
          tmppatchmfrFile << ","
                          << patch->m_patchname + "_" + patch->groupNames[i];
        }
      }
      tmppatchmfrFile << "\n";
    }
#ifdef PELE_USE_PLASMA
    if (m_do_ionsBalance) {
      tempFileName = m_temporal_dir + "/tempIons";
      tmpIonsFile.open(
        tempFileName.c_str(),
        std::ios::out | std::ios::app | std::ios_base::binary);
      tmpIonsFile.precision(12);
      tmpIonsFile << "iter,time";
      for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        tmpIonsFile << ",curr_" << i << "_low,curr_" << i << "_hi";
      }
      tmpIonsFile << "\n";
    }
#endif
  }
}

void
PeleLM::closeTempFile()
{
  if (m_do_temporals == 0) {
    return;
  }

  if (amrex::ParallelDescriptor::IOProcessor()) {
    tmpStateFile.flush();
    tmpStateFile.close();
    if (m_do_massBalance != 0) {
      tmpMassFile.flush();
      tmpMassFile.close();
    }
    if (m_do_energyBalance != 0) {
      tmpEnergyFile.flush();
      tmpEnergyFile.close();
    }
    if (m_do_speciesBalance != 0) {
      tmpSpecFile.flush();
      tmpSpecFile.close();
    }
    if (m_do_extremas != 0) {
      tmpExtremasFile.flush();
      tmpExtremasFile.close();
    }
    if (m_do_patch_mfr != 0) {
      tmppatchmfrFile.flush();
      tmppatchmfrFile.close();
    }
#ifdef PELE_USE_PLASMA
    if (m_do_ionsBalance) {
      tmpIonsFile.flush();
      tmpIonsFile.close();
    }
#endif
  }
}
