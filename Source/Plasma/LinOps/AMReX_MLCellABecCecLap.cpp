
#include <AMReX_MLCellABecCecLap.H>
#include <AMReX_MLCellABecCecLap_K.H>
#include <AMReX_MLLinOp_K.H>

#ifdef AMREX_USE_PETSC
#include <petscksp.h>
#include <AMReX_PETSc.H>
#endif

namespace amrex {

MLCellABecCecLap::MLCellABecCecLap() {}

MLCellABecCecLap::~MLCellABecCecLap() {}

void
MLCellABecCecLap::define(
  const Vector<Geometry>& a_geom,
  const Vector<BoxArray>& a_grids,
  const Vector<DistributionMapping>& a_dmap,
  const LPInfo& a_info,
  const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
  MLCellLinOp::define(a_geom, a_grids, a_dmap, a_info, a_factory);

  m_overset_mask.resize(m_num_amr_levels);
  for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
    m_overset_mask[amrlev].resize(m_num_mg_levels[amrlev]);
  }
}

void
MLCellABecCecLap::define(
  const Vector<Geometry>& a_geom,
  const Vector<BoxArray>& a_grids,
  const Vector<DistributionMapping>& a_dmap,
  const Vector<iMultiFab const*>& a_overset_mask,
  const LPInfo& a_info,
  const Vector<FabFactory<FArrayBox> const*>& a_factory)
{
  BL_PROFILE("MLCellABecCecLap::define(overset)");

  AMREX_ALWAYS_ASSERT(!hasHiddenDimension());

  m_lpinfo_arg = a_info;

  int namrlevs = a_geom.size();
  m_overset_mask.resize(namrlevs);
  for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
    m_overset_mask[amrlev].push_back(
      std::make_unique<iMultiFab>(a_grids[amrlev], a_dmap[amrlev], 1, 1));
    iMultiFab::Copy(
      *m_overset_mask[amrlev][0], *a_overset_mask[amrlev], 0, 0, 1, 0);
    if (amrlev > 1) {
      AMREX_ALWAYS_ASSERT(
        amrex::refine(a_geom[amrlev - 1].Domain(), 2) ==
        a_geom[amrlev].Domain());
    }
  }

  int amrlev = 0;
  Box dom = a_geom[0].Domain();
  for (int mglev = 1; mglev <= a_info.max_coarsening_level; ++mglev) {
    AMREX_ALWAYS_ASSERT(mg_coarsen_ratio == 2);
    iMultiFab const& fine = *m_overset_mask[amrlev][mglev - 1];
    if (dom.coarsenable(2) && fine.boxArray().coarsenable(2)) {
      dom.coarsen(2);
      auto crse = std::make_unique<iMultiFab>(
        amrex::coarsen(fine.boxArray(), 2), fine.DistributionMap(), 1, 1);
      ReduceOps<ReduceOpSum> reduce_op;
      ReduceData<int> reduce_data(reduce_op);
      using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*crse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<int const> const& fmsk = fine.const_array(mfi);
        Array4<int> const& cmsk = crse->array(mfi);
        reduce_op.eval(
          bx, reduce_data,
          [cmsk, fmsk] AMREX_GPU_HOST_DEVICE(Box const& b) -> ReduceTuple {
            return {coarsen_overset_mask(b, cmsk, fmsk)};
          });
      }
      ReduceTuple hv = reduce_data.value();
      if (amrex::get<0>(hv) == 0) {
        m_overset_mask[amrlev].push_back(std::move(crse));
      } else {
        break;
      }
    } else {
      break;
    }
  }
  int max_overset_mask_coarsening_level = m_overset_mask[amrlev].size() - 1;
  ParallelAllReduce::Min(
    max_overset_mask_coarsening_level, ParallelContext::CommunicatorSub());
  m_overset_mask[amrlev].resize(max_overset_mask_coarsening_level + 1);

  LPInfo linfo = a_info;
  linfo.max_coarsening_level = amrex::min<int>(
    a_info.max_coarsening_level, max_overset_mask_coarsening_level);

  MLCellLinOp::define(a_geom, a_grids, a_dmap, linfo, a_factory);

  amrlev = 0;
  for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev) {
    MultiFab foo(
      m_grids[amrlev][mglev], m_dmap[amrlev][mglev], 1, 0,
      MFInfo().SetAlloc(false));
    if (!amrex::isMFIterSafe(*m_overset_mask[amrlev][mglev], foo)) {
      auto osm = std::make_unique<iMultiFab>(
        m_grids[amrlev][mglev], m_dmap[amrlev][mglev], 1, 1);
      osm->ParallelCopy(*m_overset_mask[amrlev][mglev]);
      std::swap(osm, m_overset_mask[amrlev][mglev]);
    }
  }

  for (amrlev = 1; amrlev < m_num_amr_levels; ++amrlev) {
    for (int mglev = 1; mglev < m_num_mg_levels[amrlev];
         ++mglev) { // for ref_ratio 4
      m_overset_mask[amrlev].push_back(
        std::make_unique<iMultiFab>(
          m_grids[amrlev][mglev], m_dmap[amrlev][mglev], 1, 1));
#ifdef AMREX_USE_GPU
      if (
        Gpu::inLaunchRegion() &&
        m_overset_mask[amrlev][mglev]->isFusingCandidate()) {
        auto const& crsema = m_overset_mask[amrlev][mglev]->arrays();
        auto const& finema = m_overset_mask[amrlev][mglev - 1]->const_arrays();
        ParallelFor(
          *m_overset_mask[amrlev][mglev],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            coarsen_overset_mask(i, j, k, crsema[box_no], finema[box_no]);
          });
        Gpu::streamSynchronize();
      } else
#endif
      {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*m_overset_mask[amrlev][mglev], TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
          const Box& bx = mfi.tilebox();
          Array4<int> const& cmsk = m_overset_mask[amrlev][mglev]->array(mfi);
          Array4<int const> const fmsk =
            m_overset_mask[amrlev][mglev - 1]->const_array(mfi);
          AMREX_HOST_DEVICE_PARALLEL_FOR_3D(
            bx, i, j, k, { coarsen_overset_mask(i, j, k, cmsk, fmsk); });
        }
      }
    }
  }

  for (amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
    for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
      m_overset_mask[amrlev][mglev]->setBndry(1);
      m_overset_mask[amrlev][mglev]->FillBoundary(
        m_geom[amrlev][mglev].periodicity());
    }
  }
}

void
MLCellABecCecLap::prepareForSolve()
{
  MLCellLinOp::prepareForSolve();
}

void
MLCellABecCecLap::update()
{
  if (MLCellLinOp::needsUpdate())
    MLCellLinOp::update();
}

void
MLCellABecCecLap::getFluxes(
  const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& a_flux,
  const Vector<MultiFab*>& a_sol,
  Location a_loc) const
{
  BL_PROFILE("MLMG::getFluxes()");

  // TODO
  amrex::Abort("MLMG: getFluxes not implemented for MLCellABecCecLap");
  const Real betainv = 1.0 / getBScalar();
  const int nlevs = NAMRLevels();
  for (int alev = 0; alev < nlevs; ++alev) {
    compFlux(alev, a_flux[alev], *a_sol[alev], a_loc);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      unapplyMetricTerm(alev, 0, *a_flux[alev][idim]);
      if (betainv != 1.0) {
        a_flux[alev][idim]->mult(betainv);
      }
    }
  }
}

void
MLCellABecCecLap::applyInhomogNeumannTerm(
  int /*amrlev*/, MultiFab& /*rhs*/) const
{
  bool has_inhomog_neumann = hasInhomogNeumannBC();
  bool has_robin = hasRobinBC();

  if (!has_inhomog_neumann && !has_robin)
    return;

  // TODO
  amrex::Abort(
    "MLMG: InhomogNeumann or Robin not implemented for MLCellABecCecLap");
}

void
MLCellABecCecLap::applyOverset(int amrlev, MultiFab& rhs) const
{
  if (m_overset_mask[amrlev][0]) {
    const int ncomp = getNComp();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*m_overset_mask[amrlev][0], TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      Array4<Real> const& rfab = rhs.array(mfi);
      Array4<int const> const& osm =
        m_overset_mask[amrlev][0]->const_array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
        if (osm(i, j, k) == 0)
          rfab(i, j, k, n) = 0.0;
      });
    }
  }
}

#ifdef AMREX_USE_HYPRE
std::unique_ptr<Hypre>
MLCellABecCecLap::makeHypre(Hypre::Interface hypre_interface) const
{
  const BoxArray& ba = m_grids[0].back();
  const DistributionMapping& dm = m_dmap[0].back();
  const Geometry& geom = m_geom[0].back();
  const auto& factory = *(m_factory[0].back());
  MPI_Comm comm = BottomCommunicator();

  auto hypre_solver = amrex::makeHypre(ba, dm, geom, comm, hypre_interface);

  hypre_solver->setScalars(getAScalar(), getBScalar());

  const int mglev = NMGLevels(0) - 1;
  auto ac = getACoeffs(0, mglev);
  if (ac) {
    hypre_solver->setACoeffs(*ac);
  } else {
    MultiFab alpha(ba, dm, 1, 0, MFInfo(), factory);
    alpha.setVal(0.0);
    hypre_solver->setACoeffs(alpha);
  }

  auto bc = getBCoeffs(0, mglev);
  if (bc[0]) {
    hypre_solver->setBCoeffs(bc);
  } else {
    Array<MultiFab, AMREX_SPACEDIM> beta;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      beta[idim].define(
        amrex::convert(ba, IntVect::TheDimensionVector(idim)), dm, 1, 0,
        MFInfo(), factory);
      beta[idim].setVal(1.0);
    }
    hypre_solver->setBCoeffs(amrex::GetArrOfConstPtrs(beta));
  }

  // TODO
  amrex::Abort("MLMG: makeHypre not implemented for MLCellABecCecLap");

  return hypre_solver;
}
#endif

#ifdef AMREX_USE_PETSC
std::unique_ptr<PETScABecLap>
MLCellABecCecLap::makePETSc() const
{
  const BoxArray& ba = m_grids[0].back();
  const DistributionMapping& dm = m_dmap[0].back();
  const Geometry& geom = m_geom[0].back();
  const auto& factory = *(m_factory[0].back());
  MPI_Comm comm = BottomCommunicator();

  auto petsc_solver = makePetsc(ba, dm, geom, comm);

  petsc_solver->setScalars(getAScalar(), getBScalar());

  const int mglev = NMGLevels(0) - 1;
  auto ac = getACoeffs(0, mglev);
  if (ac) {
    petsc_solver->setACoeffs(*ac);
  } else {
    MultiFab alpha(ba, dm, 1, 0, MFInfo(), factory);
    alpha.setVal(0.0);
    petsc_solver->setACoeffs(alpha);
  }

  auto bc = getBCoeffs(0, mglev);
  if (bc[0]) {
    petsc_solver->setBCoeffs(bc);
  } else {
    Array<MultiFab, AMREX_SPACEDIM> beta;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      beta[idim].define(
        amrex::convert(ba, IntVect::TheDimensionVector(idim)), dm, 1, 0,
        MFInfo(), factory);
      beta[idim].setVal(1.0);
    }
    petsc_solver->setBCoeffs(amrex::GetArrOfConstPtrs(beta));
  }

  // TODO
  amrex::Abort("MLMG: makePETSc not implemented for MLCellABecCecLap");

  return petsc_solver;
}
#endif

} // namespace amrex
