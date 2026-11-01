#ifndef DIFFUSION_OP_H
#define DIFFUSION_OP_H

#include <AMReX_MLMG.H>
#ifdef AMREX_USE_HYPRE
#include <AMReX_HypreMLABecLap.H>
#endif
#include <AMReX_BC_TYPES.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBTensorOp.H>
#define COVERED_VAL 1.0e40
#endif
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLTensorOp.H>

class PeleLM;

class DiffusionOp
{
public:
  // Methods

  DiffusionOp(PeleLM* a_pelelm, int ncomp = 1);

  void diffuse_scalar(
    amrex::Vector<amrex::MultiFab*> const& a_phi,
    const int phi_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_rhs,
    const int rhs_comp,
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_fluxes,
    const int flux_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_acoeff,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
    const int bcoeff_comp,
    amrex::Vector<amrex::BCRec> a_bcrec,
    const int ncomp,
    const int isPoissonSolve,
    const amrex::Real dt,
    amrex::Vector<amrex::MultiFab const*> const& a_boundary);

#ifdef AMREX_USE_EB
  void diffuse_scalar(
    amrex::Vector<amrex::MultiFab*> const& a_phi,
    const int phi_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_phiEB,
    const int phiEB_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_rhs,
    const int rhs_comp,
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_fluxes,
    const int flux_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_acoeff,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
    const int bcoeff_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_bcoeffEB,
    const int bcoeffEB_comp,
    amrex::Vector<amrex::BCRec> a_bcrec,
    const int ncomp,
    const int isPoissonSolve,
    const amrex::Real dt,
    amrex::Vector<amrex::MultiFab const*> const& a_boundary);
#endif

  void computeDiffLap(
    amrex::Vector<amrex::MultiFab*> const& laps,
    const int lap_comp,
    amrex::Vector<amrex::MultiFab const*> const& phi,
    const int phi_comp,
    amrex::Vector<amrex::MultiFab const*> const& bcoeff,
    const int bcoeff_comp,
    amrex::Vector<amrex::BCRec> a_bcrec,
    const int ncomp);

  void computeDiffFluxes(
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_fluxes,
    const int flux_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_phi,
    const int phi_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
    const int bcoeff_comp,
    amrex::Vector<amrex::BCRec> a_bcrec,
    const int ncomp,
    const int do_avgDown,
    amrex::Vector<amrex::MultiFab const*> const& a_boundary);

#ifdef AMREX_USE_EB
  void computeDiffFluxes(
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_fluxes,
    const int flux_comp,
    amrex::Vector<amrex::MultiFab*> const& a_EBflux,
    const int ebflux_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_phi,
    const int phi_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
    const int bcoeff_comp,
    amrex::Vector<amrex::MultiFab const*> const& a_EBvalue,
    amrex::Vector<amrex::MultiFab const*> const& a_EBbcoeff,
    amrex::Vector<amrex::BCRec> a_bcrec,
    const int ncomp,
    const int do_avgDown,
    amrex::Vector<amrex::MultiFab const*> const& a_boundary);
#endif

  void computeGradient(
    const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_grad,
    const amrex::Vector<amrex::MultiFab*>& a_laps,
    const amrex::Vector<amrex::MultiFab const*>& a_phi,
    const amrex::Vector<amrex::MultiFab const*>& a_boundary,
    const amrex::BCRec& a_bcrec,
    const int do_avgDown,
    int comp = 0) const;

  void avgDownFluxes(
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_fluxes,
    const int flux_comp,
    const int ncomp) const;

  void readParameters();
  // hypre, PETSc or AlgMG solves (bottom, level 0 or composite)
  [[nodiscard]] bool useMatrixSolver() const;

  // Data and parameters
  PeleLM* m_pelelm;

#ifdef AMREX_USE_EB
  // Scalar diffusion apply operator
  std::unique_ptr<amrex::MLEBABecLap> m_scal_apply_op;

  // Scalar diffusion solve operator
  std::unique_ptr<amrex::MLEBABecLap> m_scal_solve_op;

  // Poisson-like operator for gradients
  std::unique_ptr<amrex::MLEBABecLap> m_gradient_op;
#else
  // Scalar diffusion apply operator
  std::unique_ptr<amrex::MLABecLaplacian> m_scal_apply_op;

  // Scalar diffusion solve operator
  std::unique_ptr<amrex::MLABecLaplacian> m_scal_solve_op;

  // Poisson-like operator for gradients
  std::unique_ptr<amrex::MLABecLaplacian> m_gradient_op;
#endif

  int m_ncomp = 1;

  // Components per linear solve: m_ncomp, or 1 for the matrix-based solvers
  // (see useMatrixSolver), which handle single-component systems only
  int m_ncomp_solve = 1;

  // Options to control MLMG behavior
  int m_mg_verbose = 0;
  int m_mg_bottom_verbose = 0;
  int m_mg_max_iter = 100;
  int m_mg_bottom_maxiter = 100;
  int m_mg_max_fmg_iter = 0;
  int m_mg_max_coarsening_level = 100;
  int m_mg_maxorder = 4;

  int m_num_pre_smooth = 2;
  int m_num_post_smooth = 2;

  amrex::Real m_mg_rtol = 1.0e-11;
  amrex::Real m_mg_atol = 1.0e-14;

  // Empty: keep the operator's default bottom solver
  std::string m_mg_bottom_solver;
  amrex::MultigridType m_mg_type = amrex::MultigridType::geometric;
  amrex::Real m_mg_bottom_rtol = -1.0;
  std::string m_hypre_namespace = "diffusion_hypre";

  // Solve with GMRES preconditioned by MLMG instead of MLMG alone
  bool m_use_gmres = false;
  int m_gmres_precond_iters = 1;

  // Composite (all AMR levels) solve with hypre (HypreMLABecLap), one
  // component at a time; options under m_mlhypre_prefix
  bool m_use_mlhypre = false;
  std::string m_mlhypre_prefix = "diffusion_mlhypre";
  // Reuse of the hypre solver setup. 0: no reuse, 1: within a solve call, 2:
  // across calls (AMReX redoes the setup when a solve with the reused setup
  // does not converge)
  int m_mlhypre_reuse_setup = 1;
#ifdef AMREX_USE_HYPRE
  // Kept across calls; rebuilt when the grids change
  std::unique_ptr<amrex::HypreMLABecLap> m_hmlabeclap;
  amrex::Vector<amrex::BoxArray> m_hmlabeclap_grids;
  amrex::Vector<amrex::DistributionMapping> m_hmlabeclap_dmaps;
#endif

  // MLMG failure debugging options
  int m_mg_fail_sdc_miniter = -1;
  int m_mg_fail_deltaT_miniter = -1;
  int m_mg_fail_species_maxiter_after_sdc_miniter = -1;
  int m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter = -1;
};

class DiffusionTensorOp
{
public:
  DiffusionTensorOp(PeleLM* a_pelelm);

  void compute_divtau(
    amrex::Vector<amrex::MultiFab*> const& a_divtau,
    amrex::Vector<amrex::MultiFab const*> const& a_vel,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_beta,
    const amrex::BCRec& a_bcrec,
    const amrex::Real scale);

  void diffuse_velocity(
    amrex::Vector<amrex::MultiFab*> const& a_vel,
    amrex::Vector<amrex::MultiFab const*> const& a_density,
    amrex::Vector<amrex::MultiFab const*> const& a_beta,
    const amrex::BCRec& a_bcrec,
    const amrex::Real dt);

  void computeGradientTensor(
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
      a_velgrad,
    amrex::Vector<amrex::MultiFab const*> const& a_vel);

private:
  void readParameters();

  // Data and parameters
  PeleLM* m_pelelm;

#ifdef AMREX_USE_EB
  // Apply operator
  std::unique_ptr<amrex::MLEBTensorOp> m_apply_op;
  std::unique_ptr<amrex::MLEBTensorOp> m_solve_op;
#else
  // Apply operator
  std::unique_ptr<amrex::MLTensorOp> m_apply_op;
  std::unique_ptr<amrex::MLTensorOp> m_solve_op;
#endif

  // Options to control MLMG behavior
  int m_mg_verbose = 0;
  int m_mg_bottom_verbose = 0;
  int m_mg_max_iter = 100;
  int m_mg_bottom_maxiter = 100;
  int m_mg_max_fmg_iter = 0;
  int m_mg_max_coarsening_level = 100;
  int m_mg_maxorder = 2;

  int m_num_pre_smooth = 2;
  int m_num_post_smooth = 2;

  amrex::Real m_mg_rtol = 1.0e-11;
  amrex::Real m_mg_atol = 1.0e-14;

  // Empty: keep the operator's default bottom solver
  std::string m_mg_bottom_solver;

  // Solve with GMRES preconditioned by MLMG instead of MLMG alone
  bool m_use_gmres = false;
  int m_gmres_precond_iters = 1;
  int m_gmres_restart = -1; // <= 0: GMRES default

  // MLMG failure debugging options
  int m_mg_fail_sdc_miniter = -1;
  int m_mg_fail_maxiter_after_sdc_miniter = -1;
};

#endif
