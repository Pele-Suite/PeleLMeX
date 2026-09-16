#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);
  pp.query("T_bubble", prob_parm->T_bubble);
  pp.query("bubble_radius", prob_parm->bubble_rad);
  pp.query("bubble_y0", prob_parm->bubble_y0);
  pp.query("use_symmetry", prob_parm->is_sym);
  pp.query("use_mix_bubble", prob_parm->bubble_is_mix);

  // If mesh mapping is active, pick up the ConstantMap scaling factors
  // so the IC can be evaluated in physical coordinates.  Falls back to
  // fac = 1 when mapping is off.
  {
    amrex::ParmParse ppcm("ConstantMap");
    amrex::Vector<amrex::Real> fac(AMREX_SPACEDIM, 1.0);
    ppcm.queryarr("scaling_factor", fac, 0, AMREX_SPACEDIM);
    prob_parm->fac_x = fac[0];
#if AMREX_SPACEDIM >= 2
    prob_parm->fac_y = fac[1];
#endif
#if AMREX_SPACEDIM >= 3
    prob_parm->fac_z = fac[2];
#endif
  }

#ifdef USE_CONSTANT_TRANSPORT
  auto& trans_parm = PeleLM::trans_parms.host_parm();
  amrex::ParmParse pptr("transport");
  pp.query("const_viscosity", trans_parm.const_viscosity);
  pp.query("const_bulk_viscosity", trans_parm.const_bulk_viscosity);
  pp.query("const_conductivity", trans_parm.const_conductivity);
  pp.query("const_diffusivity", trans_parm.const_diffusivity);
  PeleLM::trans_parms.sync_to_device();
#endif
}

void
PeleLM::freeProbParm()
{
}
