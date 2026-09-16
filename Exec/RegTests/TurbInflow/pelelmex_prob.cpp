#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("T_mean", prob_parm->T_mean);
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("flowDir", prob_parm->meanFlowDir);
  pp.query("flowMag", prob_parm->meanFlowMag);
  pp.query("bothSides", prob_parm->bothSides);

  // Mesh mapping: the ConstantMap factors turn Xi extents into physical ones
  // in bcnormal (defaults to 1 for the other maps and without mapping).
  {
    amrex::ParmParse ppcm("ConstantMap");
    amrex::Vector<amrex::Real> fac(AMREX_SPACEDIM, 1.0);
    ppcm.queryarr("scaling_factor", fac, 0, AMREX_SPACEDIM);
    prob_parm->fac_x = fac[0];
#if AMREX_SPACEDIM >= 2
    prob_parm->fac_y = fac[1];
#endif
#if AMREX_SPACEDIM == 3
    prob_parm->fac_z = fac[2];
#endif
  }

  /*
  if (!m_incompressible) {
     auto& trans_parm = PeleLM::trans_parms.host_parm();
     amrex::ParmParse pptr("transport");
     pp.query("const_viscosity", trans_parm.const_viscosity);
     pp.query("const_bulk_viscosity", trans_parm.const_bulk_viscosity);
     pp.query("const_conductivity", trans_parm.const_conductivity);
     pp.query("const_diffusivity", trans_parm.const_diffusivity);
     PeleLM::trans_parms.sync_to_device();
  }
  */
}

void
PeleLM::freeProbParm()
{
}
