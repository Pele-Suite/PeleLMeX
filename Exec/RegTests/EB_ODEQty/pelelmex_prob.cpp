#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("T_mean", prob_parm->T_mean);
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("meanFlowDir", prob_parm->meanFlowDir);
  pp.query("meanFlowMag", prob_parm->meanFlowMag);
  pp.query("ode_qty_test", prob_parm->ode_qty_test);
  pp.query("ode_IC", prob_parm->ode_IC);
  pp.query("ode_xy_lo", prob_parm->ode_xy_lo);
  pp.query("ode_length", prob_parm->ode_length);
  pp.query("ode_height", prob_parm->ode_height);
  pp.query("ode_srcstrength", prob_parm->ode_srcstrength);
  pp.query("composition_test", prob_parm->composition_test);
  pp.query("Y_CO2_0", prob_parm->Y_CO2_0);
  pp.query("Y_N2_0", prob_parm->Y_N2_0);
  pp.query("Y_AR_0", prob_parm->Y_AR_0);
  pp.query("extRhoYCO2", prob_parm->extRhoYCO2);
  pp.query("extRhoYCO2_ts", prob_parm->extRhoYCO2_ts);
  pp.query("extRhoYCO2_ramp", prob_parm->extRhoYCO2_ramp);
  pp.query("isothermal_test", prob_parm->isothermal_test);
  pp.query("iso_amp", prob_parm->iso_amp);
  pp.query("iso_sigma", prob_parm->iso_sigma);
}

void
PeleLM::freeProbParm()
{
}
