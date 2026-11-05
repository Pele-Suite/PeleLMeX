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
  pp.query("perturbMag", prob_parm->perturbMag);
  pp.query("channelFullHeight", prob_parm->channelFullHeight);
  pp.query("problem_type", prob_parm->flowType);
  AMREX_ALWAYS_ASSERT(
    prob_parm->flowType == 1 || prob_parm->flowType == 2 ||
    prob_parm->flowType == 3);
}

void
PeleLM::freeProbParm()
{
}
