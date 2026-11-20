#include <PeleLMeX.H>

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  std::string type;
  pp.get("P_mean", PeleLM::prob_parm->P_mean);

  // Crossflow Conditions
  pp.get("cf_velocity", PeleLM::prob_parm->cf_velocity);
  pp.get("cf_temp", PeleLM::prob_parm->cf_temp);
  {
    amrex::Vector<std::string> compositionIn;
    std::string comp_type = "mass";
    int entryCount = pp.countval("cf_composition");
    compositionIn.resize(entryCount);
    pp.getarr("cf_composition", compositionIn, 0, entryCount);
    pp.query("cf_composition_type", comp_type);
    parseComposition(compositionIn, comp_type, PeleLM::prob_parm->cf_massfracs);
  }
  pp.query("cf_dir", PeleLM::prob_parm->cf_dir);

  // Jet Conditions
  pp.get("jet_velocity", PeleLM::prob_parm->jet_velocity);
  pp.get("jet_temp", PeleLM::prob_parm->jet_temp);
  {
    amrex::Vector<std::string> compositionIn;
    std::string comp_type = "mass";
    int entryCount = pp.countval("jet_composition");
    compositionIn.resize(entryCount);
    pp.getarr("jet_composition", compositionIn, 0, entryCount);
    pp.query("jet_composition_type", comp_type);
    parseComposition(
      compositionIn, comp_type, PeleLM::prob_parm->jet_massfracs);
  }
  pp.query("jet_dir", PeleLM::prob_parm->jet_dir);
  pp.get("jet_rad", PeleLM::prob_parm->jet_rad);
  pp.get("jet_bl_thickness", PeleLM::prob_parm->jet_bl_thickness);
  pp.query("jet_init_time", PeleLM::prob_parm->jet_init_time);
  pp.query("jet_start_time", PeleLM::prob_parm->jet_start_time);
  pp.query("double_jet", PeleLM::prob_parm->double_jet);
}

void
PeleLM::freeProbParm()
{
}
