#include "PeleLMeX_BPatch.H"
#include "PelePhysics.H"
#include <unordered_set>

std::string
Trim_First_Last_Whitespace(const std::string& species_name)
{
  auto start = species_name.find_first_not_of(" \t\n\r\f\v");
  auto end = species_name.find_last_not_of(" \t\n\r\f\v");

  if (start == std::string::npos) {
    return ""; // all whitespace
  }
  return species_name.substr(start, end - start + 1);
}

BPatch::BPatch(
  const std::string& patch_name,
  const amrex::Geometry& geom,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>* eosparms_h,
  const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>* /*
    eosparms_d*/)
  : m_patchname(std::move(patch_name))
{

  std::string ppbpatch = "bpatch." + m_patchname;
  amrex::ParmParse ps(ppbpatch);

  auto prob_lo = geom.ProbLoArray();
  auto prob_hi = geom.ProbHiArray();

  // Reading patch related info
  ps.get("patchtype", m_patchtype);
  ps.get("boundary_direction", m_bpdata_h.m_boundary_dir);
  ps.get("boundary_lo_or_hi", m_bpdata_h.m_boundary_lo_hi);

  // Checking if patch type matches defined ones
#if (AMREX_SPACEDIM == 1)
  if (m_patchtype != "full-boundary") {
    amrex::Abort(
      "\nPatch type for 1-D domains should be specified as full-boundary.\n");
  }
#elif (AMREX_SPACEDIM == 2)
  if (m_patchtype != "line" && m_patchtype != "full-boundary") {
    amrex::Abort(
      "\nPatch type for 2-D domains should be line or full-boundary\n");
  }
#else
  if (
    m_patchtype != "full-boundary" && m_patchtype != "circle" &&
    m_patchtype != "rectangle" && m_patchtype != "circle-annular" &&
    m_patchtype != "rectangle-annular") {
    amrex::Abort(
      "\nPatch type do not match allowed values "
      "(full-boundary,circle,rectangle,circle-annular,rectangle-annular) \n");
  }
#endif

  if (
    m_bpdata_h.m_boundary_dir != 0 && m_bpdata_h.m_boundary_dir != 1 &&
    m_bpdata_h.m_boundary_dir != 2) {
    amrex::Abort("\nBoundary direction should be 0,1 or 2");
  }

  if (m_bpdata_h.m_boundary_lo_hi != 0 && m_bpdata_h.m_boundary_lo_hi != 1) {
    amrex::Abort("\nBoundary high low should be 0 or 1");
  }

#if (AMREX_SPACEDIM == 2)
  if (m_bpdata_h.m_boundary_dir == 2) {
    amrex::Abort("\nFor 2D problems, patch boundary direction cannot be 2");
  }
#endif

  // Define patch_num
  if (m_patchtype == "full-boundary") {
    m_bpdata_h.m_patchtype_num = 0;
  } else if (m_patchtype == "line" || m_patchtype == "circle") {
    m_bpdata_h.m_patchtype_num = 1;
  } else if (m_patchtype == "rectangle") {
    m_bpdata_h.m_patchtype_num = 2;
  } else if (m_patchtype == "circle-annular") {
    m_bpdata_h.m_patchtype_num = 3;
  } else if (m_patchtype == "rectangle-annular") {
    m_bpdata_h.m_patchtype_num = 4;
  } else {
    amrex::Abort("\nError! Unknown patch type");
  }

  // Define patch variables
  if (m_patchtype == "line") {
    for (int n = 0; n < 2; ++n) {
      ps.get("patch_line_center", m_bpdata_h.m_patch_line_center[n], n);
    }
    m_bpdata_h.m_patch_line_center[m_bpdata_h.m_boundary_dir] = 0.0;
    ps.get("patch_line_radius", m_bpdata_h.m_patch_line_radius);
    if (m_bpdata_h.m_patch_line_radius <= 0.0) {
      amrex::Abort("\nPatch radius for line should be greater than 0");
    }
  } else if (m_patchtype == "full-boundary") {
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
      m_bpdata_h.m_patch_rectangle_lo[n] = prob_lo[n];
      m_bpdata_h.m_patch_rectangle_hi[n] = prob_hi[n];
    }
    m_bpdata_h.m_patch_rectangle_lo[m_bpdata_h.m_boundary_dir] = 0.0;
    m_bpdata_h.m_patch_rectangle_hi[m_bpdata_h.m_boundary_dir] = 0.0;
  } else if (m_patchtype == "circle") {
    for (int n = 0; n < 3; ++n) {
      ps.get("patch_circle_center", m_bpdata_h.m_patch_circle_center[n], n);
    }
    m_bpdata_h.m_patch_circle_center[m_bpdata_h.m_boundary_dir] = 0.0;
    ps.get("patch_circle_radius", m_bpdata_h.m_patch_circle_radius);
    if (m_bpdata_h.m_patch_circle_radius <= 0.0) {
      amrex::Abort("\nPatch radius for circle should be greater than 0");
    }
  } else if (m_patchtype == "rectangle") {
    for (int n = 0; n < 3; ++n) {
      ps.get("patch_rectangle_lo", m_bpdata_h.m_patch_rectangle_lo[n], n);
    }
    for (int n = 0; n < 3; ++n) {
      ps.get("patch_rectangle_hi", m_bpdata_h.m_patch_rectangle_hi[n], n);
    }
    m_bpdata_h.m_patch_rectangle_lo[m_bpdata_h.m_boundary_dir] = 0.0;
    m_bpdata_h.m_patch_rectangle_hi[m_bpdata_h.m_boundary_dir] = 0.0;
  } else if (m_patchtype == "circle-annular") {
    for (int n = 0; n < 3; ++n) {
      ps.get("patch_circ_ann_center", m_bpdata_h.m_patch_circ_ann_center[n], n);
    }
    m_bpdata_h.m_patch_circ_ann_center[m_bpdata_h.m_boundary_dir] = 0.0;
    ps.get(
      "patch_circ_ann_inner_radius", m_bpdata_h.m_patch_circ_ann_inner_radius);
    ps.get(
      "patch_circ_ann_outer_radius", m_bpdata_h.m_patch_circ_ann_outer_radius);
  } else if (m_patchtype == "rectangle-annular") {
    for (int n = 0; n < 3; ++n) {
      ps.get(
        "patch_rect_ann_outer_lo", m_bpdata_h.m_patch_rect_ann_outer_lo[n], n);
    }
    for (int n = 0; n < 3; ++n) {
      ps.get(
        "patch_rect_ann_outer_hi", m_bpdata_h.m_patch_rect_ann_outer_hi[n], n);
    }
    for (int n = 0; n < 3; ++n) {
      ps.get(
        "patch_rect_ann_inner_lo", m_bpdata_h.m_patch_rect_ann_inner_lo[n], n);
    }
    for (int n = 0; n < 3; ++n) {
      ps.get(
        "patch_rect_ann_inner_hi", m_bpdata_h.m_patch_rect_ann_inner_hi[n], n);
    }

    m_bpdata_h.m_patch_rect_ann_inner_hi[m_bpdata_h.m_boundary_dir] = 0.0;
    m_bpdata_h.m_patch_rect_ann_outer_hi[m_bpdata_h.m_boundary_dir] = 0.0;
    m_bpdata_h.m_patch_rect_ann_inner_lo[m_bpdata_h.m_boundary_dir] = 0.0;
    m_bpdata_h.m_patch_rect_ann_outer_lo[m_bpdata_h.m_boundary_dir] = 0.0;
  } else {
    amrex::Abort("\nError! Unknown patch type");
  }

  m_bpdata_h.num_species = ps.countval("species");
  ps.getarr("species", speciesList); // This is not exactly species list. This
                                     // should also include species groups

  // First find out the total number of species&groups specified in the input
  // file
  if (m_bpdata_h.num_species > 0) {
    speciesList.resize(m_bpdata_h.num_species);
  }

  // Remove duplicate entries
  std::sort(speciesList.begin(), speciesList.end());
  speciesList.erase(
    std::unique(speciesList.begin(), speciesList.end()), speciesList.end());

  amrex::Vector<std::string> tmp_species_only;
  amrex::Vector<std::string> names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    names, eosparms_h);

  // Check if there are unpaired open or closing braces in group definitions.
  for (const auto& s : speciesList) {
    std::string s_trimmed;
    s_trimmed = Trim_First_Last_Whitespace(s);
    if (
      (s_trimmed.front() == '{' && s_trimmed.back() != '}') ||
      (s_trimmed.front() != '{' && s_trimmed.back() == '}')) {
      std::string msg = "\nError! Unpaired open or closing braces in group "
                        "definition in the patch " +
                        m_patchname;
      amrex::Abort(msg);
    }
  }

  // To begin with, fill tmp_species_only with pure species in the list. For
  // now, we are not checking if the species actually exist in the mechanism
  for (const auto& s : speciesList) {
    std::string s_trimmed;
    s_trimmed = Trim_First_Last_Whitespace(s);
    if ((s_trimmed.front() != '{' && s_trimmed.back() != '}')) {

      tmp_species_only.push_back(s_trimmed);
    }
  }

  // Now search for species in the group definitions and fill them in
  // tmp_species_only. For now, we are not checking if the species actually
  // exist in the mechanism

  for (const auto& s : speciesList) {
    if (s.front() == '{' && s.back() == '}') {
      size_t start = s.find('{');
      size_t colon = s.find(':');
      size_t end = s.find('}');
      std::string groupname_tmp;
      if (
        start != std::string::npos && colon != std::string::npos &&
        colon > start) {
        groupname_tmp = s.substr(start + 1, colon - start - 1);
      }
      amrex::Vector<std::string> tokens;
      if (
        colon != std::string::npos && end != std::string::npos && end > colon) {
        // Extract substring between ':' and '}'
        std::string rest = s.substr(colon + 1, end - colon - 1);

        if (rest.empty()) {
          std::string msg =
            "\nError! Empty species list in the group definition " +
            groupname_tmp + "  in the patch " + m_patchname;
          amrex::Abort(msg);
        }

        // Split by comma
        std::stringstream ss(rest);
        std::string item;
        std::string item_trimmed;
        while (std::getline(ss, item, ',')) {
          item_trimmed = Trim_First_Last_Whitespace(item);
          tokens.push_back(item_trimmed);
          auto it = std::find(
            tmp_species_only.begin(), tmp_species_only.end(), item_trimmed);
          if (it != tmp_species_only.end()) {
            // Do nothing
          } else {
            tmp_species_only.push_back(item_trimmed);
          }
        }
      }

      std::unordered_set<std::string> check_duplicate;
      for (const auto& token : tokens) {
        if (!check_duplicate.insert(token)
               .second) { // insert returns {iterator, success}
          std::string msg = "\nError! Duplicate species " + s +
                            " in group definition " + groupname_tmp +
                            "  in the patch " + m_patchname;
          amrex::Abort(msg);
        }
      }

      // If tokens is null vector abort
      if (tokens.size() <= 0) {
        std::string msg =
          "\nError! Unable to find species in the group definition " +
          groupname_tmp + "  in the patch " + m_patchname;
        amrex::Abort(msg);
      }
    }
  }

  // Now we collected all the species for which we need to find flux. Now let us
  // check if all the species in tmp_species_only exist in the mechanism

  for (const auto& s : tmp_species_only) {
    auto it = std::find(names.begin(), names.end(), s);
    if (it == names.end()) {
      std::string msg = "\nError! Unable to find species " + s +
                        " in the mechanism. Please correct the bpatch species "
                        "list in the patch " +
                        m_patchname;
      amrex::Abort(msg);
    }
  }

  // Now the species list should be devoid of any duplicates or non-existent
  // species
  m_bpdata_h.num_species = static_cast<int>(tmp_species_only.size());

  // Now allocate memory for speciesIndex and speciesFlux
  if (m_bpdata_h.num_species > 0) {
    m_bpdata_h.speciesIndex = (int*)amrex::The_Pinned_Arena()->alloc(
      m_bpdata_h.num_species * sizeof(int));
    m_bpdata_h.speciesFlux = (amrex::Real*)amrex::The_Pinned_Arena()->alloc(
      m_bpdata_h.num_species * sizeof(amrex::Real));
    m_host_allocated = true;
  } else {
    amrex::Abort("\nError! No species provided to plot flux at boundary patch");
  }

  // Initialise with -1
  for (int n = 0; n < m_bpdata_h.num_species; n++) {
    m_bpdata_h.speciesIndex[n] = -1;
  }

  // Now fill speciesIndex
  for (int n = 0; n < tmp_species_only.size(); n++) {
    auto it = std::find(names.begin(), names.end(), tmp_species_only[n]);
    if (it != names.end()) {
      size_t index = std::distance(names.begin(), it);
      m_bpdata_h.speciesIndex[n] = static_cast<int>(index);
    }
  }

  // Check if there is -1. There shouldn't be any
  for (int n = 0; n < m_bpdata_h.num_species; n++) {
    if (m_bpdata_h.speciesIndex[n] == -1) {
      amrex::Abort("\nError! -1 found in specified index vector");
    }
  }

  for (const auto& s : speciesList) {
    amrex::Vector<int> tmp;
    if (s.front() != '{' && s.back() != '}') {
      auto it = std::find(names.begin(), names.end(), s);
      if (it != names.end()) {
        size_t index = std::distance(names.begin(), it);
        tmp.push_back(static_cast<int>(index));
      }
      speciesinGroup.push_back(tmp);
      groupNames.push_back(s);
    } else if (s.front() == '{' && s.back() == '}') // Found a group definition
    {
      size_t start = s.find('{'); // opening brace
      size_t colon = s.find(':'); // first colon
      size_t end = s.find('}');   // ending brace

      if (
        start != std::string::npos && colon != std::string::npos &&
        colon > start) {
        groupNames.push_back(s.substr(start + 1, colon - start - 1));
      }

      amrex::Vector<std::string> tokens;
      if (
        colon != std::string::npos && end != std::string::npos && end > colon) {
        // Extract substring between ':' and '}'
        std::string rest = s.substr(colon + 1, end - colon - 1);

        // Split by comma
        std::stringstream ss(rest);
        std::string item;
        std::string item_trimmed;
        while (std::getline(ss, item, ',')) {
          item_trimmed = Trim_First_Last_Whitespace(item);
          auto it = std::find(names.begin(), names.end(), item_trimmed);
          size_t index = std::distance(names.begin(), it);
          tmp.push_back(static_cast<int>(index));
        }
      }
      speciesinGroup.push_back(tmp);
    }
  }

  // If species are not found by now, they could be defined in groups. Check for
  // species group definitions

  m_bpdata_h.num_groups = static_cast<int>(groupNames.size());

  allocate();
  amrex::Gpu::streamSynchronize();
}
