#include <limits>
#include <string>
#include <AMReX_Algorithm.H>
#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>
#include <PeleLMeX.H>
#include <PeleLMeX_DeriveFunc.H>
#include <PeleLMeX_BPatch.H>
#include <PelePhysics.H>
#ifdef PELE_USE_CMAKE
#include <PeleGitHashes.H>
#endif

#ifdef PELE_USE_PLASMA
#include "PeleLMeX_EOS_Extension.H"
#endif

#ifdef AMREX_USE_GPU
#include <AMReX_SUNMemory.H>
#endif

#ifdef PELE_USE_SOOT
#include "SootModel.H"
#endif

static amrex::Box
the_same_box(const amrex::Box& b)
{
  return b;
}
#ifdef PELE_USE_PLASMA
static amrex::Box
grow_box_by_one(const amrex::Box& b)
{
  return amrex::grow(b, 1);
}
#endif
static amrex::Box
grow_box_by_two(const amrex::Box& b)
{
  return amrex::grow(b, 2);
}

void
PeleLM::Setup()
{
  BL_PROFILE("PeleLMeX::Setup()");

  m_wall_start = amrex::ParallelDescriptor::second();

  // Ensure grid is isotropic.
  //
  // When mesh mapping is enabled, the AMReX grid represents the uniform
  // (Xi) computational grid and *must* be anisotropic in order for the
  // physical grid (dx_AMReX . fac_i per direction) to be isotropic.
  // We therefore skip this assertion when geometry.mesh_mapping is set
  // in the inputs; the mapping model is expected to produce a
  // physically isotropic grid (or the user has intentionally chosen
  // anisotropic physical spacing).
  {
    std::string mesh_mapping_name;
    amrex::ParmParse ppg("geometry");
    const bool mesh_mapping_on =
      static_cast<bool>(ppg.query("mesh_mapping", mesh_mapping_name));
    if (!mesh_mapping_on) {
      auto const dx = geom[0].CellSizeArray();
      AMREX_ALWAYS_ASSERT(AMREX_D_TERM(
        , amrex::almostEqual(dx[0], dx[1], 10),
        &&amrex::almostEqual(dx[1], dx[2], 10)));
    }
  }
  // Print build info to screen
#ifdef PELE_USE_CMAKE
  const std::string pele_hash = PeleBuildInfo::PeleLMeX_git_hash;
  const std::string amrex_hash = PeleBuildInfo::AMReX_git_hash;
  const std::string pelephysics_hash = PeleBuildInfo::PelePhysics_git_hash;
  const std::string amrex_hydro_hash = PeleBuildInfo::AMReXHydro_git_hash;
  const std::string sundials_hash = PeleBuildInfo::SUNDIALS_git_hash;
#else
  const std::string pele_hash = amrex::buildInfoGetGitHash(1);
  const std::string amrex_hash = amrex::buildInfoGetGitHash(2);
  const std::string pelephysics_hash = amrex::buildInfoGetGitHash(3);
  const std::string amrex_hydro_hash = amrex::buildInfoGetGitHash(4);
  const std::string sundials_hash = amrex::buildInfoGetGitHash(5);
#endif

  amrex::Print() << "\n ================ Version Info =================\n";
  amrex::Print() << " PeleLMeX    git hash: " << pele_hash << "\n";
  amrex::Print() << " AMReX       git hash: " << amrex_hash << "\n";
  amrex::Print() << " PelePhysics git hash: " << pelephysics_hash << "\n";
  amrex::Print() << " AMReX-Hydro git hash: " << amrex_hydro_hash << "\n";
  amrex::Print() << " SUNDIALS    git hash: " << sundials_hash << "\n";
  amrex::Print() << " ===============================================\n";

#ifdef PELE_USE_SOOT
  soot_model = new SootModel{};
#endif

  // Read PeleLMeX parameters
  readParameters();

#ifdef AMREX_USE_EB
  makeEBGeometry();
#endif

  // Initialize EOS
  if (m_incompressible == 0) {
    amrex::Print() << " Initialization of Eos ... \n";
    eos_parms.initialize();
    // TODO: this is a bit of a hack so the host eos_parm has access to
    // the host blackboxfunction data (manfunc_data)
#ifdef USE_MANIFOLD_EOS
    eos_parms.host_parm().manf_data =
      &(eos_parms.host_only_parm().manfunc_par->host_parm());
#endif
  }

  // Setup the state variables
  variablesSetup();

  // Initialize Transport and others
  if (m_incompressible == 0) {
    amrex::Print() << " Initialization of Transport ... \n";
#ifdef USE_MANIFOLD_TRANSPORT
    trans_parms.host_only_parm().manfunc_par =
      eos_parms.host_only_parm().manfunc_par;
#endif
    trans_parms.initialize();
    if ((m_les_verbose != 0) and m_do_les) { // Say what transport model we're
                                             // going to use

      amrex::Print() << "    Using LES in transport with Sc = "
                     << 1.0 / m_Schmidt_inv;
      if (pele::physics::PhysicsType::eos_type::identifier() == "Manifold") {
        amrex::Print() << ", enthalpy not diffused for Manifold EOS \n";
      } else {
        amrex::Print() << " and Pr = " << 1.0 / m_Prandtl_inv << "\n";
      }
    } else if (m_verbose != 0) {
      if (m_fixed_Le == 0 && m_fixed_Pr == 0) {
        if (m_use_soret == 0) {
          amrex::Print() << "    Using mixture-averaged transport \n";
        } else {
          amrex::Print()
            << "    Using mixture-averaged transport with Soret effects \n";
          if (m_soret_boundary_override != 0) {
            amrex::Print() << "    Imposing inhomogeneous Neumann conditions "
                              "for species on isothermal walls. WARNING: "
                              "use_wbar disabled. \n";
          }
        }
      } else {
        if (m_fixed_Le != 0) {
          amrex::Print() << "    Using fixed Le = " << 1.0 / m_Lewis_inv
                         << "\n";
        }
        if (m_fixed_Pr != 0) {
          amrex::Print() << "    Using fixed Pr = " << 1.0 / m_Prandtl_inv
                         << "\n";
        }
      }
    }
    if (m_do_react != 0) {
      int reactor_type = 2;
      int ncells_chem = 1;
      amrex::Print() << " Initialization of chemical reactor ... \n";
      m_chem_integrator = "ReactorNull";
      amrex::ParmParse pp("peleLM");
      pp.query("chem_integrator", m_chem_integrator);
      m_reactor =
        pele::physics::reactions::ReactorBase::create(m_chem_integrator);
      m_reactor->init(reactor_type, ncells_chem);
      m_reactor->set_eos_parm(
        &(eos_parms.host_parm()), eos_parms.device_parm());
      // For ReactorNull, we need to also skip instantaneous RR used in divU
      if (m_chem_integrator == "ReactorNull") {
        m_skipInstantRR = 1;
        m_plotChemDiag = 0;
        m_plotHeatRelease = 0;
        m_useTypValChem = 0;
        reactComponents.clear();
      }
      pp.query("plot_react", m_plot_react);
    }

#ifdef PELE_USE_PLASMA
    pele::physics::eos::charge_mass(zk.arr);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      zk[n] *= 1000.0; // CGS->MKS
    }
#endif
  }

  // Derived variables
  derivedSetup();

  // Emit mesh-mapping displacement metadata alongside the plotfile
  // when mesh mapping is active (so ParaView/VisIt renders in physical
  // space).  On by default; set to 0 to keep Xi-space plotfiles.
  // Parsed unconditionally so the query also runs for incompressible /
  // chemistry-off cases.
  {
    amrex::ParmParse ppplt("peleLM");
    ppplt.query("plot_mesh_mapping", m_plot_mesh_mapping);
  }

  // Evaluate variables
  evaluateSetup();

  // Tagging setup
  taggingSetup();

#ifdef PELE_USE_SPRAY
  SpraySetup();
#endif
#ifdef PELE_USE_SOOT
  if (do_soot_solve) {
    soot_model->define();
  }
#endif
  // Diagnostics setup
  createDiagnostics();

  // Boundary Patch Setup
  if (m_do_patch_mfr != 0) {
    initBPatches(Geom(0), &eos_parms.host_parm(), eos_parms.device_parm());
  }

  // Initialize Level Hierarchy data
  resizeArray();

  // Mixture fraction & Progress variable
  initMixtureFraction();
  initProgressVariable();

  // Initialize turbulence injection
  turb_inflow.init(Geom(0));
  if (m_do_turbulent_forcing) {
    turb_forcing.init(Geom(0).data());
  }

  // Initialize BCs
  setBoundaryConditions();

  // Problem parameters
  prob_parm = new ProbParm{};
  prob_parm_d = (ProbParm*)amrex::The_Arena()->alloc(sizeof(ProbParm));

  // Problem parameters
  readProbParm();

  // Initialize ambient pressure
  // Will be overwritten on restart.
  m_pOld = prob_parm->P_mean;
  m_pNew = prob_parm->P_mean;

  // Copy problem parameters into device copy
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, prob_parm, prob_parm + 1, prob_parm_d);

  // Initialize active control
  initActiveControl();

  // Check setup parameters
  checkSetupParams();
}

void
PeleLM::readParameters()
{
  BL_PROFILE("PeleLMeX::readParameters()");

  readIOParameters();

  amrex::ParmParse pp("peleLM");

  // -----------------------------------------
  // Misc
  // -----------------------------------------
  pp.query("run_mode", m_run_mode);
  pp.query("v", m_verbose);
  pp.query("mlmg_fail_plt_residuals", m_mlmg_fail_plt_residuals);

  // -----------------------------------------
  // Mesh mapping (optional)
  // -----------------------------------------
  // Activate via e.g.  geometry.mesh_mapping
  {
    amrex::ParmParse ppg("geometry");
    std::string mesh_mapping_name;
    if (ppg.query("mesh_mapping", mesh_mapping_name) != 0) {

      // --- Compatibility guards -----------------------------------------
      // These match amr-wind PR #545's stated limitations.  Relaxing any
      // of these requires additional implementation work (see notes).
#ifdef AMREX_USE_EB
      amrex::Abort(
        "mesh_mapping is not supported with Embedded Boundaries (EB).\n"
        "Rebuild without USE_EB or omit geometry.mesh_mapping.");
#endif
#ifdef PELE_USE_PLASMA
      amrex::Abort(
        "mesh_mapping is not supported with the plasma solver "
        "(PELE_USE_PLASMA).");
#endif
      if (geom[0].IsRZ()) {
        amrex::Abort(
          "mesh_mapping is not supported with RZ geometry "
          "(geometry.coord_sys = 1).");
      }
      if (max_level > 0) {
        amrex::Print()
          << " WARNING: mesh_mapping has only been exercised on "
             "single-level AMR;\n          multi-level behaviour with "
             "max_level > 0 is not verified.\n";
      }

      m_mesh_map = MeshMap::create(mesh_mapping_name);
      m_mesh_mapping = true;
      // Populate the coordinate evaluator now, rather than waiting for the
      // Init/Regrid paths: parameter validation further down this function
      // needs to convert between physical and Xi positions.  Init and
      // Regrid refresh it from the same source.
      m_map_eval = m_mesh_map->make_evaluator();
      amrex::Print() << " Mesh mapping enabled: " << mesh_mapping_name << "\n";
    }
  }

  // -----------------------------------------
  // Boundary conditions
  // -----------------------------------------
  int isOpenDomain = 0;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    int lo_bc = BoundaryCondition::BCInterior;
    int hi_bc = BoundaryCondition::BCInterior;
    parseUserKey(pp, "lo_bc", boundarycondition, lo_bc, idim);
    parseUserKey(pp, "hi_bc", boundarycondition, hi_bc, idim);
    m_phys_bc.setLo(idim, lo_bc);
    m_phys_bc.setHi(idim, hi_bc);
    if (
      lo_bc == BoundaryCondition::BCOutflow ||
      hi_bc == BoundaryCondition::BCOutflow) {
      isOpenDomain = 1;
    }
  }

  // Activate closed chamber if !isOpenDomain
  // enable overwriting
  m_closed_chamber = (isOpenDomain) != 0 ? 0 : 1;
  pp.query("closed_chamber", m_closed_chamber);
  if ((verbose != 0) && (m_closed_chamber != 0)) {
    amrex::Print()
      << " Simulation performed with the closed chamber algorithm \n";
  }

  // -----------------------------------------
  // Recycling-plane inflow
  // -----------------------------------------
  pp.query("use_inlet_from_plane", m_use_inlet_from_plane);
  pp.query("inlet_plane_dir", m_inlet_plane_dir);
  pp.query("inlet_plane_position", m_inlet_plane_position);
  pp.query("inlet_plane_avg_window", m_inlet_plane_avg_window);
  pp.query("inlet_plane_warmup_steps", m_inlet_plane_warmup_steps);
  if (m_use_inlet_from_plane != 0) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_inlet_plane_dir >= 0 && m_inlet_plane_dir < AMREX_SPACEDIM,
      "peleLM.inlet_plane_dir must be set to a valid direction "
      "when peleLM.use_inlet_from_plane is enabled");
    // inlet_plane_position is a physical coordinate.  Under mesh mapping the
    // physical domain bounds are the images of the Xi bounds, which for
    // ConstantMap differ from them; compare against those.
    {
      const auto gd = geom[0].data();
      const amrex::Real phys_lo = m_map_eval.x_phys_from_xi(
        m_inlet_plane_dir, geom[0].ProbLo()[m_inlet_plane_dir], gd);
      const amrex::Real phys_hi = m_map_eval.x_phys_from_xi(
        m_inlet_plane_dir, geom[0].ProbHi()[m_inlet_plane_dir], gd);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_inlet_plane_position >= phys_lo && m_inlet_plane_position <= phys_hi,
        "peleLM.inlet_plane_position must lie within the physical problem "
        "domain bounds in peleLM.inlet_plane_dir when "
        "peleLM.use_inlet_from_plane is enabled");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_inlet_plane_warmup_steps >= 0,
      "peleLM.inlet_plane_warmup_steps must be non-negative when "
      "peleLM.use_inlet_from_plane is enabled");

    std::string recycling_mode = "fluctuations";
    pp.query("recycling_mode", recycling_mode);
    if (recycling_mode == "fluctuations") {
      m_recycling_mode = RecyclingMode::Fluctuations;
    } else if (recycling_mode == "full") {
      m_recycling_mode = RecyclingMode::Full;
    } else {
      amrex::Abort(
        "peleLM.recycling_mode must be 'fluctuations' or 'full', got '" +
        recycling_mode + "'");
    }
    pp.query("inlet_plane_flux_control", m_inlet_plane_flux_control);
    {
      amrex::Real target_flow = 0.0;
      if (pp.query("inlet_plane_target_flow_rate", target_flow) != 0) {
        m_inlet_plane_target_flow = target_flow;
      }
    }
    if (
      m_recycling_mode == RecyclingMode::Fluctuations &&
      (pp.contains("inlet_plane_flux_control") ||
       pp.contains("inlet_plane_target_flow_rate"))) {
      amrex::Print()
        << "WARNING: peleLM.inlet_plane_flux_control and "
           "peleLM.inlet_plane_target_flow_rate are ignored when "
           "peleLM.recycling_mode = fluctuations; the mean inflow is "
           "anchored by bcnormal in that mode.\n";
    }
    if (m_recycling_mode == RecyclingMode::Full) {
      if (m_inlet_plane_flux_control != 0) {
        amrex::Print()
          << " Recycling in full mode with mass-flux control: the injected "
             "sample is rescaled to the target inlet flow rate every "
             "step.\n";
      } else {
        amrex::Print()
          << "WARNING: peleLM.inlet_plane_flux_control = 0: full-mode "
             "recycling has no anchor on the inlet flow rate. Any net heat "
             "transfer between the inlet and the recycling plane will "
             "drive a monotonic drift of the bulk flow (see the recycling "
             "docs); use this only for adiabatic constant-area ducts.\n";
      }
      // These controls belong to the running-mean machinery and have no
      // effect in full mode.
      if (m_inlet_plane_warmup_steps > 0) {
        amrex::Print()
          << "WARNING: peleLM.inlet_plane_warmup_steps is ignored when "
             "peleLM.recycling_mode = full; injection begins with the "
             "first snapshot.\n";
      }
      if (m_inlet_plane_avg_window > 0.0) {
        amrex::Print()
          << "WARNING: peleLM.inlet_plane_avg_window is ignored when "
             "peleLM.recycling_mode = full; no running mean is "
             "maintained.\n";
      }
    }
  }

#ifdef PELE_USE_PLASMA
  amrex::ParmParse ppef("ef");

  // Get the phiV bc
  amrex::Vector<std::string> lo_bc_char(AMREX_SPACEDIM);
  amrex::Vector<std::string> hi_bc_char(AMREX_SPACEDIM);
  ppef.getarr("phiV_lo_bc", lo_bc_char, 0, AMREX_SPACEDIM);
  ppef.getarr("phiV_hi_bc", hi_bc_char, 0, AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (lo_bc_char[idim] == "Interior") {
      m_phiV_bc.setLo(idim, 0);
    } else if (lo_bc_char[idim] == "Dirichlet") {
      m_phiV_bc.setLo(idim, 1);
    } else if (lo_bc_char[idim] == "Neumann") {
      m_phiV_bc.setLo(idim, 2);
    } else {
      amrex::Abort("Wrong PhiV bc. Should be : Interior, Dirichlet or Neumann");
    }
    if (hi_bc_char[idim] == "Interior") {
      m_phiV_bc.setHi(idim, 0);
    } else if (hi_bc_char[idim] == "Dirichlet") {
      m_phiV_bc.setHi(idim, 1);
    } else if (hi_bc_char[idim] == "Neumann") {
      m_phiV_bc.setHi(idim, 2);
    } else {
      amrex::Abort("Wrong PhiV bc. Should be : Interior, Dirichlet or Neumann");
    }
  }

  // Get the polarity of BCs
  ppef.getarr("phiV_polarity_lo", lo_bc_char, 0, AMREX_SPACEDIM);
  ppef.getarr("phiV_polarity_hi", hi_bc_char, 0, AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (lo_bc_char[idim] == "Neutral") {
      m_phiV_bcpol.setLo(idim, 0);
    } else if (lo_bc_char[idim] == "Anode") { // Pos. elec = 1
      m_phiV_bcpol.setLo(idim, 1);
      AMREX_ASSERT(m_phiV_bc.lo(idim) == 1);
    } else if (lo_bc_char[idim] == "Cathode") { // Neg. elec = 2
      m_phiV_bcpol.setLo(idim, 2);
      AMREX_ASSERT(m_phiV_bc.lo(idim) == 1);
    } else {
      amrex::Abort(
        "Wrong PhiV polarity. Should be : Neutral, Anode or Cathode");
    }
    if (hi_bc_char[idim] == "Neutral") {
      m_phiV_bcpol.setHi(idim, 0);
    } else if (hi_bc_char[idim] == "Anode") { // Pos. elec = 1
      m_phiV_bcpol.setHi(idim, 1);
      AMREX_ASSERT(m_phiV_bc.hi(idim) == 1);
    } else if (hi_bc_char[idim] == "Cathode") { // Neg. elec = 2
      m_phiV_bcpol.setHi(idim, 2);
      AMREX_ASSERT(m_phiV_bc.hi(idim) == 1);
    } else {
      amrex::Abort(
        "Wrong PhiV polarity. Should be : Neutral, Anode or Cathode");
    }
  }
#endif

  // -----------------------------------------
  // Algorithm
  // -----------------------------------------

  // -----------------------------------------
  // incompressible vs. low Mach
  pp.query("use_divu", m_has_divu);
  pp.query("incompressible", m_incompressible);
  if (m_incompressible != 0) {
    m_has_divu = 0;
    m_do_react = 0;
    pp.query("rho", m_rho);
    pp.query("mu", m_mu);
    AMREX_ASSERT_WITH_MESSAGE(
      m_rho > 0.0, "peleLM.rho is needed when running incompressible");
    AMREX_ASSERT_WITH_MESSAGE(
      m_mu > 0.0, "peleLM.mu is needed when running incompressible");
  }
  amrex::Vector<amrex::Real> grav(AMREX_SPACEDIM, 0);
  pp.queryarr("gravity", grav, 0, AMREX_SPACEDIM);
  amrex::Vector<amrex::Real> gp0(AMREX_SPACEDIM, 0);
  pp.queryarr("gradP0", gp0, 0, AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    m_background_gp[idim] = gp0[idim];
    m_gravity[idim] = grav[idim];
  }

  // Will automatically add pressure gradient for channel flow to maintain mass
  // flow rate of initial condition
  pp.query("do_periodic_channel", m_do_periodic_channel);
  if (m_do_periodic_channel != 0) {
    pp.get("periodic_channel_dir", m_periodic_channel_dir);
  }

  // Add a "spark", i.e. sphere of heat source to energy equation
  m_n_sparks = pp.countval("sparks");
  if (m_n_sparks > 0) {
    m_spark.resize(m_n_sparks);
    m_spark_time.resize(m_n_sparks);
    m_spark_duration.resize(m_n_sparks);
    m_spark_location.resize(m_n_sparks);
    m_spark_temp.resize(m_n_sparks);
    m_spark_radius.resize(m_n_sparks);
    if (m_n_sparks > 0) {
      m_spark_verbose = amrex::max<int>(m_verbose - 2, 0);
    }
    pp.query("spark_verbose", m_spark_verbose);
    for (int n = 0; n < m_n_sparks; ++n) {
      pp.get("sparks", m_spark[n], n);
      std::string spark_prefix = "peleLM." + m_spark[n];
      amrex::ParmParse pps(spark_prefix);
      pps.get("time", m_spark_time[n]);
      pps.get("duration", m_spark_duration[n]);
      m_spark_location[n].resize(AMREX_SPACEDIM);
      pps.getarr("location", m_spark_location[n], 0, AMREX_SPACEDIM);
      pps.get("temp", m_spark_temp[n]);
      pps.get("radius", m_spark_radius[n]);
    }
    if (m_spark_verbose > 0) {
      amrex::Print() << "Spark list: \n";
      for (int n = 0; n < m_n_sparks; ++n) {
        amrex::Print() << "Spark " << n << " name: " << m_spark[n] << "\n";
        amrex::Print() << "Spark " << n << " time: " << m_spark_time[n] << "\n";
        amrex::Print() << "Spark " << n << " duration: " << m_spark_duration[n]
                       << "\n";
        amrex::Print() << "Spark " << n << " location: ";
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          amrex::Print() << m_spark_location[n][d] << " ";
        }
        amrex::Print() << "\n";
        amrex::Print() << "Spark " << n << " temperature: " << m_spark_temp[n]
                       << "\n";
        amrex::Print() << "Spark " << n << " radius: " << m_spark_radius[n]
                       << "\n";
      }
    }
  }

  m_nAux = pp.countval("aux_vars");
  if (m_nAux > 0) {
    m_aux_names.resize(m_nAux);
    m_AdvTypeAux.resize(m_nAux);
    m_aux_advect.resize(m_nAux);
    m_DiffTypeAux.resize(m_nAux);
    m_aux_Schmidt.resize(m_nAux);
    for (int n = 0; n < m_nAux; ++n) {
      pp.get("aux_vars", m_aux_names[n], n);
      std::string aux_prefix = "peleLM." + m_aux_names[n];
      amrex::ParmParse ppa(aux_prefix);
      m_aux_advect[n] = 1;
      ppa.query("advect", m_aux_advect[n]);
      m_AdvTypeAux[n] = 1;
      ppa.query("conservative", m_AdvTypeAux[n]);
      m_aux_Schmidt[n] = -1.0;
      ppa.query("Schmidt", m_aux_Schmidt[n]);
      int diffuse = 1;
      ppa.query("diffuse", diffuse);
      if (diffuse == 0) {
        m_DiffTypeAux[n] = 0;
      } else {
        m_DiffTypeAux[n] = 1;
      }
    }
  }

  // -----------------------------------------
  // LES
  // -----------------------------------------
  pp.query("les_model", m_les_model);
  if (m_les_model == "None") {
    m_do_les = false;
  } else {
    if (m_les_model == "Smagorinsky") {
      pp.query("les_cs_smag", m_les_cs_smag);
    } else if (m_les_model == "WALE") {
      pp.query("les_cm_wale", m_les_cm_wale);
    } else if (m_les_model == "Sigma") {
      pp.query("les_cs_sigma", m_les_cs_sigma);
      AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 3); // Sigma only available in 3D
    } else {
      amrex::Abort(
        "LES model must be None, Smagorinsky, WALE or Sigma. Invalid choie: " +
        m_les_model);
    }
    m_do_les = true;
    m_les_verbose = m_verbose;
    pp.query("plot_les", m_plot_les);
    pp.query("les_v", m_les_verbose);
    pp.query("les_c_chi", m_les_c_chi);
#ifdef PELE_USE_PLASMA
    amrex::Abort("LES implementation is not yet compatible with plasma/ions");
#endif
  }

  // -----------------------------------------
  // diffusion
  amrex::ParmParse pptrans("transport");
  pptrans.query("use_soret", m_use_soret);
  pp.query("use_wbar", m_use_wbar);
  if (m_use_soret != 0) {
    bool isothermal = false;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      isothermal |=
        (m_phys_bc.lo(idim) == BoundaryCondition::BCSlipWallIsotherm ||
         m_phys_bc.lo(idim) == BoundaryCondition::BCNoSlipWallIsotherm ||
         m_phys_bc.hi(idim) == BoundaryCondition::BCSlipWallIsotherm ||
         m_phys_bc.hi(idim) == BoundaryCondition::BCNoSlipWallIsotherm);
    }
    if (isothermal) {
      m_soret_boundary_override = 1;
      m_use_wbar = 0;
#if PELE_USE_PLASMA
      amrex::Abort("Isothermal walls with Soret incompatible with plasma");
#endif
    }
  }
  pp.query("unity_Le", m_unity_Le);
  pp.query("fixed_Le", m_fixed_Le);
  pp.query("fixed_Pr", m_fixed_Pr);
  if (m_unity_Le != 0) {
    m_fixed_Le = 1;
    amrex::Print() << "WARNING: unity_Le is deprecated and will be removed in "
                      "future version, use fixed_Le instead \n";
  }
  // LES: Schmidt and Prandtl are specified for turbulent diffusion
  // fixed_Pr/fixed_Le: Lewis and Prandtl are specified for molecular diffusion
  // Both: Specify Prandtl and Schmidt, apply for both diffusion types
  if (m_fixed_Pr != 0 || m_do_les) {
    amrex::Real Prandtl = 0.7;
    pp.query("Prandtl", Prandtl);
    m_Prandtl_inv = 1.0 / Prandtl;
  }
  if (m_do_les) {
    amrex::Real Schmidt = 0.7;
    pp.query("Schmidt", Schmidt);
    m_Schmidt_inv = 1.0 / Schmidt;
    m_Lewis_inv = m_Schmidt_inv / m_Prandtl_inv;
  } else if (m_fixed_Le != 0) {
    amrex::Real Lewis = 1.0;
    pp.query("Lewis", Lewis);
    m_Lewis_inv = 1.0 / Lewis;
    if (m_fixed_Pr != 0) {
      m_Schmidt_inv = m_Lewis_inv * m_Prandtl_inv;
    }
  }
  if (
    (m_use_wbar != 0 || m_use_soret != 0) &&
    (m_fixed_Le != 0 || m_fixed_Pr != 0)) {
    m_use_wbar = 0;
    m_use_soret = 0;
    amrex::Print() << "WARNING: use_wbar and use_soret set to false because "
                      "fixed_Pr or fixed_Le is true \n";
  }

  // Manifold EOS: invPrandtl needs to be 0 because H not used
  if (pele::physics::PhysicsType::eos_type::identifier() == "Manifold") {
    m_Prandtl_inv = 0.0;
  }

  m_deltaT_verbose = amrex::max<int>(m_verbose - 1, m_deltaT_verbose);
  pp.query("deltaT_verbose", m_deltaT_verbose);
  pp.query("deltaT_iterMax", m_deltaTIterMax);
  pp.query("deltaT_tol", m_deltaT_norm_max);
  pp.query("deltaT_crashIfFailing", m_crashOnDeltaTFail);

  // -----------------------------------------
  // initialization
  pp.query("num_divu_iter", m_numDivuIter);
  pp.query("do_init_proj", m_do_init_proj);
  pp.query("num_init_iter", m_init_iter);
  pp.query("initDataPlt_patch_flow_variables", m_do_patch_flow_variables);
  pp.queryarr("initDataPlt_specname_map", m_initDataPlt_specname_map);
  if (
    !m_initDataPlt_specname_map.empty() &&
    m_initDataPlt_specname_map.size() != NUM_SPECIES) {
    amrex::Abort(
      "If specifying a species name map, length must equal number of species");
  }
  pp.query("initDataPlt_reset_time", m_do_reset_time);
  pp.query("initDataPlt_coarsen", m_initDataPlt_coarsen);

  // -----------------------------------------
  // advance
  // -----------------------------------------
  pp.query("sdc_iterMax", m_nSDCmax);
  m_print_chi_convergence = m_verbose > 1;
  pp.query("print_chi_convergence", m_print_chi_convergence);
  parseUserKey(pp, "chi_correction_type", chicorr, m_chi_correction_type);
  pp.query("floor_species", m_floor_species);
  pp.query("dPdt_factor", m_dpdtFactor);
  pp.query("memory_checks", m_checkMem);
  pp.query("divu_dt_factor", m_divu_dtFactor);
  pp.query("divu_dt_rhoMin", m_divu_rhoMin);
  pp.query("divu_dt_method", m_divu_checkFlag);

  // -----------------------------------------
  // Reaction
  // -----------------------------------------
  pp.query("do_react", m_do_react);
  pp.query("use_typ_vals_chem", m_useTypValChem);
  pp.query("typical_values_reset_int", m_resetTypValInt);
  pp.query("typical_values_Ymin", m_typicalYvalMin);
  if (m_do_react != 0) {
    m_plotChemDiag = 0;
    m_plotHeatRelease = 1;
    pp.query("plot_chemDiagnostics", m_plotChemDiag);
    pp.query("plot_heatRelease", m_plotHeatRelease);
  }
  // Enable the chemistry BA to have smaller grid size
  int mgsc_size = pp.countval("max_grid_size_chem");
  if (mgsc_size > 0) {
    if (mgsc_size == 1) {
      int mgsc;
      pp.query("max_grid_size_chem", mgsc);
      AMREX_D_TERM(
        m_max_grid_size_chem[0] = mgsc;, m_max_grid_size_chem[1] = mgsc;
        , m_max_grid_size_chem[2] = mgsc);
    } else if (mgsc_size == AMREX_SPACEDIM) {
      amrex::Vector<int> mgsc;
      pp.getarr("max_grid_size_chem", mgsc, 0, AMREX_SPACEDIM);
      AMREX_D_TERM(
        m_max_grid_size_chem[0] = mgsc[0];, m_max_grid_size_chem[1] = mgsc[1];
        , m_max_grid_size_chem[2] = mgsc[2]);
    } else {
      amrex::Abort(
        "peleLM.max_grid_size_chem should have 1 or AMREX_SPACEDIM values");
    }
  }

  // -----------------------------------------
  // Turbulent Forcing
  // -----------------------------------------
  pp.query("do_turbulent_forcing", m_do_turbulent_forcing);
  // Add turbulent velocity from an existing plotfile
  pp.query("velocity_plotfile", m_velocity_plotfile);
  pp.query("velocity_plotfile_scale", m_velocity_plotfile_scale);

  // -----------------------------------------
  // Load Balancing
  // -----------------------------------------
  pp.query("do_load_balancing", m_doLoadBalance);
  parseUserKey(pp, "load_balancing_method", lbmethod, m_loadBalanceMethod);
  parseUserKey(pp, "load_balancing_cost_estimate", lbcost, m_loadBalanceCost);
  pp.query(
    "load_balancing_efficiency_threshold", m_loadBalanceEffRatioThreshold);
  parseUserKey(
    pp, "chem_load_balancing_method", lbmethod, m_loadBalanceMethodChem);
  parseUserKey(
    pp, "chem_load_balancing_cost_estimate", lbcost, m_loadBalanceCostChem);

  // Deactivate load balancing for serial runs
#ifdef AMREX_USE_MPI
  if (amrex::ParallelContext::NProcsSub() == 1) {
    m_doLoadBalance = 0;
  }
#else
  m_doLoadBalance = 0;
#endif

  // -----------------------------------------
  // Advection
  // -----------------------------------------
  pp.query("advection_scheme", m_advection_key);
  if (m_advection_key == "Godunov_PLM") {
    m_advection_type = "Godunov";
    m_Godunov_ppm = 0;
    amrex::ParmParse ppg("godunov");
    ppg.query("use_forceInTrans", m_Godunov_ForceInTrans);
  } else if (m_advection_key == "Godunov_PPM") {
    m_advection_type = "Godunov";
    m_Godunov_ppm = 1;
    m_Godunov_ppm_limiter = PPM::VanLeer;
    amrex::ParmParse ppg("godunov");
    ppg.query("use_forceInTrans", m_Godunov_ForceInTrans);
  } else if (m_advection_key == "Godunov_PPM_WENOZ") {
    m_advection_type = "Godunov";
    m_Godunov_ppm = 1;
    m_Godunov_ppm_limiter = PPM::WENOZ;
    amrex::ParmParse ppg("godunov");
    ppg.query("use_forceInTrans", m_Godunov_ForceInTrans);
  } else if (m_advection_key == "Godunov_PPM_NOLIM") {
    m_advection_type = "Godunov";
    m_Godunov_ppm = 1;
    m_Godunov_ppm_limiter = PPM::NoLimiter;
    amrex::ParmParse ppg("godunov");
    ppg.query("use_forceInTrans", m_Godunov_ForceInTrans);
  } else if (m_advection_key == "Godunov_BDS") {
    m_advection_type = "BDS";
    m_Godunov_ppm = 0;
  } else {
    amrex::Abort(
      "Unknown 'advection_scheme'. Recognized options are: Godunov_PLM, "
      "Godunov_PPM or Godunov_BDS");
  }
  m_predict_advection_type =
    "Godunov"; // Only option at this point. This will disappear when
               // predict_velocity support BDS.

  // -----------------------------------------
  // Linear solvers tols
  // -----------------------------------------
  amrex::ParmParse ppnproj("nodal_proj");
  ppnproj.query("mg_max_coarsening_level", m_nodal_mg_max_coarsening_level);
  ppnproj.query("atol", m_nodal_mg_atol);
  ppnproj.query("rtol", m_nodal_mg_rtol);
  ppnproj.query("hypre_namespace", m_hypre_namespace_nodal);
  m_nproj_verbose = amrex::max<int>(m_verbose - 2, 0);
  ppnproj.queryAdd("verbose", m_nproj_verbose);

  amrex::ParmParse ppmacproj("mac_proj");
  ppmacproj.query("mg_max_coarsening_level", m_mac_mg_max_coarsening_level);
  ppmacproj.query("atol", m_mac_mg_atol);
  ppmacproj.query("rtol", m_mac_mg_rtol);
  ppmacproj.query("hypre_namespace", m_hypre_namespace_mac);
  ppmacproj.query("mlmg_fail_sdc_miniter", m_mac_mg_fail_sdc_miniter);
  ppmacproj.query(
    "mlmg_fail_maxiter_after_sdc_miniter",
    m_mac_mg_fail_maxiter_after_sdc_miniter);
  m_macproj_verbose = amrex::max<int>(m_verbose - 2, 0);
  ppmacproj.queryAdd("verbose", m_macproj_verbose);

  // -----------------------------------------
  // Temporals
  // -----------------------------------------
  pp.query("do_temporals", m_do_temporals);
  if (m_do_temporals != 0) {
    pp.query("temporal_int", m_temp_int);
    pp.query("temporal_dir", m_temporal_dir);
    m_temporal_dir = m_base_output_pref + m_temporal_dir;
    pp.query("do_extremas", m_do_extremas);
    pp.query("do_mass_balance", m_do_massBalance);
    pp.query("do_species_balance", m_do_speciesBalance);
    pp.query("do_patch_mfr", m_do_patch_mfr);
  }

  // -----------------------------------------
  // Time stepping control
  // -----------------------------------------
  amrex::ParmParse ppa("amr");
  ppa.query("max_wall_time", m_max_wall_time); // hours
  ppa.query("max_step", m_max_step);
  ppa.query("stop_time", m_stop_time);
  ppa.query("message_int", m_message_int);
  ppa.query("fixed_dt", m_fixed_dt);
  ppa.query("init_dt", m_init_dt);
  ppa.query("cfl", m_cfl);
  ppa.query("dt_shrink", m_dtshrink);
  ppa.query("dt_change_max", m_dtChangeMax);
  ppa.query("max_dt", m_max_dt);
  ppa.query("min_dt", m_min_dt);
  m_nfiles = amrex::Clamp(amrex::ParallelDescriptor::NProcs(), 1, 256);
  ppa.query("n_files", m_nfiles);

  if (max_level > 0 || (m_doLoadBalance != 0)) {
    ppa.query("regrid_int", m_regrid_int);
    ppa.query("regrid_on_restart", m_regrid_on_restart);
  }

#ifdef AMREX_USE_EB
  if (max_level > 0) {
    // Default EB refine type is Static
    pp.query("refine_EB_type", m_EB_refine_type);
    if (m_EB_refine_type != "Static" && m_EB_refine_type != "Adaptive") {
      amrex::Abort("refine_EB_type can only be 'Static' or 'Adaptive'");
    }
    // Default EB refinement level is max_level
    m_EB_refine_LevMax = max_level;
    pp.query("refine_EB_max_level", m_EB_refine_LevMax);
    pp.query("refine_EB_buffer", m_derefineEBBuffer);
    if (m_EB_refine_type == "Adaptive") {
      m_EB_refine_LevMin = 0;
      pp.query("refine_EB_min_level", m_EB_refine_LevMin);
      m_EB_refine_LevAdapt = m_EB_refine_LevMin;
    }
    if (m_EB_refine_LevMax < max_level) {
      m_signDistNeeded = 1;
    }
  }
  pp.query("isothermal_EB", m_isothermalEB);
  pp.query("adv_redist_type", m_adv_redist_type);
  pp.query("diff_redist_type", m_diff_redist_type);
  pp.query("EBinflow", m_useEBinflow);
  if (m_isothermalEB != 0 || m_useEBinflow != 0) {
    checkEBInflowFunctions();
  }
#endif

  // -----------------------------------------
  // Evaluate mode variables
  // -----------------------------------------
  if (runMode() == "evaluate") {
    m_evaluatePlotVarCount = (pp.countval("evaluate_vars"));
    if (m_evaluatePlotVarCount != 0) {
      m_evaluatePlotVars.resize(m_evaluatePlotVarCount);
      for (int ivar = 0; ivar < m_evaluatePlotVarCount; ++ivar) {
        pp.get("evaluate_vars", m_evaluatePlotVars[ivar], ivar);
      }
    }
  }

#ifdef PELE_USE_PLASMA
  // -----------------------------------------
  // PLASMA
  // -----------------------------------------
  ppef.query("JFNK_newtonTol", m_ef_newtonTol);
  ppef.query("JFNK_maxNewton", m_ef_maxNewtonIter);
  ppef.query("JFNK_lambda", m_ef_lambda_jfnk);
  ppef.query("JFNK_diffType", m_ef_diffT_jfnk);
  AMREX_ASSERT(m_ef_diffT_jfnk == 1 || m_ef_diffT_jfnk == 2);
  ppef.query("GMRES_rel_tol", m_ef_GMRES_reltol);
  ppef.query("GMRES_abs_tol", m_ef_GMRES_abstol);
  ppef.query("PC_approx", m_ef_PC_approx);
  ppef.query("PC_damping", m_ABecCecOmega);
  ppef.query("advection_scheme_order", m_nEAdvOrder);
  AMREX_ASSERT(m_nEAdvOrder == 1 || m_nEAdvOrder == 2);

  ppef.query("tabulated_Ke", m_electronKappaTab);
  ppef.query("fixed_Ke", m_fixedKappaE);

  ppef.query("restart_nonEF", m_restart_nonEF);
  ppef.query("restart_electroneutral", m_restart_electroneutral);
  ppef.query("restart_resetTime", m_restart_resetTime);
  ppef.query("plot_extras", m_do_extraEFdiags);

  // Getting the ions fluxes on the domain boundaries
  // Species balance data is needed, so override if not activated
  if (m_do_temporals) {
    ppef.query("do_ionsBalance", m_do_ionsBalance);
    if (m_do_ionsBalance) {
      m_do_speciesBalance = 1;
    }
  }
#endif

#ifdef PELE_USE_SPRAY
  SprayReadParameters();
#endif
#ifdef PELE_USE_SOOT
  do_soot_solve = true;
  pp.query("do_soot_solve", do_soot_solve);
  if ((m_verbose != 0) && do_soot_solve) {
    amrex::Print() << "Simulation performed with soot modeling \n";
  }
  soot_model->readSootParams();
#endif
#ifdef PELE_USE_RADIATION
  do_rad_solve = false;
  pp.query("do_rad_solve", do_rad_solve);
  if ((m_verbose != 0) && do_rad_solve) {
    amrex::Print() << "Simulation performed with radiation modeling \n";
  }
#endif

  // -----------------------------------------
  // External Sources
  // -----------------------------------------
  m_user_defined_ext_sources = false;
  m_ext_sources_SDC = false; // TODO: add capability to update ext_srcs in SDC
  m_plot_extSource = false;
  m_add_variance_sources = true;
  pp.query("user_defined_ext_sources", m_user_defined_ext_sources);
  pp.query("plot_extSource", m_plot_extSource);
  pp.query("add_variance_sources", m_add_variance_sources);
}

void
PeleLM::checkSetupParams()
{
  BL_PROFILE("PeleLMeX::checkSetupParams()");
  // Ensure unsupported physics is not used with manifiold models
  if (pele::physics::PhysicsType::eos_type::identifier() == "Manifold") {
    if (m_closed_chamber != 0) {
      amrex::Abort(
        "Simulation with closed chamber is not yet supported for Manifold EOS");
    }
    if (m_use_wbar != 0) {
      amrex::Abort("Use of Wbar fluxes is not compatible with Manifold EOS");
    }
#ifdef PELE_USE_RADIATION
    if (do_rad_solve) {
      amrex::Abort("Radiation models are not yet supported for Manifold EOS");
    }
#endif
#ifdef PELE_USE_SOOT
    if (do_soot_solve) {
      amrex::Abort("Soot models are not yet supported for Manifold EOS");
    }
#endif
#ifdef PELE_USE_SPRAY
    amrex::Print()
      << "WARNING: Spray models with Manifold EOS are experimental !!!!\n";
#endif
#ifdef PELE_USE_PLASMA
    amrex::Abort("Plasma models are not yet supported for Manifold EOS");
#endif
#ifdef USE_MANIFOLD_EOS
    if (
      std::abs(
        (0.1 * eos_parms.host_parm().Pnom_cgs - prob_parm->P_mean) /
        prob_parm->P_mean) > 1e-6) {
      amrex::Abort(
        "For Manifold EOS, pressure in manifold model "
        "(manifold.nominal_pressure_cgs) and pressure in PeleLMeX "
        "(prob.Pmean) must match");
    }
#endif
  }

  // ---------------------------------------------------------------------
  // Mesh mapping x inflow data
  // ---------------------------------------------------------------------
  // A turbulence file is uniform in *some* coordinate: physical position
  // for synthetic or uniform-precursor data, the precursor's Xi coordinate
  // for planes extracted from a mesh-mapped run.  Either way the file is
  // sampled by the physical position of each target cell, which on a
  // mesh-mapped grid is not ProbLo + (i+0.5)*dx: that needs the PelePhysics
  // add_turb() overload taking explicit coordinates, and sampling a
  // Xi-uniform file additionally needs a PelePhysics that inverts the
  // file's map.
  if (m_mesh_mapping && turb_inflow.is_initialized()) {
#ifndef PELEPHYSICS_TURBINFLOW_HAS_COORD_ADDTURB
    amrex::Abort(
      "geometry.mesh_mapping combined with turbinflow requires a "
      "PelePhysics that provides TurbInflow::add_turb() with explicit "
      "physical coordinates (PELEPHYSICS_TURBINFLOW_HAS_COORD_ADDTURB).\n"
      "Update Submodules/PelePhysics, or drop geometry.mesh_mapping.");
#endif
  }
  if (turb_inflow.is_initialized() && (verbose != 0)) {
    reportTurbInflowMaps();
  }

  if (m_mesh_mapping && (m_use_inlet_from_plane != 0) && (verbose != 0)) {
    const int dir = m_inlet_plane_dir;
    amrex::Print() << " Recycling source plane: physical position "
                   << m_inlet_plane_position << " -> Xi position "
                   << m_map_eval.xi_from_x_phys(
                        dir, m_inlet_plane_position, geom[0].data())
                   << " (level-0 index " << computeRecyclingSrcIndex(0)
                   << ")\n";
  }

  if (turb_inflow.is_initialized() && (verbose != 0)) {
    reportTurbInflowResolution();
  }
}

void
PeleLM::reportTurbInflowMaps()
{
#ifndef PELEPHYSICS_TURBINFLOW_SAMPLES_MESHMAP
  // Older PelePhysics: files without a MESHMAP trailer are taken to be
  // uniform in physical position; a reader that knows the trailer refuses
  // files that carry one, an older one cannot tell them apart.
  if (m_mesh_mapping) {
#ifdef PELEPHYSICS_TURBINFLOW_HAS_MESHMAP_HDR
    amrex::Print()
      << " NOTE: turbinflow data without a MESHMAP trailer is taken to be "
         "uniform in physical\n       position; only the target grid is "
         "stretched.  Files extracted from a mesh-mapped\n       precursor "
         "are refused by this PelePhysics.\n";
#else
    amrex::Print()
      << " WARNING: turbinflow data is assumed uniform in physical position. "
         "This PelePhysics cannot\n          detect a file extracted from a "
         "mesh-mapped precursor (uniform in that run's Xi\n          "
         "coordinate); injecting one here would be silently wrong.\n";
#endif
  }
#else
  // For every inflow face fed by a turbulence file, say which coordinate
  // the file is uniform in and how it relates to this run's mesh: a file
  // whose map and Xi grid coincide with ours is injected by exact index
  // (the same-grid guarantee, now also for mapped meshes); anything else
  // is interpolated -- in physical position for a physically-uniform file,
  // in the file's Xi for a mapped one.
  const auto gd = geom[0].data();
  const amrex::Box& dom = geom[0].Domain();
  auto velBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    for (int iside = 0; iside < 2; ++iside) {
      const bool is_low = (iside == 0);
      const auto side =
        is_low ? amrex::Orientation::low : amrex::Orientation::high;
      const int bctype = is_low ? velBCRec[0].lo()[dir] : velBCRec[0].hi()[dir];
      if (bctype != amrex::BCType::ext_dir) {
        continue;
      }
      MeshMapEvaluator fmap;
      amrex::Real xi_lo[2] = {0.0, 0.0};
      amrex::Real xi_hi[2] = {0.0, 0.0};
      const int has = turb_inflow.file_map(dir, side, fmap, xi_lo, xi_hi);
      if (has < 0) {
        continue;
      }
      amrex::Print() << " turbInflow on face (dir " << dir << ", "
                     << (is_low ? "low" : "high") << "): ";
      if (has == 0) {
        amrex::Print() << "file is uniform in physical position; sampled at "
                          "the physical positions of this "
                       << (m_mesh_mapping ? "mapped" : "uniform") << " mesh\n";
        continue;
      }
      int tdir1 = 0;
      int tdir2 = 0;
      pele::physics::turbinflow::TurbInflow::transverseDirs(dir, tdir1, tdir2);
      const int tdir[2] = {tdir1, tdir2};
      // Same map on the two transverse axes (file slots 0/1 against our
      // axes tdir1/tdir2) and the same Xi extent means the file's Xi grid
      // is a uniform grid on our own transverse domain; with equal cell
      // counts every target cell centre is a file cell centre.
      // The file's parameters went through a text trailer, so compare
      // them to ours up to a few ULPs rather than exactly.
      auto near = [](amrex::Real a, amrex::Real b) {
        return amrex::almostEqual(a, b, 8);
      };
      bool same = (fmap.m_kind == m_map_eval.m_kind);
      for (int t = 0; t < 2 && same; ++t) {
        const int d = tdir[t];
        same = near(fmap.m_p[t], m_map_eval.m_p[d]) &&
               near(fmap.m_p2[t], m_map_eval.m_p2[d]) &&
               near(fmap.m_p3[t], m_map_eval.m_p3[d]) &&
               (fmap.m_q[t] == m_map_eval.m_q[d]) &&
               near(xi_lo[t], gd.ProbLo()[d]) && near(xi_hi[t], gd.ProbHi()[d]);
      }
      amrex::Print() << "file is uniform in its precursor's Xi (map kind "
                     << static_cast<int>(fmap.m_kind) << "); ";
      if (same) {
        amrex::Print() << "map and Xi extent match this mesh -> ";
        amrex::Real dmin[2];
        amrex::Real dmax[2];
        bool same_cells =
          turb_inflow.file_transverse_dx_range(dir, side, dmin, dmax);
        // Same map and extent: the smallest physical cell is a monotone
        // function of the cell count, so equal minima mean equal counts.
        for (int t = 0; t < 2 && same_cells; ++t) {
          const int d = tdir[t];
          amrex::Real gmin = std::numeric_limits<amrex::Real>::max();
          for (int i = dom.smallEnd(d); i <= dom.bigEnd(d); ++i) {
            gmin = std::min(gmin, m_map_eval.dx_phys_cc(d, i, gd));
          }
          same_cells = std::abs(dmin[t] - gmin) <= 1.0e-10 * gmin;
        }
        amrex::Print()
          << (same_cells ? "exact index injection\n"
                         : "interpolation in Xi (different cell count)\n");
      } else {
        amrex::Print() << (m_mesh_mapping ? "this mesh's map differs"
                                          : "this mesh is uniform")
                       << " -> interpolation in the file's Xi\n";
      }
    }
  }
#endif
}

void
PeleLM::reportTurbInflowResolution()
{
#ifndef PELEPHYSICS_TURBINFLOW_HAS_COORD_ADDTURB
  // The accessors this needs (file_transverse_dx, transverseDirs) arrived
  // with the coordinate-based add_turb overload; without them there is
  // nothing to report.
  return;
#else
  // Compare the turbulence file's transverse spacing against the level-0
  // grid's physical spacing on each inflow face.  The ratio is the physical
  // constraint on injecting a uniformly-spaced file: substantially above one
  // and the file cannot fill the scales the grid resolves; substantially
  // below one and the injected field is aliased onto the grid.  Neither is
  // an error, so this only reports.
  const auto gd = geom[0].data();
  const amrex::Box& dom = geom[0].Domain();
  auto velBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);

  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    for (int iside = 0; iside < 2; ++iside) {
      const bool is_low = (iside == 0);
      const auto side =
        is_low ? amrex::Orientation::low : amrex::Orientation::high;
      const int bctype = is_low ? velBCRec[0].lo()[dir] : velBCRec[0].hi()[dir];
      if (bctype != amrex::BCType::ext_dir) {
        continue;
      }
      amrex::Real file_dx[2] = {0.0, 0.0};
      if (!turb_inflow.file_transverse_dx(dir, side, file_dx[0], file_dx[1])) {
        continue;
      }
#ifdef PELEPHYSICS_TURBINFLOW_SAMPLES_MESHMAP
      // A mesh-mapped file's physical spacing varies across the plane; the
      // ratio is then bounded by the two spacing ranges rather than by a
      // single file value.
      amrex::Real file_dmin[2] = {0.0, 0.0};
      amrex::Real file_dmax[2] = {0.0, 0.0};
      turb_inflow.file_transverse_dx_range(dir, side, file_dmin, file_dmax);
#else
      const amrex::Real* file_dmin = file_dx;
      const amrex::Real* file_dmax = file_dx;
#endif

      int tdir1 = 0;
      int tdir2 = 0;
      pele::physics::turbinflow::TurbInflow::transverseDirs(dir, tdir1, tdir2);
      const int tdir[2] = {tdir1, tdir2};

      amrex::Print() << " turbInflow resolution check, face (dir " << dir
                     << ", " << (is_low ? "low" : "high") << "):\n";
      for (int t = 0; t < 2; ++t) {
        if (file_dx[t] <= 0.0) {
          continue;
        }
        amrex::Real dmin = std::numeric_limits<amrex::Real>::max();
        amrex::Real dmax = 0.0;
        for (int i = dom.smallEnd(tdir[t]); i <= dom.bigEnd(tdir[t]); ++i) {
          const amrex::Real d = m_map_eval.dx_phys_cc(tdir[t], i, gd);
          dmin = std::min(dmin, d);
          dmax = std::max(dmax, d);
        }
        amrex::Print() << "   dir " << tdir[t] << ": grid dx in [" << dmin
                       << ", " << dmax << "], file dx ";
        if (amrex::almostEqual(file_dmin[t], file_dmax[t])) {
          amrex::Print() << file_dx[t] << " -> dx_grid/dx_file in ["
                         << dmin / file_dx[t] << ", " << dmax / file_dx[t]
                         << "]\n";
        } else if (
          std::abs(file_dmin[t] - dmin) <= 1.0e-10 * dmin &&
          std::abs(file_dmax[t] - dmax) <= 1.0e-10 * dmax) {
          // Same stretched grid on both sides: the pointwise ratio is one
          // everywhere, which the min/max bound below would hide.
          amrex::Print() << "in [" << file_dmin[t] << ", " << file_dmax[t]
                         << "] -> same spacing range as the grid, "
                            "dx_grid/dx_file = 1 pointwise\n";
        } else {
          amrex::Print() << "in [" << file_dmin[t] << ", " << file_dmax[t]
                         << "] (mean " << file_dx[t]
                         << ") -> dx_grid/dx_file bounded by ["
                         << dmin / file_dmax[t] << ", " << dmax / file_dmin[t]
                         << "]\n";
        }
      }
    }
  }
#endif
}

void
PeleLM::readIOParameters()
{
  BL_PROFILE_VAR("PeleLMeX::readIOParameters()", readIOParameters);

  {
    amrex::ParmParse pp("peleLM");
    pp.query("base_output_prefix", m_base_output_pref);
  }

  amrex::ParmParse pp("amr");

  pp.query("check_file", m_check_file);
  m_check_file = m_base_output_pref + m_check_file;
  pp.query("check_int", m_check_int);
  pp.query("check_overwrite", m_check_overwrite);
  pp.query("check_per", m_check_per);
  pp.query("restart", m_restart_chkfile);
  pp.query("initDataPlt", m_restart_pltfile);
  pp.query("initDataPltSource", pltfileSource);
  pp.query("plot_file", m_plot_file);
  m_plot_file = m_base_output_pref + m_plot_file;
  pp.query("plot_int", m_plot_int);
  pp.query("plot_overwrite", m_plot_overwrite);
  pp.query("plot_init_state", m_plot_init_state);
  if (pp.contains("plot_per")) {
    int do_exact = 0;
    pp.query("plot_per_exact", do_exact);
    if (do_exact != 0) {
      pp.query("plot_per", m_plot_per_exact);
    } else {
      pp.query("plot_per", m_plot_per_approx);
    }
  }
  pp.query("plot_grad_p", m_plot_grad_p);
  m_derivePlotVarCount = (pp.countval("derive_plot_vars"));
  if (m_derivePlotVarCount != 0) {
    m_derivePlotVars.resize(m_derivePlotVarCount);
    for (int ivar = 0; ivar < m_derivePlotVarCount; ++ivar) {
      pp.get("derive_plot_vars", m_derivePlotVars[ivar], ivar);
    }
  }
  pp.query("plot_zeroEBcovered", m_plot_zeroEBcovered);
  pp.query("plot_speciesState", m_plotStateSpec);
  m_initial_grid_file = "";
  m_regrid_file = "";
  pp.query("initial_grid_file", m_initial_grid_file);
  pp.query("regrid_file", m_regrid_file);
  pp.query("file_stepDigits", m_ioDigits);
  pp.query("regrid_interp_method", m_regrid_interp_method);
#ifdef AMREX_USE_HDF5
  pp.query("use_hdf5_plt", m_write_hdf5_pltfile);
#endif
  AMREX_ASSERT(m_regrid_interp_method == 0 || m_regrid_interp_method == 1);
}

void
PeleLM::variablesSetup()
{
  BL_PROFILE("PeleLMeX::variablesSetup()");

  //----------------------------------------------------------------
  // Variables ordering is defined through compiler macro in PeleLMeX_Index.H
  // Simply print on screen the state layout and append to the stateComponents
  // list
  amrex::Print() << "\n";
  amrex::Print() << PrettyLine;
  amrex::Print() << " State components \n";
  amrex::Print() << PrettyLine;

  amrex::Print() << " Velocity X: " << VELX;
  stateComponents.emplace_back(VELX, "x_velocity");
#if AMREX_SPACEDIM > 1
  amrex::Print() << ", Velocity Y: " << VELY;
  stateComponents.emplace_back(VELY, "y_velocity");
#if AMREX_SPACEDIM > 2
  amrex::Print() << ", Velocity Z: " << VELZ;
  stateComponents.emplace_back(VELZ, "z_velocity");
#endif
#endif
  amrex::Print() << " \n";

  if (m_incompressible == 0) {
    amrex::Print() << " Density: " << DENSITY << "\n";
    stateComponents.emplace_back(DENSITY, "density");
    amrex::Print() << " First species: " << FIRSTSPEC << "\n";
    amrex::Vector<std::string> names;
    pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
      names, &(eos_parms.host_parm()));
    for (int n = 0; n < NUM_SPECIES; ++n) {
      stateComponents.emplace_back(FIRSTSPEC + n, "rho.Y(" + names[n] + ")");
      reactComponents.emplace_back(n, "I_R(" + names[n] + ")");
    }
    amrex::Print() << " Enthalpy: " << RHOH << "\n";
    stateComponents.emplace_back(RHOH, "rhoh");
    amrex::Print() << " Temperature: " << TEMP << "\n";
    stateComponents.emplace_back(TEMP, "temp");
    amrex::Print() << " thermo. pressure: " << RHORT << "\n";
    stateComponents.emplace_back(RHORT, "RhoRT");
#ifdef PELE_USE_PLASMA
    amrex::Print() << " nE: " << NE << "\n";
    stateComponents.emplace_back(NE, "nE");
    amrex::Print() << " PhiV: " << PHIV << "\n";
    stateComponents.emplace_back(PHIV, "PhiV");
#endif
#ifdef PELE_USE_SOOT
    for (int mom = 0; mom < NUMSOOTVAR; ++mom) {
      std::string sootname = soot_model->sootVariableName(mom);
      amrex::Print() << " " << sootname << ": " << FIRSTSOOT + mom << "\n";
      stateComponents.emplace_back(FIRSTSOOT + mom, sootname);
    }
    setSootIndx();
#endif
#if NUM_ODE > 0
    amrex::Print() << " First ODE: " << FIRSTODE << "\n";
    ProblemSpecificFunctions::set_ode_names(m_ode_names);
    if (m_ode_names.size() != NUM_ODE) {
      amrex::Abort(
        "ODEQty names improperly set. Adjust set_ode_names in "
        "ProblemSpecificFunctions or NUM_ODE in GNUMakefile");
    }
    for (int n = 0; n < NUM_ODE; ++n) {
      if (m_ode_names[n].empty()) {
        amrex::Abort(
          "ODEQty names improperly set. Adjust set_ode_names in "
          "ProblemSpecificFunctions or NUM_ODE in GNUMakefile");
      }
      stateComponents.emplace_back(FIRSTODE + n, m_ode_names[n]);
    }
#endif
  }

  if (m_incompressible != 0) {
    amrex::Print() << " => Total number of state variables: " << AMREX_SPACEDIM
                   << "\n";
  } else {
    amrex::Print() << " => Total number of state variables: " << NVAR << "\n";
  }
  if (m_nAux > 0) {
    for (int n = 0; n < m_nAux; ++n) {
      amrex::Print() << " Auxiliary " + std::to_string(n + 1) + ": "
                     << m_aux_names[n] << "\n";
      amrex::Print() << "   Advective: " << m_aux_advect[n] << "\n";
      amrex::Print() << "   Conservative: " << m_AdvTypeAux[n] << "\n";
      amrex::Print() << "   Diffusive: " << m_DiffTypeAux[n];
      if (m_aux_Schmidt[n] > 0) {
        amrex::Print() << " - Schmidt number: " << m_aux_Schmidt[n];
      }
      amrex::Print() << "\n";
    }
    amrex::Print() << " => Total number of auxiliary variables: " << m_nAux
                   << "\n";
  }
  amrex::Print() << PrettyLine;
  amrex::Print() << "\n";

  //----------------------------------------------------------------
  // Set advection/diffusion types
  if (m_incompressible != 0) {
    m_AdvTypeState.resize(AMREX_SPACEDIM);
    m_DiffTypeState.resize(AMREX_SPACEDIM);
  } else {
    m_AdvTypeState.resize(NVAR);
    m_DiffTypeState.resize(NVAR);
  }

  // Velocity - follow incflo
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    m_AdvTypeState[VELX + idim] = 0;  // NonConservative
    m_DiffTypeState[VELX + idim] = 1; // Diffusive
  }

  if (m_incompressible == 0) {
    m_AdvTypeState[DENSITY] = 1;
    m_DiffTypeState[DENSITY] = 0;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      m_AdvTypeState[FIRSTSPEC + n] = 1;
      m_DiffTypeState[FIRSTSPEC + n] = 1;
    }
    m_AdvTypeState[RHOH] = 1;
    m_DiffTypeState[RHOH] = 1;
    m_AdvTypeState[TEMP] = 0;
    m_DiffTypeState[TEMP] = 0;
    m_AdvTypeState[RHORT] = 0;
    m_DiffTypeState[RHORT] = 0;
#ifdef PELE_USE_PLASMA
    m_AdvTypeState[NE] = 0;
    m_DiffTypeState[NE] = 0;
    m_AdvTypeState[PHIV] = 0;
    m_DiffTypeState[PHIV] = 0;
#endif
#ifdef PELE_USE_SOOT
    for (int mom = 0; mom < NUMSOOTVAR; ++mom) {
      m_AdvTypeState[FIRSTSOOT + mom] = 0;
      m_DiffTypeState[FIRSTSOOT + mom] = 0;
    }
#endif
  }

  //----------------------------------------------------------------
  // Typical values container
  if (m_incompressible != 0) {
    typical_values.resize(AMREX_SPACEDIM, -1.0);
  } else {
    typical_values.resize(NVAR, -1.0);
  }

  if (m_incompressible == 0) {
    // -----------------------------------------
    // Combustion
    // -----------------------------------------
    amrex::ParmParse pp("peleLM");
    std::string fuel_name;
    pp.query("fuel_name", fuel_name);
    fuel_name = "rho.Y(" + fuel_name + ")";
    if (isStateVariable(fuel_name)) {
      fuelID = stateVariableIndex(fuel_name);
      fuelID -= FIRSTSPEC;
    }
  }

  if (max_level > 0 && !m_initial_grid_file.empty()) {
    readGridFile(m_initial_grid_file, m_initial_ba);
    if (verbose > 0) {
      amrex::Print() << "Read initial_ba. Size is " << m_initial_ba.size()
                     << "\n";
    }
  }
  if (max_level > 0 && !m_regrid_file.empty() && m_regrid_int > 0) {
    readGridFile(m_regrid_file, m_regrid_ba);
    if (verbose > 0) {
      amrex::Print() << "Read regrid_ba. Size is " << m_regrid_ba.size()
                     << "\n";
    }
  }
}

void
PeleLM::readGridFile(
  const std::string& grid_file, amrex::Vector<amrex::BoxArray>& input_ba)
{
#define STRIP                \
  while (is.get() != '\n') { \
  }
  std::ifstream is(grid_file.c_str(), std::ios::in);
  if (!is.good()) {
    amrex::FileOpenFailed(grid_file);
  }
  int in_finest, ngrid;

  is >> in_finest;
  STRIP;
  input_ba.resize(in_finest);
  use_fixed_upto_level = in_finest;
  if (in_finest > max_level) {
    amrex::Error(
      "You have fewer levels in your inputs file then in your grids file!");
  }

  for (int lev = 1; lev <= in_finest; ++lev) {
    amrex::BoxList bl;
    is >> ngrid;
    STRIP;
    for (int i = 0; i < ngrid; ++i) {
      amrex::Box bx;
      is >> bx;
      STRIP;
      bx.refine(ref_ratio[lev - 1]);
      bl.push_back(bx);
    }
    input_ba[lev - 1].define(bl);
  }
  is.close();

#undef STRIP
}

void
PeleLM::derivedSetup()
{
  BL_PROFILE("PeleLMeX::derivedSetup()");

  if (m_incompressible == 0) {

    // Get species names
    amrex::Vector<std::string> spec_names;
    pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
      spec_names, &(eos_parms.host_parm()));

    // Set species mass fractions
    amrex::Vector<std::string> var_names_massfrac(NUM_SPECIES);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names_massfrac[n] = "Y(" + spec_names[n] + ")";
    }
    derive_lst.add(
      "mass_fractions", amrex::IndexType::TheCellType(), NUM_SPECIES,
      var_names_massfrac, pelelmex_dermassfrac, the_same_box);

    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names_massfrac[n] = "X(" + spec_names[n] + ")";
    }
    derive_lst.add(
      "mole_fractions", amrex::IndexType::TheCellType(), NUM_SPECIES,
      var_names_massfrac, pelelmex_dermolefrac, the_same_box);

    // Element mass fractions - for now only FUEGO/SRK
    if (
      pele::physics::PhysicsType::eos_type::identifier() == "Fuego" ||
      pele::physics::PhysicsType::eos_type::identifier() == "SRK") {
      amrex::Vector<std::string> ename;
      CKSYME_STR(ename);
      amrex::Vector<std::string> var_names_elemfrac(NUM_ELEMENTS);
      for (int n = 0; n < NUM_ELEMENTS; ++n) {
        var_names_elemfrac[n] = "Z(" + ename[n] + ")";
      }
      derive_lst.add(
        "element_fractions", amrex::IndexType::TheCellType(), NUM_ELEMENTS,
        var_names_elemfrac, pelelmex_derelementfrac, the_same_box);
    }

    // Species diffusion coefficients
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names_massfrac[n] = "D_" + spec_names[n];
    }
    if (m_use_soret != 0) {
      var_names_massfrac.resize(2 * NUM_SPECIES);
      for (int n = 0; n < NUM_SPECIES; ++n) {
        var_names_massfrac[n + NUM_SPECIES] = "theta_" + spec_names[n];
      }
      derive_lst.add(
        "diffcoeff", amrex::IndexType::TheCellType(), 2 * NUM_SPECIES,
        var_names_massfrac, pelelmex_derdiffc, the_same_box);
    } else {
      derive_lst.add(
        "diffcoeff", amrex::IndexType::TheCellType(), NUM_SPECIES,
        var_names_massfrac, pelelmex_derdiffc, the_same_box);
    }

    // Rho - sum rhoYs
    derive_lst.add(
      "rhominsumrhoY", amrex::IndexType::TheCellType(), 1, pelelmex_derrhomrhoy,
      the_same_box);

    // Heat Release
    derive_lst.add(
      "HeatRelease", amrex::IndexType::TheCellType(), 1,
      pelelmex_derheatrelease, the_same_box);

    // Thermal diffusivity
    derive_lst.add(
      "lambda", amrex::IndexType::TheCellType(), 1, pelelmex_derlambda,
      the_same_box);

    // Mixture fraction
    derive_lst.add(
      "mixture_fraction", amrex::IndexType::TheCellType(), 1,
      pelelmex_dermixfrac, the_same_box);

    // Progress variable
    derive_lst.add(
      "progress_variable", amrex::IndexType::TheCellType(), 1,
      pelelmex_derprogvar, the_same_box);
  }

  // Distribution Map
  derive_lst.add(
    "DistributionMap", amrex::IndexType::TheCellType(), 1, pelelmex_derdmap,
    the_same_box);

  // Turbulent Forcing Terms
  amrex::Vector<std::string> var_names_turbforcing = {
    AMREX_D_DECL("forcex", "forcey", "forcez")};
  derive_lst.add(
    "turbforces", amrex::IndexType::TheCellType(), AMREX_SPACEDIM,
    var_names_turbforcing, pelelmex_derturbforcing, the_same_box);

  // Cell average pressure
  derive_lst.add(
    "avg_pressure", amrex::IndexType::TheCellType(), 1, pelelmex_deravgpress,
    the_same_box);

  // Viscosity
  derive_lst.add(
    "viscosity", amrex::IndexType::TheCellType(), 1, pelelmex_dervisc,
    the_same_box);

  // Velocity magnitude
  derive_lst.add(
    "mag_vel", amrex::IndexType::TheCellType(), 1, pelelmex_dermgvel,
    the_same_box);

  // Vorticity magnitude
  derive_lst.add(
    "mag_vort", amrex::IndexType::TheCellType(), 1, pelelmex_dermgvort,
    grow_box_by_two);

  // Spatial coordinates
  {
    amrex::Vector<std::string> var_names({AMREX_D_DECL("x", "y", "z")});
    derive_lst.add(
      "coordinates", amrex::IndexType::TheCellType(), AMREX_SPACEDIM, var_names,
      pelelmex_dercoord, the_same_box);
  }

  // Vorticity components
  {
    const int vort_ncomp = 2 * AMREX_SPACEDIM - 3;
#if (AMREX_SPACEDIM == 2)
    amrex::Vector<std::string> var_names({"VortZ"});
#elif (AMREX_SPACEDIM == 3)
    amrex::Vector<std::string> var_names({"VortX", "VortY", "VortZ"});
#endif
    derive_lst.add(
      "vorticity", amrex::IndexType::TheCellType(), vort_ncomp, var_names,
      pelelmex_dervort, grow_box_by_two);
  }

  // UserDefined derived
  {
    amrex::Vector<std::string> var_names = pelelmex_setuserderives();
    derive_lst.add(
      "derUserDefined", amrex::IndexType::TheCellType(),
      static_cast<int>(var_names.size()), var_names, pelelmex_deruserdef,
      the_same_box);
  }

#if (AMREX_SPACEDIM == 3)
  // Q-criterion
  derive_lst.add(
    "Qcrit", amrex::IndexType::TheCellType(), 1, pelelmex_derQcrit,
    grow_box_by_two);
#endif

  // Kinetic energy
  derive_lst.add(
    "kinetic_energy", amrex::IndexType::TheCellType(), 1,
    pelelmex_derkineticenergy, the_same_box);

  // Enstrophy
  derive_lst.add(
    "enstrophy", amrex::IndexType::TheCellType(), 1, pelelmex_derenstrophy,
    grow_box_by_two);

#ifdef USE_MANIFOLD_EOS
  auto& mani_data = eos_parms.host_only_parm().manfunc_par->host_parm();
  const int nmanivar = mani_data.Nvar;
  amrex::Vector<std::string> var_names_maniout(nmanivar);
  for (int n = 0; n < nmanivar; ++n) {
    std::string nametmp = std::string(
      &(mani_data.varnames)[n * mani_data.len_str], mani_data.len_str);
    var_names_maniout[n] = "MANI_" + amrex::trim(nametmp);
  }
  derive_lst.add(
    "maniout", amrex::IndexType::TheCellType(), nmanivar, var_names_maniout,
    pelelmex_dermaniout, the_same_box);
#endif

#ifdef PELE_USE_PLASMA
  // Charge distribution
  derive_lst.add(
    "chargedistrib", amrex::IndexType::TheCellType(), 1, pelelmex_derchargedist,
    the_same_box);

  // Electric field
  derive_lst.add(
    "efieldx", amrex::IndexType::TheCellType(), 1, pelelmex_derefx,
    grow_box_by_one);
#if (AMREX_SPACEDIM > 1)
  derive_lst.add(
    "efieldy", amrex::IndexType::TheCellType(), 1, pelelmex_derefy,
    grow_box_by_one);
#if (AMREX_SPACEDIM > 2)
  derive_lst.add(
    "efieldz", amrex::IndexType::TheCellType(), 1, pelelmex_derefz,
    grow_box_by_one);
#endif
#endif
  // Lorentz forces
  derive_lst.add(
    "LorentzFx", amrex::IndexType::TheCellType(), 1, pelelmex_derLorentzx,
    grow_box_by_one);
#if (AMREX_SPACEDIM > 1)
  derive_lst.add(
    "LorentzFy", amrex::IndexType::TheCellType(), 1, pelelmex_derLorentzy,
    grow_box_by_one);
#if (AMREX_SPACEDIM > 2)
  derive_lst.add(
    "LorentzFz", amrex::IndexType::TheCellType(), 1, pelelmex_derLorentzz,
    grow_box_by_one);
#endif
#endif
#endif
  auto it = m_derivePlotVars.begin();
  while (it != m_derivePlotVars.end()) {
    if (!derive_lst.canDerive(*it)) {
      it = m_derivePlotVars.erase(it);
    } else {
      it++;
    }
  }
  m_derivePlotVarCount = static_cast<int>(m_derivePlotVars.size());
}

void
PeleLM::evaluateSetup()
{
  BL_PROFILE("PeleLMeX::evaluateSetup()");

  // Get species names
  amrex::Vector<std::string> spec_names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(
    spec_names, &(eos_parms.host_parm()));

  // divU
  evaluate_lst.add("divU", amrex::IndexType::TheCellType(), 1, the_same_box);

  // projected velocity field
  {
    amrex::Vector<std::string> var_names = {
      AMREX_D_DECL("x_velProj", "y_velProj", "z_velProj")};
    evaluate_lst.add(
      "velProj", amrex::IndexType::TheCellType(), AMREX_SPACEDIM, var_names,
      the_same_box);
  }

  // velocity force
  {
    amrex::Vector<std::string> var_names = {
      AMREX_D_DECL("x_velForce", "y_velForce", "z_velForce")};
    evaluate_lst.add(
      "velForce", amrex::IndexType::TheCellType(), AMREX_SPACEDIM, var_names,
      the_same_box);
  }

  // divTau
  {
    amrex::Vector<std::string> var_names = {
      AMREX_D_DECL("x_divTau", "y_divTau", "z_divTau")};
    evaluate_lst.add(
      "divTau", amrex::IndexType::TheCellType(), AMREX_SPACEDIM, var_names,
      the_same_box);
  }

  // scalar diffusion term
  {
    amrex::Vector<std::string> var_names(NUM_SPECIES + 2);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[n] = "D(" + spec_names[n] + ")";
    }
    var_names[NUM_SPECIES] = "D(RhoH)";
    var_names[NUM_SPECIES + 1] = "D(Temp)";
    evaluate_lst.add(
      "diffTerm", amrex::IndexType::TheCellType(), NUM_SPECIES + 2, var_names,
      the_same_box);
  }

  // advection terms
  {
    amrex::Vector<std::string> var_names(
      NVAR - 2); // Skip temperature and RhoRT, unused
    AMREX_D_TERM(
      var_names[VELX] = "A(VELX)";, var_names[VELY] = "A(VELY)";
      , var_names[VELZ] = "A(VELZ)");
    var_names[DENSITY] = "A(Rho)";
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[FIRSTSPEC + n] = "A(" + spec_names[n] + ")";
    }
    var_names[RHOH] = "A(RhoH)";
    evaluate_lst.add(
      "advTerm", amrex::IndexType::TheCellType(), NVAR - 2, var_names,
      the_same_box);
  }

  // Chemical state and external chem. forcing (used in ReactEval)
  {
    amrex::Vector<std::string> var_names(2 * (NUM_SPECIES + 1) + 1);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[n] = "rhoY(" + spec_names[n] + ")";
    }
    var_names[NUM_SPECIES] = "rhoH";
    var_names[NUM_SPECIES + 1] = "Temp";
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[NUM_SPECIES + 2 + n] = "F_rhoY(" + spec_names[n] + ")";
    }
    var_names[2 * NUM_SPECIES + 2] = "F_rhoH";
    evaluate_lst.add(
      "chemTest", amrex::IndexType::TheCellType(), 2 * (NUM_SPECIES + 1) + 1,
      var_names, the_same_box);
  }

  // instantaneous reaction rate
  {
    amrex::Vector<std::string> var_names(NUM_SPECIES);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[n] = "I_R(" + spec_names[n] + ")";
    }
    evaluate_lst.add(
      "instRR", amrex::IndexType::TheCellType(), NUM_SPECIES, var_names,
      the_same_box);
  }

  // cell-centered transport coefficients
  {
    amrex::Vector<std::string> var_names(NUM_SPECIES + 2);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      var_names[n] = "rhoD(" + spec_names[n] + ")";
    }
    var_names[NUM_SPECIES] = "Lamdba";
    var_names[NUM_SPECIES + 1] = "Mu";
    evaluate_lst.add(
      "transportCC", amrex::IndexType::TheCellType(), NUM_SPECIES + 2,
      var_names, the_same_box);
  }
}

void
PeleLM::taggingSetup()
{
  BL_PROFILE("PeleLMeX::taggingSetup()");

  std::string amr_prefix = "amr";
  amrex::ParmParse ppamr(amr_prefix);

  amrex::Vector<std::string> refinement_indicators;
  ppamr.queryarr(
    "refinement_indicators", refinement_indicators, 0,
    ppamr.countval("refinement_indicators"));
  for (const auto& refinement_indicator : refinement_indicators) {
    std::string ref_prefix = amr_prefix;
    ref_prefix += ".";
    ref_prefix += refinement_indicator;
    amrex::ParmParse ppr(ref_prefix);

    // Tag a given box
    amrex::RealBox realbox;
    if (ppr.countval("in_box_lo") != 0) {
      amrex::Vector<amrex::Real> box_lo(AMREX_SPACEDIM);
      amrex::Vector<amrex::Real> box_hi(AMREX_SPACEDIM);
      ppr.getarr("in_box_lo", box_lo, 0, static_cast<int>(box_lo.size()));
      ppr.getarr("in_box_hi", box_hi, 0, static_cast<int>(box_hi.size()));
      realbox = amrex::RealBox(box_lo.data(), box_hi.data());
    }

    amrex::AMRErrorTagInfo info;

    if (realbox.ok()) {
      info.SetRealBox(realbox);
    }

    if (ppr.countval("start_time") > 0) {
      amrex::Real min_time;
      ppr.get("start_time", min_time);
      info.SetMinTime(min_time);
    }

    if (ppr.countval("end_time") > 0) {
      amrex::Real max_time;
      ppr.get("end_time", max_time);
      info.SetMaxTime(max_time);
    }

    if (ppr.countval("max_level") > 0) {
      int tag_max_level;
      ppr.get("max_level", tag_max_level);
      info.SetMaxLevel(tag_max_level);
    }

    bool itexists = false;
    if (ppr.countval("value_greater") != 0) {
      amrex::Real value;
      ppr.get("value_greater", value);
      std::string field;
      ppr.get("field_name", field);
      errTags.push_back(
        amrex::AMRErrorTag(value, amrex::AMRErrorTag::GREATER, field, info));
      itexists = derive_lst.canDerive(field) || isStateVariable(field) ||
                 isReactVariable(field);
    } else if (ppr.countval("value_less") != 0) {
      amrex::Real value;
      ppr.get("value_less", value);
      std::string field;
      ppr.get("field_name", field);
      errTags.push_back(
        amrex::AMRErrorTag(value, amrex::AMRErrorTag::LESS, field, info));
      itexists = derive_lst.canDerive(field) || isStateVariable(field) ||
                 isReactVariable(field);
    } else if (ppr.countval("vorticity_greater") != 0) {
      amrex::Real value;
      ppr.get("vorticity_greater", value);
      const std::string field = "mag_vort";
      errTags.push_back(
        amrex::AMRErrorTag(value, amrex::AMRErrorTag::VORT, field, info));
      itexists = derive_lst.canDerive(field) || isStateVariable(field) ||
                 isReactVariable(field);
    } else if (ppr.countval("adjacent_difference_greater") != 0) {
      amrex::Real value;
      ppr.get("adjacent_difference_greater", value);
      std::string field;
      ppr.get("field_name", field);
      errTags.push_back(
        amrex::AMRErrorTag(value, amrex::AMRErrorTag::GRAD, field, info));
      itexists = derive_lst.canDerive(field) || isStateVariable(field) ||
                 isReactVariable(field);
    } else if (realbox.ok()) {
      errTags.push_back(amrex::AMRErrorTag(info));
      itexists = true;
    } else {
      amrex::Abort(
        std::string(
          "Unrecognized refinement indicator for " + refinement_indicator)
          .c_str());
    }

    if (!itexists) {
      amrex::Error(
        "PeleLM::taggingSetup(): unknown variable field for criteria " +
        refinement_indicator);
    }
  }
}

void
PeleLM::resizeArray()
{

  if (m_verbose != 0) {
    amrex::Print() << " Initializing data for " << max_level + 1
                   << " levels \n";
  }

  // State data
  m_leveldata_old.resize(max_level + 1);
  m_leveldata_new.resize(max_level + 1);
  m_leveldatareact.resize(max_level + 1);
  m_halfTimeDensity.resize(max_level + 1);
  if (max_level > 0) {
    m_coveredMask.resize(max_level);
  }
  m_baChem.resize(max_level + 1);
  m_dmapChem.resize(max_level + 1);
  m_baChemFlag.resize(max_level + 1);

#ifdef PELE_USE_PLASMA
  m_leveldatanlsolve.resize(max_level + 1);
  m_ionsFluxes.resize(max_level + 1);
#endif

  // External sources
  m_extSource.resize(max_level + 1);

  // Factory
  m_factory.resize(max_level + 1);

  // Time
  m_t_old.resize(max_level + 1);
  m_t_new.resize(max_level + 1);

#ifdef PELE_USE_SPRAY
  m_spraystate.resize(max_level + 1);
  m_spraysource.resize(max_level + 1);
#endif

  // Load balancing
  m_costs.resize(max_level + 1);
  m_loadBalanceEff.resize(max_level + 1);

  // Mesh mapping metric fields (if enabled; otherwise no-op)
  if (m_mesh_map) {
    m_mesh_map->resize(max_level + 1);
  }
}
