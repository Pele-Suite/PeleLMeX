#ifndef BPATCH_H
#define BPATCH_H

#include <PeleLMeX.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

class BPatch
{

public:
  BPatch() = default;
  ~BPatch() { deallocate(); }

  BPatch(
    const std::string& patch_name,
    const amrex::Geometry& geom,
    pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
      eosparms_h,
    const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
      eosparms_d);

  struct BpatchDataContainer
  {
    int m_boundary_dir;
    int m_boundary_lo_hi;
    int num_species = 0;
    int num_groups = 0;
    int m_patchtype_num;
    int* speciesIndex = nullptr;
    amrex::Real* speciesFlux = nullptr;

    // In 2D, a boundary patch is a line
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_line_center;
    amrex::Real m_patch_line_radius;

    // circle variables
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_circle_center;
    amrex::Real m_patch_circle_radius;

    // rectangle variables
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rectangle_lo;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rectangle_hi;

    // circle annular variables
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_circ_ann_center;
    amrex::Real m_patch_circ_ann_inner_radius;
    amrex::Real m_patch_circ_ann_outer_radius;

    // Rectangular annular variables
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_outer_lo;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_outer_hi;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_inner_lo;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_patch_rect_ann_inner_hi;

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    bool CheckifPointInside(
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> point_coordinate,
      amrex::Real dx) const
    {
      bool inside = false;

      // auto dx = geom.CellSizeArray();
      amrex::Real xp, yp;

      point_coordinate[m_boundary_dir] = 0.0;
#if (AMREX_SPACEDIM == 2)
      xp = point_coordinate[0];
      yp = point_coordinate[1];
      if (m_patchtype_num == 0) {
        inside = true;
      } else if (m_patchtype_num == 1) {
        amrex::Real patch_line_radius_touse = m_patch_circle_radius + dx / 2.0;
        amrex::Real rad = sqrt(
          (xp - m_patch_circle_center[0]) * (xp - m_patch_circle_center[0]) +
          (yp - m_patch_circle_center[1]) * (yp - m_patch_circle_center[1]));
        if (rad <= patch_line_radius_touse) {
          inside = true;
        }
      }
#elif (AMREX_SPACEDIM == 3)
      amrex::Real zp;
      constexpr amrex::Real sqrt2 = 1.4142135623730951;

      xp = point_coordinate[0];
      yp = point_coordinate[1];
      zp = point_coordinate[2];

      if (m_patchtype_num == 0) {
        inside = true;
      } else if (m_patchtype_num == 1) {

        const amrex::Real patch_circle_radius_touse =
          m_patch_circle_radius + dx * sqrt2 / 2.0;
        const amrex::Real rad = sqrt(
          (xp - m_patch_circle_center[0]) * (xp - m_patch_circle_center[0]) +
          (yp - m_patch_circle_center[1]) * (yp - m_patch_circle_center[1]) +
          (zp - m_patch_circle_center[2]) * (zp - m_patch_circle_center[2]));
        if (rad <= patch_circle_radius_touse) {
          inside = true;
        }

      } else if (m_patchtype_num == 2) {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> patch_rectangle_lo_touse =
          {AMREX_D_DECL(
            m_patch_rectangle_lo[0] - dx / 2.0_rt,
            m_patch_rectangle_lo[1] - dx / 2.0_rt,
            m_patch_rectangle_lo[2] - dx / 2.0_rt)};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> patch_rectangle_hi_touse =
          {AMREX_D_DECL(
            m_patch_rectangle_hi[0] + dx / 2.0_rt,
            m_patch_rectangle_hi[1] + dx / 2.0_rt,
            m_patch_rectangle_hi[2] + dx / 2.0_rt)};
        patch_rectangle_hi_touse[m_boundary_dir] = 0.0;
        patch_rectangle_lo_touse[m_boundary_dir] = 0.0;

        if (
          (xp >= patch_rectangle_lo_touse[0] &&
           xp <= patch_rectangle_hi_touse[0]) &&
          (yp >= patch_rectangle_lo_touse[1] &&
           yp <= patch_rectangle_hi_touse[1]) &&
          (zp >= patch_rectangle_lo_touse[2] &&
           zp <= patch_rectangle_hi_touse[2])) {
          inside = true;
        }
      } else if (m_patchtype_num == 3) {
        const amrex::Real patch_circ_ann_outer_radius_touse =
          m_patch_circ_ann_outer_radius + dx * sqrt2 / 2.0;
        const amrex::Real patch_circ_ann_inner_radius_touse =
          m_patch_circ_ann_inner_radius - dx * sqrt2 / 2.0;

        const amrex::Real rad = sqrt(
          (xp - m_patch_circ_ann_center[0]) *
            (xp - m_patch_circ_ann_center[0]) +
          (yp - m_patch_circ_ann_center[1]) *
            (yp - m_patch_circ_ann_center[1]) +
          (zp - m_patch_circ_ann_center[2]) *
            (zp - m_patch_circ_ann_center[2]));
        if (
          rad <= patch_circ_ann_outer_radius_touse &&
          rad >= patch_circ_ann_inner_radius_touse) {
          inside = true;
        }
      } else if (m_patchtype_num == 4) {
        bool inside_outer = false;
        bool inside_inner = false;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
          patch_rect_ann_inner_lo_touse = {AMREX_D_DECL(
            m_patch_rect_ann_inner_lo[0] + dx / 2.0_rt,
            m_patch_rect_ann_inner_lo[1] + dx / 2.0_rt,
            m_patch_rect_ann_inner_lo[2] + dx / 2.0_rt)};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
          patch_rect_ann_inner_hi_touse = {AMREX_D_DECL(
            m_patch_rect_ann_inner_hi[0] - dx / 2.0_rt,
            m_patch_rect_ann_inner_hi[1] - dx / 2.0_rt,
            m_patch_rect_ann_inner_hi[2] - dx / 2.0_rt)};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
          patch_rect_ann_outer_lo_touse = {AMREX_D_DECL(
            m_patch_rect_ann_outer_lo[0] - dx / 2.0_rt,
            m_patch_rect_ann_outer_lo[1] - dx / 2.0_rt,
            m_patch_rect_ann_outer_lo[2] - dx / 2.0_rt)};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
          patch_rect_ann_outer_hi_touse = {AMREX_D_DECL(
            m_patch_rect_ann_outer_hi[0] + dx / 2.0_rt,
            m_patch_rect_ann_outer_hi[1] + dx / 2.0_rt,
            m_patch_rect_ann_outer_hi[2] + dx / 2.0_rt)};
        patch_rect_ann_inner_lo_touse[m_boundary_dir] = 0.0;
        patch_rect_ann_inner_hi_touse[m_boundary_dir] = 0.0;
        patch_rect_ann_outer_lo_touse[m_boundary_dir] = 0.0;
        patch_rect_ann_outer_hi_touse[m_boundary_dir] = 0.0;

        if (((xp >= patch_rect_ann_outer_lo_touse[0] &&
              xp <= patch_rect_ann_outer_hi_touse[0]) &&
             (yp >= patch_rect_ann_outer_lo_touse[1] &&
              yp <= patch_rect_ann_outer_hi_touse[1]) &&
             (zp >= patch_rect_ann_outer_lo_touse[2] &&
              zp <= patch_rect_ann_outer_hi_touse[2]))) {
          inside_outer = true;
        }
        if (((xp >= patch_rect_ann_inner_lo_touse[0] &&
              xp <= patch_rect_ann_inner_hi_touse[0]) &&
             (yp >= patch_rect_ann_inner_lo_touse[1] &&
              yp <= patch_rect_ann_inner_hi_touse[1]) &&
             (zp >= patch_rect_ann_inner_lo_touse[2] &&
              zp <= patch_rect_ann_inner_hi_touse[2]))) {
          inside_inner = true;
        }
        if (inside_outer && !inside_inner) {
          inside = true;
        }
      }
#endif

      return inside;
    }
  };

  void allocate()
  {
    if (!m_device_allocated) {
      m_bpdata_d =
        (BpatchDataContainer*)amrex::The_Arena()->alloc(sizeof(m_bpdata_h));
      m_device_allocated = true;
      sync_to_device();
    }
  }

  void deallocate()
  {
    if (m_host_allocated) {
      amrex::The_Pinned_Arena()->free(m_bpdata_h.speciesIndex);
      amrex::The_Pinned_Arena()->free(m_bpdata_h.speciesFlux);
    }
    if (m_device_allocated) {
      amrex::The_Arena()->free(m_bpdata_d);
    }
  }

  void sync_to_device()
  {
    if (!m_device_allocated) {
      amrex::Abort("Device params not allocated yet");
    } else {
      amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, &m_bpdata_h, &m_bpdata_h + 1, m_bpdata_d);
    }
  }

  // Accessors
  BpatchDataContainer getHostData() { return m_bpdata_h; }
  BpatchDataContainer* getHostDataPtr() { return &m_bpdata_h; }
  BpatchDataContainer* getDeviceData() { return m_bpdata_d; }

  std::string m_patchname;
  std::string
    m_patchtype; // values:
                 // "fullboundary","circle","rectangle","annular-circle","annular-rectangle"
  amrex::Vector<std::string> speciesList;
  amrex::Vector<std::string> groupNames;
  amrex::Vector<amrex::Vector<int>> speciesinGroup;

protected:
  BpatchDataContainer m_bpdata_h;
  BpatchDataContainer* m_bpdata_d = nullptr;
  bool m_device_allocated{false};
  bool m_host_allocated{false};
};

#endif
