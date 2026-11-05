# ========================================================================
# Conservation check for the reacting mesh-mapping + AMR regression
# (flamesheet-drm19-mapped-amr-3d and its identity-map control
# flamesheet-drm19-amr-3d).
#
# Open domain (inflow/outflow), so total mass is NOT constant; the right
# statement is a FLUX balance:  d(total mass)/dt == net boundary flux.
# PeleLM writes temporals/tempMass with columns:
#     iter, time, massNew, dmdt, netMassFlux, balance
# where dmdt = dM/dt, netMassFlux = sum of conservative advective fluxes
# over the domain faces (PeleLM::addMassFluxes), and balance = |dmdt -
# netMassFlux|.
#
# WHY WE DO NOT TEST balance / flux_scale DIRECTLY:
#   balance is an instantaneous rate residual.  dmdt = (M^{n+1}-M^n)/dt,
#   so balance ~ (round-off of the M reduction)/dt.  With dt_shrink the
#   startup dt is ~1e-12 s, which amplifies the ~eps*M reduction round-off
#   into a balance/flux of ~1e-8 even though mass is conserved to machine
#   precision.  That metric is dt-fragile and misleading.
#
# WHAT WE TEST (dt-robust, mass-relative):
#   (1) worst per-step mass imbalance  |dM - netMassFlux*dt| / M
#       ( = balance*dt / M ).  This removes the 1/dt amplification and is
#       the per-step conservation error in mass units.  At round-off it is
#       O(machine eps); a genuine metric/conservation bug makes it O(1e-6)
#       or larger and/or growing at regrid steps.
#   (2) signed cumulative drift  |(M_end - M_0) - sum(netMassFlux*dt)| / M
#       -- the actual accumulated imbalance over the run.
#
# MESH MAPPING: the mapped FV update conserves rho*detJ and telescopes to
# the Xi-space boundary flux that addMassFluxes already sums, so only the
# volume integral needs the metric: PeleLM::massBalance integrates rho
# with the detJ weight (MFSum, cached physical cell volume); the boundary-flux sum stays in Xi
# space (must NOT be fac-weighted).  The detJ-weighted reduction has a
# slightly larger round-off floor than the uniform one, but on the
# mass-relative basis both the identity-map control and the stretched
# case sit at ~1e-15 -- i.e. mapping does not degrade conservation.
#
# Robustness: tempMass opens in append mode, so re-running appends another
# header + iter-0..N block.  We read with no header, coerce to numeric,
# drop the repeated-header rows, keep only the LAST block, and drop the
# iter == 0 startup row (dmdt undefined there).
# ========================================================================

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Tolerances calibrated against the identity-map control, which achieves
# per-step |dM - flux*dt|/M ~ 7e-16 and signed cumulative drift ~ 3.5e-13.
# These leave a few decades of margin over the round-off floor while
# staying many decades below any real conservation defect (>= ~1e-6).
TOL_PERSTEP = 1.0e-11  # max |dM - netMassFlux*dt| / M
TOL_CUMULATIVE = 1.0e-10  # |(M_end - M_0) - sum(flux*dt)| / M

COLS = ["iter", "time", "massNew", "dmdt", "netMassFlux", "balance"]


def _load_last_block(fname):
    """Return the last contiguous iter-block of tempMass as numeric data,
    with the iter == 0 startup row removed.

    Handles append-mode reopening (repeated headers, multiple blocks)."""
    raw = pd.read_csv(fname, header=None, names=COLS)
    num = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    num = num.reset_index(drop=True)
    it = num["iter"].to_numpy()
    # A new block starts wherever iter does not increase.
    starts = [0] + [i for i in range(1, len(it)) if it[i] <= it[i - 1]]
    last = num.iloc[starts[-1]:] if len(num) else num
    last = last[last["iter"] > 0]
    return last.reset_index(drop=True)


class ConsTestCase(unittest.TestCase):
    """Flux-based mass-conservation test for the mapped AMR flame."""

    def test_mass_flux_balance(self):
        fname = os.path.join(os.path.abspath("."), "temporals", "tempMass")
        self.assertTrue(
            os.path.isfile(fname),
            f"missing {fname} -- needs peleLM.do_temporals=1 and "
            "peleLM.do_mass_balance=1",
        )

        df = _load_last_block(fname)
        self.assertGreater(len(df), 0, "tempMass has no usable time history")

        mass = df["massNew"].to_numpy()
        flux = df["netMassFlux"].to_numpy()
        balance = np.abs(df["balance"].to_numpy())  # |dmdt - netMassFlux|
        time = df["time"].to_numpy()
        dt = np.clip(np.diff(time, prepend=time[0]), 0.0, None)
        mass_scale = float(np.abs(mass).max())

        # (1) dt-robust per-step mass imbalance: |dM - flux*dt| / M.
        perstep = (balance * dt) / mass_scale
        worst = float(perstep.max())

        # (2) signed cumulative drift over the run, relative to mass.
        cum = float(abs((mass[-1] - mass[0]) - np.sum(flux * dt)) / mass_scale)

        # Report to stderr (visible in `ctest -VV` on pass or fail;
        # nosetests captures stdout but lets stderr through).
        print(
            "\n[mass-balance] rows used (last block)        : {}\n"
            "[mass-balance] worst |dM - flux*dt| / M      : {:.3e}  (allowed < {:.1e})\n"
            "[mass-balance] signed cumulative drift / M   : {:.3e}  (allowed < {:.1e})\n"
            "[mass-balance] (machine eps ~ 1.1e-16)".format(
                len(df), worst, TOL_PERSTEP, cum, TOL_CUMULATIVE
            ),
            file=sys.stderr,
            flush=True,
        )

        self.assertLessEqual(
            worst,
            TOL_PERSTEP,
            f"per-step mass imbalance |dM - flux*dt|/M = {worst:.3e} "
            f"exceeds {TOL_PERSTEP:.1e} (round-off floor ~1e-15; a value "
            f"this large indicates a real conservation error)",
        )
        self.assertLessEqual(
            cum,
            TOL_CUMULATIVE,
            f"signed cumulative mass drift / M = {cum:.3e} exceeds "
            f"{TOL_CUMULATIVE:.1e}",
        )


if __name__ == "__main__":
    unittest.main()
