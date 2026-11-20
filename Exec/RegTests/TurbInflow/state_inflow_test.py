#!/usr/bin/env python3
"""State inflow test: temperature and species from turbinflow scalar fields.

usage:
  state_inflow_test.py make <velocity_turbfile> <out_turbfile>
  state_inflow_test.py check <ghost_plane_prefix> [--map tanh --beta 2.0]
      [--tol 1e-3]

make: copy a periodic (turb_box) turbulence file and append analytic scalar
fields temp, Y(O2), Y(N2) (SCALARS_V1 trailer).  File transverse node i lies
at X = (i - 0.5) dx from the low end of the file (one ghost node on each
side), plane k at Z = (k + 0.5) dz:

  T     = 600 + 200 sin(X) cos(Y) + 100 sin(Z)
  Y(O2) = 0.233 + 0.1 sin(X + Y),  Y(N2) = 1 - Y(O2)

check: compare the inflow ghost layers dumped by DiagFramePlane
(input.3d_StateInflow) with these fields at the physical cell centres
(X = turb_scale_loc * x, Z = turb_scale_loc * turb_conv_vel * t), and the
density with the ideal gas law at 1 atm.  The tolerance covers the quadratic
interpolation of the file; --map tanh selects the physical positions of a
tanh-mapped x direction.
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_inflow_roundtrip import load_plane, tanh_x_from_xi  # noqa: E402

SCALE = 2.0 * np.pi / 0.01  # turb_scale_loc of input.3d_StateInflow
CONV = 5.0  # turb_conv_vel
P_AMB = 101325.0
W_O2, W_N2, R_GAS = 31.998e-3, 28.014e-3, 8.31446261815324


def temp(X, Y, Z):
    return 600.0 + 200.0 * np.sin(X) * np.cos(Y) + 100.0 * np.sin(Z)


def y_o2(X, Y, Z):
    return 0.233 + 0.1 * np.sin(X + Y) + 0.0 * Z


FIELDS = {
    "temp": temp,
    "Y(O2)": y_o2,
    "Y(N2)": lambda X, Y, Z: 1.0 - y_o2(X, Y, Z),
}


def make(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    with open(os.path.join(dst, "HDR")) as f:
        hdr = f.read().split("\n")
    nx, ny, nz = map(int, hdr[0].split())
    lx, ly, lz = map(float, hdr[1].split())
    if hdr[2].split()[2] != "1":
        sys.exit("make expects a file periodic in the normal direction")
    dx, dy, dz = lx / (nx - 1), ly / (ny - 1), lz / nz
    X, Y = np.meshgrid(
        (np.arange(nx) - 0.5) * dx, (np.arange(ny) - 0.5) * dy, indexing="ij"
    )
    head = (
        "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"
        f"((0,0,0) ({nx - 1},{ny - 1},0) (0,0,0)) 1\n"
    ).encode()
    offsets = []
    with open(os.path.join(dst, "DAT"), "ab") as f:
        for fn in FIELDS.values():
            for k in range(nz):
                offsets.append(f.tell())
                f.write(head)
                plane = fn(X, Y, (k + 0.5) * dz)
                # Fortran order: first index fastest
                f.write(np.asarray(plane, "<f8").T.tobytes())
    with open(os.path.join(dst, "HDR"), "a") as f:
        f.write(f"SCALARS_V1 {len(FIELDS)}\n" + " ".join(FIELDS) + "\n")
        f.write("\n".join(map(str, offsets)) + "\n")
    print(f"wrote {dst}: {len(FIELDS)} scalar fields x {nz} planes")


def check(prefix, mapping, beta, tol):
    planes = sorted(glob.glob(prefix + "*"))
    if not planes:
        sys.exit("no ghost planes matching " + prefix + "*")
    worst = {}
    for d in planes:
        hdr, a = load_plane(d)
        names = hdr["names"]
        nx, ny = a.shape[1:]
        lo, hi = hdr["prob_lo"], hdr["prob_hi"]
        xc = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
        yc = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
        if mapping == "tanh":
            xc = tanh_x_from_xi(xc, beta, lo[0], hi[0])
        X, Y = np.meshgrid(SCALE * xc, SCALE * yc, indexing="ij")
        Z = SCALE * CONV * hdr["time"]
        ref = {k: fn(X, Y, Z) for k, fn in FIELDS.items()}
        ref["density"] = P_AMB / (
            R_GAS
            * ref["temp"]
            * (ref["Y(O2)"] / W_O2 + ref["Y(N2)"] / W_N2)
        )
        for k, r in ref.items():
            err = np.abs(a[names.index(k)] - r).max() / np.abs(r).max()
            worst[k] = max(worst.get(k, 0.0), err)
    ok = True
    for k, e in worst.items():
        status = "PASS" if e <= tol else "FAIL"
        ok = ok and e <= tol
        print(f"{k:8s} max rel. error over {len(planes)} planes = {e:.3e} {status}")
    print("state inflow: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("src")
    m.add_argument("dst")
    c = sub.add_parser("check")
    c.add_argument("prefix")
    c.add_argument("--map", choices=["none", "tanh"], default="none")
    c.add_argument("--beta", type=float, default=2.0)
    c.add_argument("--tol", type=float, default=1.0e-3)
    args = p.parse_args()
    if args.cmd == "make":
        make(args.src, args.dst)
        return 0
    return check(args.prefix, args.map, args.beta, args.tol)


if __name__ == "__main__":
    sys.exit(main())
