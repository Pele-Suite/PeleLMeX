#!/usr/bin/env python3
"""Score the inflow ghost layer injected from a mesh-mapped turbulence file
against the planes the file was made from.

usage:
  check_inflow_roundtrip.py <source_plane> <ghost_plane> [--tol 1e-12]
      [--level L] [--source-map tanh [--target-map tanh] --beta 2.0]
      [--fields x_velocity,y_velocity]

Both arguments are single-level AMReX plotfiles written by PelePhysics'
DiagFramePlane with normal = 2 (plane axes x, y), 2D or flat 3D.  The source
plane comes from the run that generated the turbulence file (its inflow
ghost layer, uniform in that run's Xi); the ghost plane is the inflow ghost
layer of a run that injected the file.

Without --source-map the two runs share one mesh (same-map round trip): the
file is uniform in the target's own Xi, TurbInflow's inverse map lands on
file cell centres, and the ghost plane must reproduce the source to
round-off -- max |ghost - source| <= tol * max |source| per field.

With --source-map the source run carried the named map (tanh:
TanhStretchMap with --beta on x) and the target's cell centres do not
coincide with the file's: a uniform target (default), or a target with the
same map (--target-map tanh) but a different cell count, e.g. the level-1
ghost layer of a refined run (--level 1).  TurbInflow then inverts the
file's map at each target cell centre and interpolates linearly in the
file's Xi; this script does the same with numpy and requires agreement to
tol.  It also reports the error of the wrong hypothesis (the source read
at Xi = x, i.e. no inverse) and requires it to be clearly larger, so a
silently missing inverse fails the check.  --level selects the AMR level
of the ghost plane (default 0); cells of that level not covered by the
plane's grids are skipped.

Only the transverse velocity fields are compared by default: the normal
component carries the mean inflow, which the file adds to.
"""
import argparse
import os
import re
import sys

import numpy as np


# --------------------------------------------------------------------------
# Minimal reader for a single-level AMReX plotfile (VisMF v1, native double)
# --------------------------------------------------------------------------
def _box(s):
    nums = [int(v) for v in re.findall(r"-?\d+", s)]
    d = len(nums) // 3
    return tuple(nums[:d]), tuple(nums[d : 2 * d])


def read_header(pltdir, level=0):
    with open(os.path.join(pltdir, "Header")) as f:
        lines = [ln.rstrip("\n") for ln in f]
    nvar = int(lines[1])
    names = lines[2 : 2 + nvar]
    i = 2 + nvar
    dim = int(lines[i])
    time = float(lines[i + 1])
    finest = int(lines[i + 2])
    if level > finest:
        sys.exit("%s has finest level %d, requested %d" % (pltdir, finest, level))
    lo = [float(v) for v in lines[i + 3].split()]
    hi = [float(v) for v in lines[i + 4].split()]
    # line i+5: refinement ratios; i+6: one domain box per level
    boxes = re.findall(r"\(\([^()]*\)\s*\([^()]*\)\s*\([^()]*\)\)", lines[i + 6])
    dlo, dhi = _box(boxes[level])
    return {"names": names, "dim": dim, "time": time, "prob_lo": lo,
            "prob_hi": hi, "dom_lo": dlo, "dom_hi": dhi, "level": level}


def read_fab(path, offset, ncomp):
    with open(path, "rb") as f:
        f.seek(offset)
        hdr = b""
        while not hdr.endswith(b"\n"):
            c = f.read(1)
            if not c:
                sys.exit("unexpected EOF reading FAB header in " + path)
            hdr += c
        hdr = hdr.decode()
        if "(64 11 52 0 1 12 0 1023)" not in hdr:
            sys.exit("unsupported FAB real format in " + path + ": " + hdr)
        big_endian = "(1 2 3 4 5 6 7 8)" in hdr
        # The box is the last '((lo) (hi) (type))' group; the component
        # count is the trailing integer.
        boxes = re.findall(r"\(\([^()]*\)\s*\([^()]*\)\s*\([^()]*\)\)", hdr)
        box = _box(boxes[-1])
        nc = int(hdr.strip().split()[-1])
        if ncomp is not None and nc != ncomp:
            sys.exit("FAB in %s has %d components, expected %d" % (path, nc, ncomp))
        lo, hi = box
        shape = [h - l + 1 for l, h in zip(lo, hi)]
        n = int(np.prod(shape))
        data = np.frombuffer(f.read(8 * n * nc), dtype=">f8" if big_endian else "<f8")
    # Fortran order: fastest index first
    arr = data.reshape((nc,) + tuple(shape[::-1])).transpose(0, *range(len(shape), 0, -1))
    return lo, hi, arr


def load_plane(pltdir, level=0):
    hdr = read_header(pltdir, level)
    lev = os.path.join(pltdir, "Level_%d" % level)
    with open(os.path.join(lev, "Cell_H")) as f:
        txt = f.read()
    fabs = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt)
    if not fabs:
        sys.exit("no FabOnDisk entries in " + os.path.join(lev, "Cell_H"))
    dlo, dhi = hdr["dom_lo"], hdr["dom_hi"]
    nx, ny = dhi[0] - dlo[0] + 1, dhi[1] - dlo[1] + 1
    nvar = len(hdr["names"])
    full = np.full((nvar, nx, ny), np.nan)
    for fname, off in fabs:
        lo, hi, arr = read_fab(os.path.join(lev, fname), int(off), nvar)
        sl = tuple(slice(l - d, h - d + 1) for l, h, d in zip(lo[:2], hi[:2], dlo[:2]))
        if len(lo) == 3:
            arr = arr[:, :, :, 0]
        full[(slice(None),) + sl] = arr
    if level == 0 and np.isnan(full).any():
        sys.exit("plane not fully covered by FABs in " + pltdir)
    return hdr, full


# --------------------------------------------------------------------------
# Maps (same conventions as PeleLMeX's MeshMapEvaluator, physical <-> Xi)
# --------------------------------------------------------------------------
def tanh_xi_from_x(x, beta, plo, phi):
    L = phi - plo
    if abs(beta) < 1e-8:
        return x
    off = np.clip((x - plo) / L, 0.0, 1.0)
    arg = (2.0 * off - 1.0) * np.tanh(beta)
    return plo + L * 0.5 * (1.0 + np.arctanh(arg) / beta)


def tanh_x_from_xi(xi, beta, plo, phi):
    L = phi - plo
    if abs(beta) < 1e-8:
        return xi
    s = (xi - plo) / L
    return plo + L * 0.5 * (1.0 + np.tanh(beta * (2.0 * s - 1.0)) / np.tanh(beta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("ghost")
    ap.add_argument("--tol", type=float, default=1e-12, help="relative tolerance (default 1e-12)")
    ap.add_argument("--source-map", choices=["tanh"], default=None, help="map of the source run")
    ap.add_argument("--target-map", choices=["tanh"], default=None,
                    help="map of the target run (default: uniform); same beta as the source")
    ap.add_argument("--beta", type=float, default=2.0, help="tanh beta on x")
    ap.add_argument("--level", type=int, default=0, help="AMR level of the ghost plane to check (default 0)")
    ap.add_argument("--fields", default="x_velocity,y_velocity")
    ap.add_argument("--min-ratio", type=float, default=100.0,
                    help="cross-map: required ratio of the no-inverse error to the checked error")
    a = ap.parse_args()

    hs, S = load_plane(a.source)
    hg, G = load_plane(a.ghost, a.level)
    fields = a.fields.split(",")
    for f in fields:
        if f not in hs["names"] or f not in hg["names"]:
            sys.exit("field %s not in both planes" % f)
    if a.source_map is None and S.shape[2] != G.shape[2]:
        sys.exit("planes differ in y cell count (%d vs %d)" % (S.shape[2], G.shape[2]))

    if a.source_map is None:
        if S.shape != G.shape:
            sys.exit("same-map round trip needs equal plane shapes, got %s vs %s" % (S.shape, G.shape))
        ok = True
        for f in fields:
            s = S[hs["names"].index(f)]
            g = G[hg["names"].index(f)]
            scale = np.abs(s).max()
            err = np.abs(g - s).max()
            passed = err <= a.tol * scale
            ok &= passed
            print("%-12s max|ghost-source| = %.3e  (max|source| = %.3e, rel %.3e) %s"
                  % (f, err, scale, err / scale if scale > 0 else 0.0, "PASS" if passed else "FAIL"))
        print("same-map round trip:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    # Cross-map: the target's cell centres do not coincide with the file's.
    # Source and target may differ in map (uniform vs tanh) and in cell count
    # (a refined level); y is uniform on both, so a differing y count is a
    # pure refinement and is handled by nearest-cell lookup in y.
    nxs, nxg = S.shape[1], G.shape[1]
    nys, nyg = S.shape[2], G.shape[2]
    plo_s, phi_s = hs["prob_lo"][0], hs["prob_hi"][0]       # source Xi extent (= its physical extent for tanh)
    plo_g, phi_g = hg["prob_lo"][0], hg["prob_hi"][0]       # target extent (Xi for a mapped target)
    xi_src = plo_s + (np.arange(nxs) + 0.5) * (phi_s - plo_s) / nxs
    xi_grid_tgt = plo_g + (np.arange(nxg) + 0.5) * (phi_g - plo_g) / nxg
    if a.target_map == "tanh":
        x_tgt = tanh_x_from_xi(xi_grid_tgt, a.beta, plo_g, phi_g)     # physical centres of the mapped target
    else:
        x_tgt = xi_grid_tgt                                          # uniform target: grid coordinate is physical
    if a.source_map == "tanh":
        xi_tgt = tanh_xi_from_x(x_tgt, a.beta, plo_s, phi_s)
    inside = (xi_tgt >= xi_src[0]) & (xi_tgt <= xi_src[-1])
    if inside.sum() < 4:
        sys.exit("too few target cells inside the source plane's interior stencil")
    if nyg % nys != 0:
        sys.exit("target y count %d is not a multiple of the source's %d" % (nyg, nys))
    ry = nyg // nys
    jsrc = np.arange(nyg) // ry          # source column under each target column
    if ry > 1:
        # y is uniform on both: with the file's y spacing ry times the
        # target's, TurbInflow interpolates linearly in y too.
        y_src = (np.arange(nys) + 0.5) / nys
        y_tgt = (np.arange(nyg) + 0.5) / nyg
    covered = ~np.isnan(G[0])

    def interp_in_xi(s, xi):
        out = np.empty((xi.size, nyg))
        if ry == 1:
            for j in range(nys):
                out[:, j] = np.interp(xi, xi_src, s[:, j])
        else:
            tmp = np.empty((xi.size, nys))
            for j in range(nys):
                tmp[:, j] = np.interp(xi, xi_src, s[:, j])
            for i in range(xi.size):
                # y is periodic in these cases: wrap rather than clamp at the
                # edges, as the file's periodic ghost rows do.
                out[i, :] = np.interp(y_tgt, y_src, tmp[i, :], period=1.0)
        return out

    ok = True
    for f in fields:
        s = S[hs["names"].index(f)]
        g = G[hg["names"].index(f)]
        scale = np.abs(s).max()
        ref = interp_in_xi(s, xi_tgt)          # what TurbInflow must produce
        wrong = interp_in_xi(s, x_tgt)         # no inverse: source read at Xi = x
        mask = inside[:, None] & covered
        err = np.abs(g - ref)[mask].max()
        err_wrong = np.abs(g - wrong)[mask].max()
        passed = (err <= a.tol * scale) and (err_wrong >= a.min_ratio * max(err, np.finfo(float).tiny))
        ok &= passed
        print("%-12s max|ghost-interp_xi(source)| = %.3e  no-inverse hypothesis: %.3e  (max|source| = %.3e) %s"
              % (f, err, err_wrong, scale, "PASS" if passed else "FAIL"))
    print("cross-map sampling:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
