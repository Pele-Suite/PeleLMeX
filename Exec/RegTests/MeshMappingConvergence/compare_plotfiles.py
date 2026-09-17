#!/usr/bin/env python3
"""Compare two single-level PeleLMeX plotfiles on index-identical grids:
a reference run and a mesh-mapped (ConstantMap) run of the same physical
problem.  Every field must agree to round-off once the mapping's known
scalings are undone:

  gradpx/y/z   stored as the Xi-space gradient fac_d * dp/dx_d  -> divided by fac_d
  everything else is physical (velocity, density, species, T, divu, ...)

Fields whose derivation is not mapping-aware are skipped (mag_vort by
default; add more with --skip).  A field passes when
    max |mapped - ref| <= tol * max(max |ref|, floor)
so fields that are numerically zero in both runs do not trip the gate.

usage: compare_plotfiles.py REF MAPPED [--fac fx,fy(,fz)] [--tol 1e-10]
                            [--floor 1e-30] [--skip mag_vort,...] [--only a,b]
exit status 0 on PASS, 1 on FAIL.
"""
import argparse
import os
import re
import sys

import numpy as np


def _box(s):
    n = [int(v) for v in re.findall(r"-?\d+", s)]
    d = len(n) // 3
    return tuple(n[:d]), tuple(n[d:2 * d])


def read_fab(path, off):
    with open(path, "rb") as f:
        f.seek(off)
        hdr = b""
        while not hdr.endswith(b"\n"):
            c = f.read(1)
            if not c:
                sys.exit("unexpected EOF reading FAB header in " + path)
            hdr += c
        hdr = hdr.decode()
        if "(64 11 52 0 1 12 0 1023)" not in hdr:
            sys.exit("unsupported FAB real format in " + path)
        boxes = re.findall(r"\(\([^()]*\)\s*\([^()]*\)\s*\([^()]*\)\)", hdr)
        lo, hi = _box(boxes[-1])
        nc = int(hdr.strip().split()[-1])
        big = "(1 2 3 4 5 6 7 8)" in hdr
        shape = [h - l + 1 for l, h in zip(lo, hi)]
        data = np.frombuffer(f.read(8 * int(np.prod(shape)) * nc),
                             dtype=">f8" if big else "<f8")
    arr = data.reshape((nc,) + tuple(shape[::-1])).transpose(0, *range(len(shape), 0, -1))
    return lo, arr


def load_plotfile(plt):
    with open(os.path.join(plt, "Header")) as f:
        lines = [ln.rstrip("\n") for ln in f]
    nvar = int(lines[1])
    names = lines[2:2 + nvar]
    finest = int(lines[2 + nvar + 2])
    if finest != 0:
        sys.exit("%s has %d levels; this tool compares single-level plotfiles" % (plt, finest + 1))
    lev = os.path.join(plt, "Level_0")
    with open(os.path.join(lev, "Cell_H")) as f:
        txt = f.read()
    ng = int(txt.split("\n")[3].split()[0])
    fabs = {}
    for fname, off in re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt):
        lo, arr = read_fab(os.path.join(lev, fname), int(off))
        if ng:
            arr = arr[(slice(None),) + tuple(slice(ng, -ng) for _ in lo)]
            lo = tuple(l + ng for l in lo)
        fabs[lo] = arr
    return names, fabs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("mapped")
    ap.add_argument("--fac", default="1,1,1", help="ConstantMap factors of the mapped run, comma separated")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--floor", type=float, default=1e-30, help="absolute floor on the reference scale of a field")
    ap.add_argument("--skip", default="mag_vort", help="comma separated fields to ignore")
    ap.add_argument("--only", default="", help="comma separated fields to compare (default: all)")
    a = ap.parse_args()
    fac = [float(v) for v in a.fac.split(",")]
    skip = {s for s in a.skip.split(",") if s}
    only = {s for s in a.only.split(",") if s}

    names_r, R = load_plotfile(a.ref)
    names_m, M = load_plotfile(a.mapped)
    if names_r != names_m:
        sys.exit("variable lists differ:\n  ref:    %s\n  mapped: %s" % (names_r, names_m))
    if set(R) != set(M) or any(R[k].shape != M[k].shape for k in R):
        sys.exit("box layouts differ (same amr.n_cell and max_grid_size are required)")

    ok = True
    print("%-24s %14s %14s %12s" % ("field", "max|ref|", "max|diff|", "diff/scale"))
    for c, name in enumerate(names_r):
        if name in skip or (only and name not in only):
            continue
        scale_m = 1.0
        mgp = re.match(r"gradp([xyz])$", name)
        if mgp:
            scale_m = 1.0 / fac["xyz".index(mgp.group(1))]
        maxref = 0.0
        maxdiff = 0.0
        for k in R:
            r = R[k][c]
            m = M[k][c] * scale_m
            if not (np.isfinite(r).all() and np.isfinite(m).all()):
                sys.exit("non-finite values in field " + name)
            maxref = max(maxref, float(np.abs(r).max()))
            maxdiff = max(maxdiff, float(np.abs(m - r).max()))
        scale = max(maxref, a.floor)
        rel = maxdiff / scale
        flag = "" if rel <= a.tol else "  <-- FAIL"
        ok &= (rel <= a.tol)
        print("%-24s %14.4e %14.4e %12.3e%s" % (name, maxref, maxdiff, rel, flag))
    print("\n%s: mapped run %s the reference to %.1e (relative to each field's max, floor %.0e)"
          % ("PASS" if ok else "FAIL", "matches" if ok else "does not match", a.tol, a.floor))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
