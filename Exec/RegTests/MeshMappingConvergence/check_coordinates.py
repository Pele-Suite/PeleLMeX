#!/usr/bin/env python3
"""Check the x/y/z coordinate derives of a TanhStretchMap plotfile against
the analytic map.

The plotfile is written on the uniform Xi grid; the "coordinates" derive
must hold the *physical* cell-centre position, i.e. the tanh image of the
Xi cell centre, in every mapped direction (and the Xi centre itself in an
unmapped one).  Before the derives were made mapping-aware it held the Xi
centre everywhere.

usage: check_coordinates.py PLOTFILE --beta bx,by,bz [--tol 1e-12]
exit status 0 on PASS, 1 on FAIL.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_plotfiles import load_plotfile  # noqa: E402


def read_geometry(plt):
    with open(os.path.join(plt, "Header")) as f:
        lines = [ln.rstrip("\n") for ln in f]
    nvar = int(lines[1])
    dim = int(lines[2 + nvar])
    plo = [float(v) for v in lines[2 + nvar + 3].split()]
    phi = [float(v) for v in lines[2 + nvar + 4].split()]
    return dim, plo, phi


def tanh_offset(s, beta):
    if abs(beta) < 1e-12:
        return s
    return 0.5 * (1.0 + np.tanh(beta * (2.0 * s - 1.0)) / np.tanh(beta))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotfile")
    ap.add_argument("--beta", required=True, help="TanhStretchMap.beta, comma separated")
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    beta = [float(v) for v in a.beta.split(",")]

    names, fabs = load_plotfile(a.plotfile)
    dim, plo, phi = read_geometry(a.plotfile)
    # Domain size in cells from the union of the FABs.
    hi = [max(lo[d] + fab.shape[1 + d] for lo, fab in fabs.items()) for d in range(dim)]
    ncell = hi
    dxi = [(phi[d] - plo[d]) / ncell[d] for d in range(dim)]

    ok = True
    for d, name in enumerate("xyz"[:dim]):
        if name not in names:
            sys.exit("field %s not in plotfile (add 'coordinates' to amr.derive_plot_vars)" % name)
        c = names.index(name)
        worst = 0.0
        for lo, fab in fabs.items():
            idx = lo[d] + np.arange(fab.shape[1 + d])
            s = (idx + 0.5) / ncell[d]
            x_exact = plo[d] + (phi[d] - plo[d]) * tanh_offset(s, beta[d])
            shape = [1] * dim
            shape[d] = -1
            diff = np.abs(fab[c] - x_exact.reshape(shape))
            worst = max(worst, float(diff.max()))
        rel = worst / (phi[d] - plo[d])
        good = rel <= a.tol
        ok &= good
        print("%s: max |plotfile - tanh map| / L = %.3e  %s" % (name, rel, "OK" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
