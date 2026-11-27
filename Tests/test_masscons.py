# ========================================================================
#
# Imports
#
# ========================================================================
import io
import os
import numpy.testing as npt
import pandas as pd
import unittest


# ========================================================================
#
# Test definitions
#
# ========================================================================
class ConsTestCase(unittest.TestCase):
    """Tests for conservation in Pele."""

    def test_conservation(self):
        """Is mass conserved?"""

        # Load the data
        fdir = os.path.abspath(".")
        fname = os.path.join(fdir, "temporals/tempMass")
        # CSV with a header: iter,time,massNew,dmdt,netMassFlux,balance.
        # The file is appended to, so a rerun or restart in the same
        # directory adds another header: keep the last run only.
        with open(fname) as f:
            lines = f.readlines()
        last = max(i for i, l in enumerate(lines) if l.startswith("iter,"))
        df = pd.read_csv(io.StringIO("".join(lines[last:])))
        self.assertGreater(len(df), 1)
        self.assertFalse(df.massNew.isnull().any())
        init_mass = df.massNew[0]
        # Closed/periodic domains conserve mass to round-off, and MFSum sums
        # each box with Kahan compensation, so the domain total holds to a
        # few ulps (CPU; measured on the CI cases on 1 and 4 ranks).
        npt.assert_allclose(df.massNew, init_mass, rtol=1e-15)


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    unittest.main()
