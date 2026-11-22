import os
import re
import numpy as np
import pandas as pd
import unittest

def read_last_dataset(path, **read_csv_kwargs):
    """Read only the most-recently-appended block from a header-repeating CSV.

    The file is assumed to have the same column-header line at the start
    of every appended block. This finds the last occurrence of that line
    and returns a DataFrame of just the rows after it, with the header
    taken from that line. Extra kwargs are forwarded to pandas.read_csv.
    """
    with open(path) as f:
        header = f.readline().rstrip('\n')
        last_header_lineno = 0  # 0 = the very first header at the top
        for lineno, line in enumerate(f, start=1):
            if line.rstrip('\n') == header:
                last_header_lineno = lineno
    return pd.read_csv(path, skiprows=last_header_lineno, **read_csv_kwargs)

@unittest.skipUnless(
    os.path.exists("composition-test-2d.inp"), "not the composition test"
)
class CompTestCase(unittest.TestCase):
    """Test composition of species with external sources"""

    def test_composition(self):
        """Are the number of moles, mass fractions, and density correct?"""

        # Molar masses (kg/mol)
        molar_masses = {"AR": 0.040, "N2": 0.028, "CO2": 0.044}

        # Load the data
        file_dir = os.path.abspath(".")
        file_name = os.path.join(file_dir, "temporals/tempExtremas")
        col_names = ["time", "max_density", "max_rho.Y(AR)", "max_rho.Y(N2)", "max_rho.Y(CO2)"]
        var_names = ["AR", "N2", "CO2"]
        data = read_last_dataset(file_name, usecols=col_names, delimiter=',')
        print("Raw data:\n", data)

        # Ensure numeric types for all data columns
        data = data.apply(pd.to_numeric, errors="coerce")

        # Check for NaN values and raise an error if found
        if data.isnull().any().any():
            raise ValueError("Data contains NaN values after conversion.")
        
        time = data["time"]

        # Parse input file for necessary solution parameters
        input_file = os.path.join(file_dir, "composition-test-2d.inp")
        with open(input_file, 'r') as file:
            content = file.read()
            soln_params = {
                "ts": float(re.search(r"prob\.extRhoYCO2_ts\s*=\s*([\d\.]+)", content).group(1)),
                "src_strength": float(re.search(r"prob\.extRhoYCO2\s*=\s*([\d\.]+)", content).group(1))
            }
            ramp = re.search(r"prob\.extRhoYCO2_ramp\s*=\s*([\d\.eE+-]+)", content)
            soln_params["ramp"] = float(ramp.group(1)) if ramp else 0.0
        ts, k, w = soln_params["ts"], soln_params["src_strength"], soln_params["ramp"]

        # Density and species mass data
        data_rho = data["max_density"]
        data_rhoY = data.iloc[:, 2:].copy()
        data_rhoY.columns = var_names

        # Ensure numeric types for all data columns
        data_rho = pd.to_numeric(data_rho, errors="coerce")
        data_rhoY = data_rhoY.apply(pd.to_numeric, errors="coerce")

        # Initial values
        AR_0, N2_0, CO2_0 = data_rhoY.iloc[0].to_dict().values()
        rho_0 = data_rho.iloc[0]

        # Mass fractions and number of moles
        data_Y = data_rhoY.div(data_rho, axis=0)
        data_Y["Sum(Y_m)"] = data_Y.sum(axis=1)
        data_moles = data_rhoY.div([molar_masses[comp] for comp in var_names], axis=1)

        # Exact solutions: d(rhoY_CO2)/dt = k H(t) rhoY_CO2, with H the
        # smoothstep switch-on over [ts, ts + w] (a step when w = 0), so
        # rhoY_CO2 = CO2_0 exp(k int_0^t H)
        tau = np.maximum(time - ts, 0.0)
        if w > 0:
            s = np.minimum(tau / w, 1.0)
            int_H = w * (s**3 - 0.5 * s**4) + np.maximum(tau - w, 0.0)
        else:
            int_H = tau
        exact_rhoY = {
            "AR": AR_0,
            "N2": N2_0,
            "CO2": CO2_0 * np.exp(k * int_H)
        }
        exact_density = rho_0 + CO2_0 * (np.exp(k * int_H) - 1)
        exact_Y = pd.DataFrame({
            key: exact_rhoY[key] / exact_density for key in var_names},
            index=data.index)
        exact_moles = pd.DataFrame({
            key: exact_rhoY[key] / molar_masses[key] for key in var_names}, 
            index=data.index)

        # Error calculations
        max_error_density = np.abs(data_rho - exact_density).max()
        max_error_Y = np.abs(data_Y.iloc[:, :-1] - exact_Y).max()
        max_error_moles = np.abs(data_moles - exact_moles).max()

        # Expected max errors ["AR", "N2", "CO2"]
        # (dt = 1e-3, sdc_iterMax = 2, ext_sources_SDC on: second order in
        # dt; with ext_sources_SDC = 0 the errors are ~40x larger)
        expected_errors_moles = np.array([1e-15, 1e-15, 4e-02])
        expected_errors_massfrac = np.array([6e-04, 2.5e-04, 8e-04])
        expected_error_rho = 1.8e-03

        # Perform tests using np.testing to make sure errors haven't increased
        np.testing.assert_array_less(
            np.array([max_error_moles[var] for var in var_names]),
            expected_errors_moles,
            err_msg="Maximum mole errors exceed specified tolerance."
        )

        np.testing.assert_array_less(
            np.array([max_error_Y[var] for var in var_names]),
            expected_errors_massfrac,
            err_msg="Maximum mass fraction errors exceed specified tolerance."
        )

        np.testing.assert_array_less(
            np.array([max_error_density]),
            np.array([expected_error_rho]),
            err_msg="Maximum density error exceeds specified tolerance."
        )

@unittest.skipUnless(
    os.path.exists("isothermal-injection-2d.inp"),
    "not the isothermal injection test",
)
class IsothermalInjectionTestCase(unittest.TestCase):
    """CO2 injected at the gas temperature must leave T unchanged"""

    def test_temperature(self):
        """Does T stay at T_mean while the injection drives a flow?

        The source adds CO2 with its enthalpy at T_mean, so it adds no heat
        to a mixture at T_mean, and a uniform T stays uniform under
        advection and diffusion. The rhoH fluxes are built from the
        predicted T edge states, so a missing -Sum{hk extRhoYk} in the
        temperature edge forcing shows up as a drift of T (~2 K in 10
        steps).
        """
        file_name = os.path.join(os.path.abspath("."), "temporals/tempExtremas")
        data = read_last_dataset(
            file_name, usecols=["time", "min_temp", "max_temp"], delimiter=","
        ).apply(pd.to_numeric, errors="coerce")
        if data.isnull().any().any():
            raise ValueError("Data contains NaN values after conversion.")

        with open("isothermal-injection-2d.inp") as f:
            T_mean = float(
                re.search(r"prob\.T_mean\s*=\s*([\d\.eE+-]+)", f.read()).group(1)
            )

        max_dev = max(
            np.abs(data["min_temp"] - T_mean).max(),
            np.abs(data["max_temp"] - T_mean).max(),
        )
        print("Max |T - T_mean| =", max_dev)
        np.testing.assert_array_less(
            np.array([max_dev]),
            np.array([1.0e-6]),
            err_msg="T drifted from T_mean under an isothermal injection.",
        )


if __name__ == "__main__":
    unittest.main()
