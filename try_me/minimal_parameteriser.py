#!/usr/bin/env python3

from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import pybop
import pybamm
import fitter


"""
Headers store the relevant column names in a battery cycler file.
These can easily be changed for different battery cyclers.
"""
NewareHeaders = fitter.Headers(
    "Step Type", "Total Time", "Current(A)", "Voltage(V)", "CC Chg", "CC DChg", "Rest"
)


def import_neware(filename: str) -> pd.DataFrame:
    """
    Import a datafile from a Neware battery cycler. The datafile
    should store the GITT parameterisation experiment we're wanting to
    work from.
    """
    # Load the battery cycler file
    try:
        df = pd.read_csv(filename, encoding_errors="ignore")
    except pd.errors.ParserError:
        df = pd.read_excel(filename, sheet_name="record")
    # Convert the timestamp to a time in seconds
    seconds = (
        df[NewareHeaders.time]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
    )
    df[NewareHeaders.time] = seconds.to_numpy()
    return df


def get_ocv(
    df: pd.DataFrame, socs: np.ndarray, headers: fitter.Headers, direction: str
) -> Tuple[np.ndarray, np.ndarray, pybamm.Interpolant]:
    """
    This function generates OCV data, by taking the cell voltage and
    SOC at the end of each GITT pulse. The raw voltage and SOC data
    are returned, so that they can be saved as part of the model
    parameters. Additionally, a PyBaMM interpolant is produced, which
    is used to build a model for parameterising.
    """
    pulses = fitter.get_pulse_data(
        df,
        socs,
        headers,
        direction,
    )
    ocv_socs, ocv_vs = fitter.get_ocvs_from_pulsedataset_list(pulses)
    warnings.warn(
        "Adding fictitious OCV points outside cell operating voltages, for interpolator stability"
    )
    ocv_socs = np.r_[-0.01, ocv_socs, 1.01]
    ocv_vs = np.r_[2.99, ocv_vs, 4.21]
    return ocv_socs, ocv_vs, fitter.build_ocv_interpolant(ocv_socs, ocv_vs)


def main():
    capacity_Ah = 2.2
    df = import_neware("MLP001_25degC.xlsx")

    # Calculate state-of-charge throughout the experiment
    socs = fitter.coulomb_count(
        df[NewareHeaders.time],
        df[NewareHeaders.current],
        capacity_Ah,
        1,
    )

    # Find OCV(SOC) data
    ocv_socs, ocv_vs, ocv_func = get_ocv(
        df,
        socs,
        NewareHeaders,
        "discharge",
    )

    # Extract a list of pulse-data to fit to
    pulses = fitter.get_pulse_data(
        df,
        socs,
        NewareHeaders,
        "discharge",
        ignore_rests=True,
    )
    # Rubbish data at low SOC, as we keep hitting lower voltage
    # cutoff; ignore these pulses.
    pulses = pulses[:-3]

    # Define our bounds and initial guesses for each parameter
    fitting_params = {
        "R0 [Ohm]": pybop.Parameter(
            initial_value=0.2,
            bounds=[1e-4, 0.4],
        ),
        "R1 [Ohm]": pybop.Parameter(
            initial_value=0.02,
            bounds=[1e-5, 1e-1],
        ),
        "R2 [Ohm]": pybop.Parameter(
            initial_value=0.02,
            bounds=[1e-5, 1e-1],
        ),
        "tau1 [s]": pybop.Parameter(
            initial_value=6,
            bounds=[5, 180],
        ),
        "tau2 [s]": pybop.Parameter(
            initial_value=60,
            bounds=[5, 180],
        ),
    }
    # The model to be parameterised needs to be seeded with some
    # initial information; here, we get an initial parameter set which
    # includes the OCV function, cell capacity, number of RC pairs to
    # parameterise, and the initial parameter guesses. These are
    # included alongside a range of less interesting parameters which
    # PyBOP expects to see, but we don't care about.
    base_params = fitter.get_initial_parameters(
        capacity_Ah, fitting_params, ocv_func, n_rc=2
    )

    # Run the parameterisation!
    pars_df, _, _ = fitter.parameterise(
        pulses,
        base_params,
        fitting_params,
        n_rc=2,
        method=[  # Each method in this list is applied to every pulse,
            # and the best result is retained each time
            "SLSQP",
            "trust-constr",
            pybop.PSO,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ],
        plot=False,
    )

    pars_df.to_csv("minimal_parameteriser_parameters.csv", index=False)
    pd.DataFrame({"SOC": ocv_socs, "OCV[V]": ocv_vs}).to_csv(
        "minimal_parameteriser_ocv.csv", index=False
    )


if __name__ == "__main__":
    main()
