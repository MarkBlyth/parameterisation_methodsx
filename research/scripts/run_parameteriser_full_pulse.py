#!/usr/bin/env python3

from __future__ import annotations
import warnings
import argparse
import csv
import pandas as pd
import numpy as np
import pybop
import pybamm
import fitter

NewareHeaders = fitter.Headers(
    "Step Type", "Total Time", "Current(A)", "Voltage(V)", "CC Chg", "CC DChg", "Rest"
)


def import_neware(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, encoding_errors="ignore")
    except pd.errors.ParserError:
        df = pd.read_excel(filename, sheet_name="record")
    seconds = (
        df[NewareHeaders.time]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
    )
    df[NewareHeaders.time] = seconds.to_numpy()
    return df


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_parameteriser.py",
        description="Run the parameterisation routine on a single battery cycler output-file. Note that this is not a general parameteriser, but has been tuned to the chosen test-cell.",
    )
    parser.add_argument(
        "filename",
        help="Battery cycler file to parameterise from",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Temperature of this experiment",
        default=np.nan,
    )
    parser.add_argument(
        "-p",
        "--paramfile",
        help="Output filename for parameter file",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--errorfile",
        help="Output filename for saving parameterisation mean-squared errors",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--ocvfile",
        help="Output filename for OCV file",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--charge",
        help="If set, parameterise in charge; otherwise, parameterise discharge",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def get_ocv(
    df: pd.DataFrame, socs: np.ndarray, headers: fitter.Headers, direction: str
) -> Tuple[np.ndarray, np.ndarray, pybamm.Interpolant]:
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
    for i in range(ocv_vs.size):
        monotone_failures = ocv_vs[i + 1 :] < ocv_vs[i]
        if any(monotone_failures):
            warnings.warn("OCV not monotone in SOC; applying correction.")
            ocv_vs[i + 1 :][monotone_failures] = ocv_vs[i]
    return ocv_socs, ocv_vs, fitter.build_ocv_interpolant(ocv_socs, ocv_vs)


def add_fictitious_params(
    df: pd.DataFrame,
    temperature: float = np.nan,
    high_soc: float | None = 1,
    low_soc: float | None = None,
) -> pd.DataFrame:
    df["Temperature_degC"] = temperature
    if high_soc is not None:
        max_soc_row = df.iloc[df["SOC"].idxmax()].copy()
        max_soc_row["SOC"] = high_soc
        df = pd.concat((df, max_soc_row.to_frame().T), ignore_index=True)
    if low_soc is not None:
        min_soc_row = df.iloc[df["SOC"].idxmin()].copy()
        min_soc_row["SOC"] = low_soc
        df = pd.concat((df, min_soc_row.to_frame().T), ignore_index=True)
    return df.sort_values(by=["SOC"])


def main():
    args = get_args()
    capacity_Ah = 2.2

    df = import_neware(args.filename)
    socs = fitter.coulomb_count(
        df[NewareHeaders.time],
        df[NewareHeaders.current],
        capacity_Ah,
        1,
    )

    direction = "charge" if args.charge else "discharge"
    ocv_socs, ocv_vs, ocv_func = get_ocv(
        df,
        socs,
        NewareHeaders,
        direction,
    )
    pulses = fitter.get_pulse_data(
        df,
        socs,
        NewareHeaders,
        direction,
        ignore_rests=False,
    )
    # Rubbish data at low SOC, as we keep hitting lower voltage
    # cutoff; ignore these pulses.
    pulses = pulses[:-3]
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
    base_params = fitter.get_initial_parameters(
        capacity_Ah, fitting_params, ocv_func, n_rc=2
    )
    pars_df, _, solns = fitter.parameterise(
        pulses,
        base_params,
        fitting_params,
        n_rc=2,
        method=[
            "SLSQP",
            "trust-constr",
            pybop.PSO,
            pybop.CMAES,
            pybop.SNES,
            pybop.XNES,
        ],
        plot=False,
    )

    if args.errorfile:
        costs = [soln.best_cost for soln in solns]
        with open(args.errorfile, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(pars_df["SOC"].to_numpy())
            writer.writerow(costs)

    if args.paramfile:
        # Duplicate the parameters from max. SOC at SOC=1.01
        pars_df = add_fictitious_params(pars_df, args.temperature, 1.01, 0.05)
        pars_df.to_csv(args.paramfile, index=False)

    if args.ocvfile:
        ocv_df = pd.DataFrame.from_dict(
            {"SOC": ocv_socs, "OCV[V]": ocv_vs, "Temperature_degC": args.temperature}
        )
        ocv_df.to_csv(args.ocvfile, index=False)


if __name__ == "__main__":
    main()
