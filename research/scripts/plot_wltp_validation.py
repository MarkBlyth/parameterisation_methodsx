#!/usr/bin/env python3

from __future__ import annotations
import sys
from typing import Tuple
import pandas as pd
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import model
import utils


def get_drivecycle(cycler_file_name: str) -> pd.DataFrame:
    df = pd.read_excel(cycler_file_name, engine="openpyxl", sheet_name="record")

    hours = (
        df["Total Time"]
        .str.split(":")
        .apply(lambda x: (int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2])) / 3600)
    ).to_numpy()
    hours = hours - hours[0]
    df["Total Time"] = hours

    return df


def run_simulation(
    ocv_file: str,
    param_file: str,
    df: pd.DataFrame,
    capacity: float,
    initial_soc: float,
    t_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    simulator = model.TheveninModel(ocv_file, param_file, capacity)

    times = df["Total Time"].to_numpy() * 3600
    currents = -df["Current(A)"].to_numpy()
    if t_max is None:
        t_max = times[-1]

    def currentfunc(t):
        if t > times[-1]:
            return None
        return np.interp(t, times, currents, 0)

    ts, _, vs, _, _, _ = simulator.simulate(
        currentfunc,
        initial_soc,
        temp_inf=25,
        t_max=t_max,
        max_step=2,
        atol=1e-5,
        rtol=1e-5,
    )
    return ts / 3600, vs


def calculate_errors(
    model_ts: np.ndarray, model_vs: np.ndarray, df: pd.DataFrame
) -> np.ndarray:
    resampled_experiment = np.interp(
        model_ts,
        df["Total Time"],
        df["Voltage(V)"],
    )
    error = model_vs - resampled_experiment
    return error


def make_a_plot(
    df: pd.DataFrame,
    model_ts: np.ndarray,
    model_vs: np.ndarray,
    save_name: str | None = None,
):
    error = calculate_errors(model_ts, model_vs, df)
    print(max(error))

    (ax_I, ax_v, ax_e), tidy_up = utils.get_ax(
        bool(save_name),
        n_axes=3,
        bottom_extra=0.35,
        l_margin=1.7,
    )

    ax_I.set_xticks([])
    ax_v.set_xticks([])

    """ Applied current """
    cycler_time = df["Total Time"]
    cycler_current = df["Current(A)"]
    ax_I.plot(
        cycler_time,
        cycler_current,
        color=utils.current_colour,
    )
    ax_I.set_ylabel("Current [A]")

    """ Voltage: experiments / predictions """
    ax_v.plot(
        df["Total Time"],
        df["Voltage(V)"],
        label="Experiment",
        color="tab:cyan",
    )
    ax_v.plot(model_ts, model_vs, label="Model", color=utils.voltage_colour)
    ax_v.set_ylabel("Voltage [V]")
    ax_v.legend(frameon=False)

    """ Voltage prediction errors """
    ax_e.plot(model_ts, 1000 * error, color=utils.error_colour)
    ax_e.set_ylim(-16, 23)
    ax_e.set_ylabel("Model error [mV]")
    ax_e.set_xlabel("Time [h]")

    tidy_up(save_name)


def main(
    cycler_file: str,
    ocv_file: str,
    param_file: str,
    capacity_Ah: str,
    plot_file: str | None = None,
):
    df = get_drivecycle(cycler_file)
    ts, vs = run_simulation(ocv_file, param_file, df, float(capacity_Ah), 1)
    make_a_plot(
        df,
        ts,
        vs,
        plot_file,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print(
            "./plot_wltp_validation.py cycler_file ocv_file param_file capacity_Ah [plotfile]"
        )
    main(*sys.argv[1:])
