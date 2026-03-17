#!/usr/bin/env python3

from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model


def main(
    drivecycle_file: str,
    param_file: str,
    ocv_file: str,
    capacity_Ah: float,
):
    simulator = model.TheveninModel(ocv_file, param_file, capacity_Ah)

    df = pd.read_csv(drivecycle_file)
    currents = df["Current[A]"].to_numpy()
    times = df["Time[s]"].to_numpy()
    t_max = times[-1] - 100 # to avoid going out of SOC parameter range

    def currentfunc(t):
        return np.interp(t, times, currents, 0)

    ts, _, vs, _, _, _ = simulator.simulate(
        currentfunc,
        initial_soc=0.99,
        temp_inf=25,
        t_max=t_max,
        max_step=2,
        atol=1e-5,
        rtol=1e-5,
    )
    _, ax = plt.subplots()
    ax.plot(ts, vs)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell voltage [V]")
    plt.show()


if __name__ == "__main__":
    main("wltp.csv", "MLP001_params.csv", "MLP001_ocv.csv", 2.132)
