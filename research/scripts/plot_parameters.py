#!/usr/bin/env python3

import sys
import scipy.interpolate
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_soc_temperature_interpolant(
    df: pd.DataFrame, param_header: str
) -> scipy.interpolate.CloughTocher2DInterpolator:
    sortings = ["Temperature_degC", "SOC"]
    sort_df = df.sort_values(by=sortings)
    temps = sort_df["Temperature_degC"].to_numpy()
    socs = sort_df["SOC"].to_numpy()
    targetdata = sort_df[param_header].to_numpy()
    preds = np.vstack((temps, socs)).T
    return scipy.interpolate.CloughTocher2DInterpolator(
        np.vstack((temps, socs)).T,
        targetdata,
    )


def main(parfile: str, ocvfile: str, plot_file: str | None = None):
    if plot_file:
        mpl.use("pgf")
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    parsdf = pd.read_csv(parfile)
    ocvdf = pd.read_csv(ocvfile)

    r_interps = [
        get_soc_temperature_interpolant(parsdf, f"R{i} [Ohm]") for i in range(3)
    ]
    tau_interps = [
        get_soc_temperature_interpolant(parsdf, f"tau{i+1} [s]") for i in range(2)
    ]
    ocv_interp = get_soc_temperature_interpolant(ocvdf, "OCV[V]")

    ocv_temps = np.unique(ocvdf["Temperature_degC"])

    socs = np.linspace(0, 1, 100)
    base_temp = 25 * np.ones_like(socs)

    temps = np.linspace(5, 40, 100)
    base_soc = 0.5 * np.ones_like(temps)

    tab10 = mpl.colormaps["tab10"]
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    axocv = fig.add_subplot(gs[0, :])
    for i, temp in enumerate(ocv_temps):
        axocv.plot(
            socs,
            ocv_interp(temp * np.ones_like(socs), socs),
            label=f"{temp} $^\\circ$C",
            color=tab10(9-i),
        )
    axocv.set_xlabel("State of charge")
    axocv.set_ylabel("Open-circuit voltage $v_\\mathrm{oc}$ [V]")
    axocv.set_title("$ v_\\mathrm{oc}(SOC, T)$", loc="left", y=0.75)
    axocv.legend(frameon=False, bbox_to_anchor=[1, 1, 0.22, 0])

    axr0soc = fig.add_subplot(gs[1, 0])
    axr0soc.plot(socs, r_interps[0](base_temp, socs), color=tab10(0))
    axr0soc.plot(socs, r_interps[1](base_temp, socs), color=tab10(1))
    axr0soc.plot(socs, r_interps[2](base_temp, socs), color=tab10(2))
    axr0soc.set_title("$R_i(SOC, T=25^\\circ\\mathrm{C}) $", loc="right", y=0.75)
    axr0soc.set_ylabel("Resistance [$\\Omega$]")

    axtau1soc = fig.add_subplot(gs[2, 0])
    axtau1soc.plot(socs, tau_interps[0](base_temp, socs), color=tab10(1))
    axtau1soc.plot(socs, tau_interps[1](base_temp, socs), color=tab10(2))
    axtau1soc.set_title("$\\tau_i(SOC, T=25^\\circ\\mathrm{C}) $", loc="right", y=0.75)
    axtau1soc.set_xlabel("State of charge")
    axtau1soc.set_ylabel("Time constant [s]")

    axr0temp = fig.add_subplot(gs[1, 1])
    axr0temp.plot(temps, r_interps[0](temps, base_soc), label="$R_0$", color=tab10(0))
    axr0temp.plot(temps, r_interps[1](temps, base_soc), label="$R_1$", color=tab10(1))
    axr0temp.plot(temps, r_interps[2](temps, base_soc), label="$R_2$", color=tab10(2))
    axr0temp.set_title("$R_i(SOC=0.5, T) $", loc="right", y=0.75)
    axr0temp.legend(frameon=False, bbox_to_anchor=[1, 1, 0.35, 0])

    axtau1temp = fig.add_subplot(gs[2, 1])
    axtau1temp.plot(
        temps, tau_interps[0](temps, base_soc), label="$\\tau_1$", color=tab10(1)
    )
    axtau1temp.plot(
        temps, tau_interps[1](temps, base_soc), label="$\\tau_2$", color=tab10(2)
    )
    axtau1temp.legend(frameon=False, bbox_to_anchor=[1, 1, 0.35, 0])
    axtau1temp.set_title("$\\tau_i(SOC=0.5, T) $", loc="right", y=0.75)
    axtau1temp.set_xlabel("Temperature [$^\\circ$C]")

    axtau1soc.set_xlim(axr0soc.get_xlim())
    axtau1temp.set_xlim(axr0temp.get_xlim())

    axtau1temp.set_ylim(axtau1soc.get_ylim())
    axr0soc.set_ylim(axr0temp.get_ylim())

    axr0soc.set_xticklabels([])
    axr0temp.set_xticklabels([])

    axr0temp.set_yticklabels([])
    axtau1temp.set_yticklabels([])

    if plot_file:
        plt.savefig(plot_file)
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_parameters.py parfile ocvfile [plotfile]")
    main(*sys.argv[1:])
