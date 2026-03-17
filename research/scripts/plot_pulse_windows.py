#!/usr/bin/env python3

from typing import List
import sys
import utils
import numpy as np


def main(cyclerfile: str, capacity_Ah: str, plot_file: str | None = None):
    headers = utils.BasytecHeaders
    df = utils.import_basytec(cyclerfile)
    socs = utils.coulomb_count(
        df[headers.time], df[headers.current], float(capacity_Ah)
    )
    pulses = utils.get_pulse_data(df, socs, headers)

    vs0 = pulses[0].vs
    vs1 = pulses[1].vs
    is0 = pulses[0].currents
    is1 = pulses[1].currents
    ts0 = pulses[0].ts
    ts1 = pulses[1].ts

    restmask = ((ts0 - ts0[0]) / (ts0[-1] - ts0[0])) > 0.95
    pulsemask = ((ts1 - ts1[0]) / (ts1[-1] - ts1[0])) < 0.15

    ts = np.hstack((ts0[restmask], ts1[pulsemask]))
    ts -= ts[0]
    vs = np.hstack((vs0[restmask], vs1[pulsemask]))
    currents = np.hstack((is0[restmask], is1[pulsemask]))

    currentmask = currents != 0
    start_of_current = ts[currentmask][0]

    x0 = ts[currentmask][0]
    x1 = ts[currentmask][-1]
    vs_mask = np.logical_and(currentmask, ts <= 0.5 * (x0 + x1))

    (ax, ax2), tidy_up = utils.get_ax(bool(plot_file), 2, bottom_extra=0.4)

    ax.plot(ts, currents, color=utils.current_colour, alpha=0.25, linestyle="--")
    ax2.plot(ts, vs, color=utils.voltage_colour, alpha=0.25, linestyle="--")

    ax.plot(ts[currentmask][1:], currents[currentmask][1:], color=utils.current_colour)
    ax2.plot(ts[vs_mask], vs[vs_mask], color=utils.voltage_colour)

    y_ax = np.mean(vs)
    y_ax2 = np.mean(currents)
    ax.annotate(
        "$I\\neq0$",
        xy=(x0, y_ax2),
        xytext=(x1, y_ax2),
        arrowprops=dict(arrowstyle="<->"),
        verticalalignment="center",
    )
    ax2.annotate(
        "$\\Delta t$",
        xy=(x0, y_ax),
        xytext=((x0 + 2 * x1) / 3, y_ax),
        arrowprops=dict(arrowstyle="<->"),
        verticalalignment="center",
    )

    x = 0.75 * start_of_current
    y0_ax = 0
    y1_ax = np.max(currents)
    y0_ax2 = np.max(vs)
    y1_ax2 = vs[currentmask][1]
    ax.annotate(
        "",
        xy=(x, y0_ax),
        xytext=(x, y1_ax),
        arrowprops=dict(arrowstyle="<->"),
        horizontalalignment="center",
    )
    ax.text(
        x,
        0.5 * (y0_ax + y1_ax),
        "$\\Delta I$ ",
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax2.annotate(
        "",
        xy=(x, y0_ax2),
        xytext=(x, y1_ax2),
        arrowprops=dict(arrowstyle="<->"),
        horizontalalignment="center",
    )
    ax2.text(
        x,
        0.5 * (y0_ax2 + y1_ax2),
        "$\\Delta v$ ",
        horizontalalignment="right",
        verticalalignment="center",
    )

    ax2.set_xticks(0.5 * start_of_current * np.arange(9))
    ax2.set_xticklabels(["", "", "$t_0$"] + [""] * 6)
    ax2.set_yticklabels([])
    ax.set_xticks(0.5 * start_of_current * np.arange(9))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax2.set_xlabel("Time")
    ax.set_ylabel("Current")
    ax2.set_ylabel("Cell voltage")
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_pulse_windows.py cyclerfile capacity_Ah [plotfile]")
    main(*sys.argv[1:])
