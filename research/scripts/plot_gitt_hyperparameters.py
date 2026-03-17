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

    is0 = pulses[0].currents
    is1 = pulses[1].currents
    ts0 = pulses[0].ts
    ts1 = pulses[1].ts

    restmask = ((ts0 - ts0[0]) / (ts0[-1] - ts0[0])) > 0.95
    pulsemask = ((ts1 - ts1[0]) / (ts1[-1] - ts1[0])) < 0.15

    ts = np.hstack((ts0[restmask], ts1[pulsemask]))
    ts -= ts[0]
    currents = np.hstack((is0[restmask], is1[pulsemask]))

    currentmask = currents != 0
    start_of_current = ts[currentmask][0]

    ax, tidy_up = utils.get_ax(bool(plot_file), l_margin=1.75)

    ax.plot(ts, currents, color=utils.current_colour)

    currentmask = currents != 0
    x0 = ts[currentmask][0]
    x1 = ts[currentmask][-1]
    y_ax = np.mean(currents)
    ax.annotate(
        "",
        xy=(x0, y_ax),
        xytext=(x1, y_ax),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax.text(
        0.5 * (x0 + x1),
        y_ax * 1.05,
        "$t_\\mathrm{pulse}$",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    ax.annotate(
        "",
        xy=(x1, y_ax),
        xytext=(ts[-1], y_ax),
        arrowprops=dict(arrowstyle="<->"),
        verticalalignment="center",
        horizontalalignment="right",
    )
    ax.text(
        0.5 * (x1 + ts[-1]),
        y_ax * 1.05,
        "$t_\\mathrm{rest}$",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    curr = np.max(currents)
    ax.set_xticklabels([])
    ax.set_yticks([0, 0.5 * curr, curr])
    ax.set_yticklabels(["0", "", "$I_\\mathrm{pulse}$"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Current")
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_pulse_windows.py cyclerfile capacity_Ah [plotfile]")
    main(*sys.argv[1:])
