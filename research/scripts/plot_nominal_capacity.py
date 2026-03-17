#!/usr/bin/env python3

from typing import List
import sys
import numpy as np
import utils


def main(cyclerfile: str, capacity_Ah: str, plot_file: str | None = None):
    headers = utils.BasytecHeaders
    df = utils.import_basytec(cyclerfile)
    throughput = 1 - utils.coulomb_count(
        df[headers.time],
        df[headers.current],
        1,
    )
    voltages = df[headers.voltage]
    min_voltage = np.argmin(voltages)
    capacity = throughput[min_voltage]

    ax, tidy_up = utils.get_ax(bool(plot_file), bottom_extra=0.4, l_margin=1.75)

    dydx = 0.2
    ax.annotate(
        "", xytext=(1, 3.6), xy=(3, 3.6 - 2 * dydx), arrowprops=dict(arrowstyle="->")
    )
    ax.annotate(
        "", xytext=(4, 3.9), xy=(3, 3.9 + dydx), arrowprops=dict(arrowstyle="->")
    )
    ax.text(1.6, 3.1, "Discharge")
    ax.text(3.9, 4, "Charge")
    ax.plot(throughput, voltages, color=utils.voltage_colour)
    ax.scatter(throughput[min_voltage], voltages[min_voltage], **utils.scatter_kwargs)

    ax.set_xticks([0, capacity / 3, 2*capacity/3, capacity])
    ax.set_xticklabels([0, "", "", "$Q_\\mathrm{nom}$"])
    ax.set_yticklabels([])
    ax.set_yticks([np.min(voltages), voltages[0]])
    ax.set_yticklabels(["$v_\\mathrm{min}$", "$v_\\mathrm{max}$"])
    ax.set_ylabel("Cell voltage")
    ax.set_xlabel("Capacity throughput [Ah]")
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_nominal_capacity.py cyclerfile capacity_Ah [plotfile]")
    main(*sys.argv[1:])
