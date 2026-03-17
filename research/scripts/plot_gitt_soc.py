#!/usr/bin/env python3

from typing import List
import sys
import utils

def main(cyclerfile: str, capacity_Ah: str, plot_file: str | None = None):
    headers = utils.BasytecHeaders
    df = utils.import_basytec(cyclerfile)
    socs = utils.coulomb_count(df[headers.time], df[headers.current], float(capacity_Ah))
    pulses = utils.get_pulse_data(df, socs, headers)
    first_pulse = pulses[0]
    (ax, ax2), tidy_up = utils.get_ax(bool(plot_file), 2)

    ts = pulses[0].ts
    restmask = ((ts-ts[0]) / (ts[-1] - ts[0])) > 0.95
    ax.plot(pulses[0].ts[restmask], pulses[0].vs[restmask], color=utils.voltage_colour)
    ax2.plot(pulses[0].ts[restmask], pulses[0].socs[restmask], color=utils.soc_colour)

    ts = pulses[1].ts
    pulsemask = ((ts - ts[0]) / (ts[-1] - ts[0])) < 0.15
    ax.plot(pulses[1].ts[pulsemask], pulses[1].vs[pulsemask], color=utils.voltage_colour)
    ax2.plot(pulses[1].ts[pulsemask], pulses[1].socs[pulsemask], color=utils.soc_colour)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax.set_ylabel("Cell voltage")
    ax2.set_ylabel("State of charge")
    ax2.set_xlabel("Time")
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_gitt_soc.py cyclerfile capacity_Ah [plotfile]")
    main(*sys.argv[1:])
