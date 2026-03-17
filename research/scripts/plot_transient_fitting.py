#!/usr/bin/env python3

from typing import List
import sys
import numpy as np
import utils


def main(cyclerfile: str, capacity_Ah: str, plot_file: str | None = None):
    i = 10 # dR/dSOC bigger at lower SOC, so use later pulses
    headers = utils.BasytecHeaders
    df = utils.import_basytec(cyclerfile)
    socs = utils.coulomb_count(
        df[headers.time], df[headers.current], float(capacity_Ah)
    )
    pulses = utils.get_pulse_data(df, socs, headers)

    firstpulse_ts = pulses[i].ts
    restmask = (
        (firstpulse_ts - firstpulse_ts[0]) / (firstpulse_ts[-1] - firstpulse_ts[0])
    ) > 0.95
    firstpulse_ts = firstpulse_ts[restmask]
    firstpulse_vs = pulses[i].vs[restmask]
    firstpulse_socs = pulses[i].socs[restmask]
    firstpulse_currents = pulses[i].currents[restmask]

    secondpulse_ts = pulses[i+1].ts
    pulsemask = (
        (secondpulse_ts - secondpulse_ts[0]) / (secondpulse_ts[-1] - secondpulse_ts[0])
    ) < 0.15
    secondpulse_ts = secondpulse_ts[pulsemask]
    secondpulse_vs = pulses[i+1].vs[pulsemask]
    secondpulse_socs = pulses[i+1].socs[pulsemask]
    secondpulse_currents = pulses[i+1].currents[pulsemask]

    ts = np.concatenate((firstpulse_ts, secondpulse_ts))
    pulsesocs = np.concatenate((firstpulse_socs, secondpulse_socs))
    vs = np.concatenate((firstpulse_vs, secondpulse_vs))
    current = np.concatenate((firstpulse_currents, secondpulse_currents))

    currentdiff = np.diff(current)
    currentdiff[np.abs(currentdiff) < 1] = 0 # Remove small noise in I
    r0s = np.diff(vs) / currentdiff

    r0s_socs = pulsesocs[np.argwhere(np.isfinite(r0s))].squeeze()
    r0s = r0s[np.argwhere(np.isfinite(r0s))].squeeze()
    r0s_t = np.interp(pulsesocs, r0s_socs[::-1], r0s[::-1])

    ocv_samples_t = np.array([firstpulse_ts[-1], secondpulse_ts[-1]])
    ocv_samples_soc = np.array([firstpulse_socs[-1], secondpulse_socs[-1]])
    ocv_samples_v = np.array([firstpulse_vs[-1], secondpulse_vs[-1]])
    ocv_t = np.interp(pulsesocs, ocv_samples_soc[::-1], ocv_samples_v[::-1])

    transients = vs - ocv_t - current * r0s_t
    ir_drop = current * r0s_t
    imask = current != 0

    ax, tidy_up = utils.get_ax(bool(plot_file))
    ax.plot(
        ts,
        vs - ocv_t,
        color=utils.voltage_colour,
        alpha=0.25,
    )
    ax.plot(ts, ir_drop, zorder=-1, alpha=0.5, color="tab:blue")
    ax.plot(ts, transients, linestyle=(0, (4,4)), color="tab:orange", alpha=0.75)

    ax.plot(
        ts[imask],
        vs[imask] - ocv_t[imask],
        color=utils.voltage_colour,
        label="Total overpotential",
    )
    ax.plot(ts[imask], ir_drop[imask], label="$I R_0(SOC)$", zorder=-1, color="tab:blue")
    ax.plot(ts[imask], transients[imask], linestyle=(0, (2,4)), color="tab:orange")
    ax.plot(ts[imask], transients[imask], label="Transients", color="tab:orange")
    ax.legend(frameon=False)


    ax.set_xticklabels([])
    ax.set_xlabel("Time")
    ax.set_yticklabels([])
    ax.set_ylabel("Overpotentials")
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_gitt_ocv.py cyclerfile capacity_Ah [plotfile]")
    main(*sys.argv[1:])
