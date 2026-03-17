import collections
from typing import List
import csv
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@dataclass
class Headers:
    command: str
    time: str
    current: str
    voltage: str
    charging: str
    discharging: str
    resting: str


PulseDataset = collections.namedtuple("PulseDataset", ["ts", "vs", "socs", "currents"])
BasytecHeaders = Headers(
    "Command", "Time[s]", "I[A]", "U[V]", "Charge", "Discharge", "Pause"
)

param_colour = "tab:purple"
soc_colour = "red"
current_colour = "green"
voltage_colour = "k"
error_colour = "tab:brown"
scatter_kwargs = {"marker": "x", "color": "red", "s": 100}


def _get_header_line_number(filename: str) -> int:
    with open(filename, "r", encoding="utf8", errors="ignore") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if row[0][0] != "~":
                return max(i - 1, 0)
    return 0


def import_basytec(filename: str) -> pd.DataFrame:
    header_line = _get_header_line_number(filename)
    return pd.read_csv(filename, header=header_line, encoding="unicode_escape")


def coulomb_count(
    ts: np.ndarray,
    currents: np.ndarray,
    capacity: float,
    initial_soc: float = 1,
) -> np.ndarray:
    if currents.shape != ts.shape:
        raise ValueError("Current and ts must have same shape")
    ret = np.zeros_like(currents)
    ret[0] = 0
    ret[1:] = np.diff(ts) * currents[:-1]
    return np.cumsum(ret) / (capacity * 3600) + initial_soc


def get_pulse_data(
    df: pd.DataFrame,
    socs: np.ndarray,
    headers: Headers,
    direction: str = "discharge",
    ignore_rests: bool = False,
    skip_initial_points: int = 0,
) -> List[PulseDataset]:
    if direction not in ["charge", "discharge", "switch"]:
        raise ValueError(
            f"direction must be charge, discharge, or switch; received {direction}"
        )
    active_command = headers.charging if direction == "charge" else headers.discharging
    wrong_command = headers.discharging if direction == "charge" else headers.charging

    end_of_rests = df[
        df[headers.command].eq(headers.resting)
        & df.shift(-1)[headers.command].eq(active_command)
    ]
    ret = []
    for start, end in zip(end_of_rests.index, end_of_rests.index[1:]):
        pulse_df = df.iloc[start:end]
        if any(pulse_df[headers.command].eq(wrong_command)):
            if direction != "switch":
                continue
            if not any(pulse_df[headers.command].eq(active_command)):
                continue
        elif direction == "switch":
            continue
        if ignore_rests:
            pulse_df = pulse_df[pulse_df[headers.command].ne(headers.resting)]
        soclist = socs[pulse_df.index]

        unique_times = np.r_[True, np.diff(pulse_df[headers.time]) != 0]
        if not all(unique_times):
            pulse_df = pulse_df[unique_times]
            soclist = soclist[unique_times]

        ts = pulse_df[headers.time].to_numpy()[skip_initial_points:]
        dataset = PulseDataset(
            ts,
            pulse_df[headers.voltage].to_numpy()[skip_initial_points:],
            soclist[skip_initial_points:],
            -pulse_df[headers.current].to_numpy()[skip_initial_points:],
        )
        ret.append(dataset)
    return ret


def get_ocvs_from_df(
    df: pd.DataFrame,
    socs: np.ndarray,
    headers: Headers,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    active_command = headers.charging if direction == "charge" else headers.discharging
    end_of_rests = df[
        df[headers.command].eq(headers.resting)
        & df.shift(-1)[headers.command].eq(active_command)
    ]
    vs = end_of_rests[headers.voltage].to_numpy()
    return socs[end_of_rests.index], vs


def get_ocvs_from_pulsedataset_list(
    pulses: list[PulseDataset],
) -> tuple[np.ndarray, np.ndarray]:
    socs, vs = np.zeros(len(pulses) + 1), np.zeros(len(pulses) + 1)
    socs[0] = pulses[0].socs[0]
    vs[0] = pulses[0].vs[0]
    for i, pulse in enumerate(pulses):
        is_resting = pulse.currents == 0
        rest_soc = pulse.socs[is_resting][-1]
        rest_v = pulse.vs[is_resting][-1]
        socs[i + 1] = rest_soc
        vs[i + 1] = rest_v
    ordering = np.argsort(socs)
    return socs[ordering], vs[ordering]


def get_ax(save: bool = False, n_axes: int = 1, **ax_kwargs):
    if save:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    ax = subjectively_better_subplots(n_axes, **ax_kwargs)

    def tidy_up(filename: str | None = None):
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    return ax, tidy_up


def subjectively_better_subplots(
    nrows,
    subheight=10 / 3,
    subwidth=10,
    bottom_extra=0.2,
    vertical_padding=0.25,
    l_margin=1,
    r_margin=0.1,
    **kwargs,
):
    # Source - https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    # Retrieved 2025-12-08, License - CC BY-SA 4.0
    lm = l_margin / 2.54
    rm = r_margin / 2.54
    ax_padding = vertical_padding / 2.54
    a = subheight / 2.54
    w = subwidth / 2.54
    width = lm + w + rm
    height = nrows * (2 * ax_padding + a) + bottom_extra
    fig = plt.figure(figsize=(width, height), **kwargs)
    axarr = np.empty(nrows, dtype=object)
    for i in range(nrows):
        axarr[i] = fig.add_axes(
            [
                lm / width,
                (height - (i + 1) * (2 * ax_padding + a) + ax_padding) / height,
                w / width,
                a / height,
            ]
        )
    if nrows == 1:
        return axarr[0]
    return axarr
