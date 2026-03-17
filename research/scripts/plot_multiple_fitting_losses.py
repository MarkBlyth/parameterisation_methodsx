#!/usr/bin/env python3

from typing import List
import sys
import csv
import utils


def main(datafile_1: str, datafile_2: str, plot_file: str | None = None):
    labels = ["Fitting to $I\\neq0$", "Fitting to whole pulse"]
    ax, tidy_up = utils.get_ax(bool(plot_file), bottom_extra=0.35, l_margin=2.05)
    for f, label in zip((datafile_1, datafile_2), labels):
        with open(f) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            rows = [row for row in reader]
        socs, losses = rows
        #ax.bar(socs, [l * 1000 for l in losses], width=0.8 * (1 / (len(socs) + 1)), label=label)
        ax.bar(socs, [l * 1000 for l in losses], width=0.4 * (1 / (len(socs) + 1)), label=label)
    ax.set_xlabel("Average state-of-charge across pulse")
    ax.set_ylabel("RMS fitting error [mV]")
    ax.legend(frameon=False)
    tidy_up(plot_file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_fitting_loss.py lossfile [plotfile]")
    main(*sys.argv[1:])
