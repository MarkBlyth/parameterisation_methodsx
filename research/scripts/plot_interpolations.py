#!/usr/bin/env python3

from typing import List
import sys
import utils
import scipy.interpolate
import numpy as np


def main(_: str, __: str, plot_file: str | None = None):

    xdata = [0.5, 1, 2, 3]
    ydata = [10, 2, 0.1, 0.2]

    xs = np.linspace(xdata[0], xdata[-1], 1000)
    ys_pchip = scipy.interpolate.PchipInterpolator(xdata, ydata)(xs)
    ys_spline = scipy.interpolate.CubicSpline(xdata, ydata)(xs)

    ax, tidy_up = utils.get_ax(bool(plot_file))
    ax.scatter(xdata, ydata, label="Parameter samples", **utils.scatter_kwargs)
    ax.plot(xs, ys_pchip, label="Monotone spline", color=utils.param_colour, linestyle=":")
    ax.plot(xs, ys_spline, label="Cubic spline", color=utils.param_colour)
    ax.legend()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
#    ax.set_xlabel("State-of-charge")
    ax.set_ylabel("Parameter")
    ax.legend(frameon=False)
    tidy_up(plot_file)



if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit()
    if sys.argv[1] == "-h":
        print("./plot_interpolations.py unused unused [plotfile]")
    main(*sys.argv[1:])
