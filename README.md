[![DOI](https://zenodo.org/badge/1184463962.svg)](https://doi.org/10.5281/zenodo.19072963)

# Contents

* [License](#license)
* [Project structure](#project-structure)
* [Using the provided model and parameters](#using-the-provided-model-and-parameters)
* [Running your own parameterisation](#running-your-own-parameterisation)
* [Reproducing the paper results](#reproducing-the-paper-results)

The repo contains everything needed to reproduce the results in the MethodsX paper [*Mark Blyth, Amey Gupta, Alastair Hales: How to parameterise an equivalent-circuit empirical battery model from time-domain data (2026)*](https://www.sciencedirect.com/science/article/pii/S221501612600083X). The following sections explain how the project is licensed, structured, and how to use the code. Read the project structure first, then jump from there to whatever interests you!


# License
This work is released under the GNU GPL v3 license. Loosely, this means the software is free and user-modifiable, however any derivatives must be released under the same license. See LICENSE for more details.


# Project structure

This project uses a build-system so that all the results from the paper can be reproduced automatically. A full overview of how to run this is given in the section [Reproducing the paper results](#reproducing-the-paper-results).

Lab data are stored in the `research/lab_data/` directory, which contain battery cycler files from an LG M50 and NMC 622 pouch cell. Scripts for running the parameterisation and reproducing the paper's figures are in `research/scripts/`. The full process is managed automatically by `dodo.py`, which oversees the running of parameterisation scripts, and plotting of figures. Plotted figures get saved into `research/figures/` as *pgf* files, so `dodo.py` also copies them into `doc/figs/` and converts them to *png*, *svg*, and *pdf* files.

Probably you're here either to do your own parameterising, or to reuse the provided model. The next sections explain how to do this. Modelling and parameterisation codes are all in the `research/scripts/` directory, and parameters are shipped in `research/processed_data/`. To make life easier, the directory `try_me/` contains links to the model, parameters, and parameterisation script (note that these are symlinks, which might not play very nicely with Windows; in that case, you'll have to track down the files in the `research/` directories instead!).

*If you are on Windows, the links won't work, so you will need to copy the parameter and script files into your current working directory first.*

# Using the provided model and parameters
## Quick start

An example script, [`try_me/model_demo.py`](https://github.com/MarkBlyth/parameterisation_methodsx/blob/main/try_me/model_demo.py), shows an example of how to use the model for an isothermal simulation. Parameters are loaded in, and a WLTP current function is defined. Voltage is plotted as a function of time. The script can be run from any standard python environment. Pip or Conda can be used to set up the python environment; see the [Reproducing the paper results](#reproducing-the-paper-results) section for details on how to do this.

*As before, if you are on Windows, the links won't work, so you will need to copy the parameter and script files into your current working directory first.*

## Full description

An equivalent-circuit model and parameters are included with the publication. Parameters are stored in `research/processed_data/MLP001_params.csv` and `research/processed_data/MLP001_ocv.csv`, with links in the `try_me/` directory; the model is found in `research/scripts/model.py`, linked in `try_me/`. The parameters are for a 2.2 Ah NMC622/Graphite pouch cell, as [parameterised in the paper](https://www.sciencedirect.com/science/article/pii/S221501612600083X) with these scripts. The model is a 2-RC Thevenin model which takes a current profile, and predicts voltage and temperature.

`research/scripts/model.py` (also linked to `try_me/model.py`) provides a thermally coupled $n$ RC model in the `TheveninModel` class. Model parameters are interpolated with respect to state-of-charge (`soc`) and temperature; following literature conventions, we call these lookup tables, or luts. The `__init__` of `TheveninModel` loads in parameters from a CSV file, and builds parameter interpolations for use later. The class also defines a set of differential equations, available from `get_ode_rhs` ('get the right-hand side of the ordinary differential equations') which define the Thevenin model. These are given by
```math
    \begin{align}
        \frac{\mathrm{d}}{\mathrm{d} t} SOC &= \frac{I(t)}{Q_\mathrm{nom}}~,\\
        \frac{\mathrm{d} v_{\mathrm{rc}_i}}{\mathrm{d} t}  &= \frac{1}{C_i(T, SOC)}\left(I(t) - \frac{v_{\mathrm{rc}_i}}{R_i(T, SOC)} \right)~,\\
        v_\mathrm{batt}(t) &= v_\mathrm{oc}(T, SOC) + I(t)R_0(T, SOC) + \sum_{i=1}^{n_\mathrm{rc}} v_{\mathrm{rc}_i}, \\
        \frac{\mathrm{d} T}{\mathrm{d} t} &= \frac{1}{c}\left( q(t) - h(T - T_\infty) \right),\\
        q(t) &= I^2(t) R_0(T, SOC) + \sum_{i=1}^{n_\mathrm{rc}} \frac{v^2_{\mathrm{rc}_i}}{R_i(T, SOC)} - I(t) T(t) \frac{\partial v_\mathrm{oc}}{\partial T}
    \end{align}
```
(see [one of](https://www.sciencedirect.com/science/article/pii/S2352152X25035698) [our papers](https://www.sciencedirect.com/science/article/pii/S221501612600083X) for a full definition of all the variables!).
The model is evaluated by calling the `simulate` function
```python
    TheveninModel.simulate(
        currentfunc: Callable[[float], float],
        initial_soc: float,
        temp_inf: float,
        t_max: float,
        initial_temp: float | None = None,
        initial_vrcs: None | list | np.ndarray = None,
        **solver_kwargs,
    )
```
where `initial_soc` is the starting state-of-charge; `temp_inf` is the far-field (ambient) temperature for the thermal model; `t_max` is the time to stop integrating at; `initial_temp` is the initial cell temperature, and defaults to `temp_inf` if not set; `initial_vrcs` is the initial overpotential over the RC pairs, and defaults to zero; and `solver_kwargs` are keyword arguments passed to the ode solver.

The interpolations are not allowed to extrapolate (by design, so that the model does not go outside its parameterised regime). That means the simulation will fail if the state-of-charge or temperature leave the parameterised range. To fix this, choose the simulation stop-time `t_max` so that the simulation is finished before the model gets forced to extrapolate.

The model includes a lumped heat equation. By default, this is switched off - the cell has a thermal mass $m \rho c_p$ of infinity, so that its temperature never changes - however it can be activated by setting `thermal_mass` and `convection_rate` [W K$^{-1}$] when constructing the model.


# Running your own parameterisation

A minimal parameterisation script is provided in `try_me/minimal_parameteriser.py`. This loads in a file from a GITT test on a Neware battery cycler, then uses [PyBOP](https://pybop-docs.readthedocs.io/en/latest/) to find a set of model parameters. The bulk of the parameterisation work is handled by `try_me/fitter.py`, which is a link to `research/scripts/fitter.py`. This script is a bit messy because it's trying to be pretty general. A good starting place is to look at the PyBOP examples...
* [Empirical models with SciPy Minimize](https://github.com/pybop-team/PyBOP/blob/26.3/examples/notebooks/battery_parameterisation/ecm_scipy_constraints.ipynb); a Jupyter notebook which gives an overview of using PyBOP. This is useful for getting an idea of how PyBOP problems are structured, but really we want to fit timescales directly, which is done in...
* [Timescale fitting for empirical models](https://github.com/pybop-team/PyBOP/blob/26.3/examples/scripts/battery_parameterisation/ecm_tau_redefined.py); this is a script example, which demonstrates the method used in this work for defining and optimising model timescales.
Once these are familiar, `fitter.py` will start to make a bit more sense.

Pip or Conda can be used to set up the python environment; see the [Reproducing the paper results](#reproducing-the-paper-results) section for details on how to do this.



# Reproducing the paper results
This work was built using the [`doit`](https://pydoit.org/index.html) automation tool. Parameterisation codes are written in Python, and dependencies are managed with a Conda environment. Therefore, if you are on Linux, all you'll need to do is clone the repo, launch conda, and type doit. Conda will handle all the python packages, and doit will run the parameterisation scripts and build the figures for you.

If you are not in Linux, you can still reproduce the results either by running [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install), or using a Mac terminal.

*Note that since the paper was produced, the codebase has been updated to use the latest version of PyBOP; this will help in reusing the code, but might change the results ever so slightly!*

## Requirements

+ `bash` environment; either use Linux, a Mac terminal, or run [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)
+ A LaTeX distribution; this is used to convert the figures from *pgf* files, used for typesetting, to *png*, *svg*, and *pdf* files for viewing
+ inkscape; as above, used for turning *pdf* images into *png* and *svg*
+ [Optional] Conda; used to automatically manage python dependencies

## Steps

Clone:

```bash
git clone https://github.com/MarkBlyth/parameterisation_methodsx.git
```

Enter the directory:

```bash
cd parameterisation_methodsx
```

Create and activate a Conda environment:

```bash
conda env create -f environment.yml
conda activate parameterisation_methodsx
```

Manually install PyBOP and PyBaMM. This is a bit messy, but needs to be done because PyBOP isn't in conda forge (yet), and installing PyBaMM with conda would cause a Casadi conflict with PyBOP.

```bash
pip install pybop pybamm
```

Run doit:

```bash
doit
```

This will run the full parameterisation, and generate every figure from the paper. Figures will be put into the `doc/figs` directory, in *png*, *svg*, and *pdf* form. Copies will also be saved as a *pgf* in the research directory.

## Running without Conda

The scripts can be run using `doit` by manually installing the dependencies (if they are not already installed):

```bash
python3 -m pip install doit scipy numpy matplotlib pandas pybop pybamm openpyxl pytz
```

Then proceed as before, to clone and run `doit`:

```bash
git clone https://github.com/MarkBlyth/parameterisation_methodsx.git
cd parameterisation_methodsx
doit
```
