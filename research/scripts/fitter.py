from typing import Tuple, Dict
import warnings
from dataclasses import dataclass
import collections
import copy
import numpy as np
import pandas as pd
import pybop
import pybamm


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

BASE_PARAMETER_SET = {
    "chemistry": "ecm",
    "Initial temperature [K]": 25 + 273.15,
    "Upper voltage cut-off [V]": np.inf,
    "Lower voltage cut-off [V]": -np.inf,
    "Nominal cell capacity [A.h]": None,
    "Ambient temperature [K]": 25 + 273.15,
    "Current [A]": None,
    "R0 [Ohm]": None,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "Entropic change [V/K]": 0.0004,
    "Initial SoC": None,
}


def get_initial_parameters(
    capacity_Ah: float,
    fitting_parameters: list[pybop.Parameter],
    ocv_func: pybamm.Interpolant,
    n_rc: int,
) -> pybamm.ParameterValues:
    """
    Combine the BASE_PARAMETER_SET (bits PyBOP needs to see in order
    to work) with information we already know (capacity, OCV, number
    of RC pairs to fit), and the parameters that we don't yet know,
    but want to fit. This returns a PyBaMM parameter values object
    that can then be dropped into the model we're parameterising.
    """
    # Add the parameters we know
    pars = copy.deepcopy(BASE_PARAMETER_SET)
    pars["Cell capacity [A.h]"] = capacity_Ah
    pars["Open-circuit voltage [V]"] = ocv_func

    # Set up RC parameters, and give each RC pair a zero initial
    # overpotential
    for i in range(n_rc):
        pars[f"Element-{i+1} initial overpotential [V]"] = 0
        # These will be overwritten during fitting
        if f"R{i+1} [Ohm]" not in pars:
            pars[f"R{i+1} [Ohm]"] = 0.0002
        if f"C{i+1} [F]" not in pars:
            pars[f"C{i+1} [F]"] = 1000

    # For each timescale we fit, update the parameters to define it in
    # terms of the R and C that the PyBOP model understands
    for par in fitting_parameters:
        if "tau" in par:
            tau_idx = par[3]
            pars.update(
                {
                    f"tau{tau_idx} [s]": pars[f"R{tau_idx} [Ohm]"]
                    * pars[f"C{tau_idx} [F]"],
                },
                check_already_exists=False,
            )
            pars.update(
                {
                    f"C{tau_idx} [F]": pybamm.Parameter(f"tau{tau_idx} [s]")
                    / pybamm.Parameter(f"R{tau_idx} [Ohm]"),
                }
            )
    par_set = pybamm.ParameterValues(pars)
    return par_set


def coulomb_count(
    ts: np.ndarray,
    currents: np.ndarray,
    capacity: float,
    initial_soc: float = 1,
) -> np.ndarray:
    """
    Find state-of-charge throughout an experiment
    """
    if currents.shape != ts.shape:
        raise ValueError("Current and ts must have same shape")
    ret = np.zeros_like(currents)
    ret[0] = 0
    ret[1:] = np.diff(ts) * currents[:-1]
    return np.cumsum(ret) / (capacity * 3600) + initial_soc


def build_ocv_interpolant(socs: np.ndarray, ocvs: np.ndarray) -> pybamm.Interpolant:
    idxs = np.argsort(socs)

    def ocv(soc):
        return pybamm.Interpolant(socs[idxs], ocvs[idxs], soc, "OCV(SOC)")

    return ocv


def fit_parameter_set(
    data: PulseDataset,
    model: pybamm.equivalent_circuit.Thevenin,
    parameters: pybamm.ParameterValues,
    maxiter=50,
    method=pybop.XNES,
) -> tuple[pybop.Result | None, pybop.Problem | None, float | None]:
    """
    Do a single parameter fit. Take the data from a single pulse, a
    model structure to fit, some parameter values and guesses to fit
    from, and an optimisation method. Run the optimisation, and return
    the solution, PyBOP problem, and final fitting cost.
    """
    dataset = pybop.Dataset(
        {
            "Time [s]": data.ts,
            "Current [A]": data.currents,
            "Voltage [V]": data.vs,
        }
    )
    simulator = pybop.pybamm.Simulator(model, parameters, protocol=dataset)
    simulator.debug_mode = True
    cost = pybop.RootMeanSquaredError(dataset)
    problem = pybop.Problem(simulator, cost)
    try:
        if isinstance(method, str):
            options = pybop.SciPyMinimizeOptions(method=method)
            optim = pybop.SciPyMinimize(problem, options=options)
        else:
            options = pybop.PintsOptions(max_iterations=maxiter)
            optim = method(problem, options=options)
        soln = optim.run()
    except ValueError as e:
        # Typically happens when a point is requested outside of the
        # specified bounds
        warnings.warn(f"Something went wrong: {e}")
        return None, None, None
    return soln, problem, soln.best_cost


def parameterise(
    datasets: PulseDataset | list[PulseDataset],
    base_parameterset: pybamm.ParameterValues,
    fitting_parameters: Dict[str, pybop.Parameter],
    n_rc: int,
    maxiter=250,
    method=pybop.XNES,
    verbose=True,
    plot=True,
):
    """
    Take a set of pulse data, base parameters, and fitting parameters.
    For each pulse, try a range of optimisation methods. Record which
    method gives us the best results. Record the results in various
    forms (raw data, PyBaMM parameters object, optimiser solutions)
    and return these. Often only the parameters dataframe is of
    interest.
    """
    base_parameterset = copy.deepcopy(base_parameterset)

    datasets = [datasets] if isinstance(datasets, PulseDataset) else datasets
    if isinstance(method, str) or not hasattr(method, "__len__"):
        methodlist = [method]
    else:
        methodlist = method
    methodcounts = {m: 0 for m in methodlist}

    model = pybamm.equivalent_circuit.Thevenin(
        options={"number of rc elements": n_rc},
    )
    # PyBaMM Thevenin requires SOC < 1, which means things won't work
    # if initial_soc=1. We get around this by simply removing the
    # check on max. SOC.
    model.events = [e for e in model.events if e.name != "Maximum SoC"]

    fitting_solns = []
    full_parameter_sets = []
    average_socs = []
    for i, dataset in enumerate(datasets):
        set_initial_parameter_values(
            fitting_solns,
            fitting_parameters,
        )
        base_parameterset.update(fitting_parameters, check_already_exists=False)
        base_parameterset["Initial SoC"] = dataset.socs[0]
        best_cost = np.inf
        best_cost, best_pars, best_problem, best_method = np.inf, None, None, None
        for thismethod in methodlist:
            fitted, problem, finalcost = fit_parameter_set(
                dataset,
                model,
                base_parameterset,
                maxiter,
                thismethod,
            )
            if fitted is None:
                continue
            if finalcost < best_cost:
                best_cost = finalcost
                best_pars = fitted
                best_problem = problem
                best_method = thismethod
        if best_pars is None:
            continue
        fitting_solns.append(best_pars)
        average_socs.append(np.mean(dataset.socs))
        methodcounts[best_method] += 1

        full_fitted_pars = copy.deepcopy(base_parameterset)
        full_fitted_pars.update(best_pars.best_inputs, check_already_exists=False)
        full_parameter_sets.append(full_fitted_pars)

        if verbose:
            for key in best_pars.best_inputs:
                print(key, ":", best_pars.best_inputs[key])
            print(f"Best method: {best_method}")
            print(f"Final cost: {best_cost}\n")
        if plot:
            pybop.plot.problem(best_problem, inputs=best_pars.best_inputs)

    names = fitting_parameters.keys()
    ret_df = pd.DataFrame([soln.best_inputs for soln in fitting_solns], columns=names)
    ret_df.insert(0, "SOC", average_socs)

    if verbose and len(methodcounts) > 1:
        print("Finished parameterising; best optimisation methods:\n", methodcounts)

    return ret_df, full_parameter_sets, fitting_solns


def set_initial_parameter_values(parameterisation_results, fitting_parameters):
    """
    We need to supply an initial guess for the parameter values. Where
    a previous pulse has been parameterised, use those results as a
    guess for the next pulse. Otherwise, change nothing, because the
    user will have already provided a parameter guess.
    """
    if len(parameterisation_results) == 0:
        return
    recent_parameters = parameterisation_results[-1]
    new_fitting_parameters = []
    for parname in fitting_parameters:
        fitting_parameters[parname].update_initial_value(
            recent_parameters.best_inputs[parname]
        )
        # fitting_parameters[parname].prior.loc = recent_parameters.best_inputs[parname]


def get_pulse_data(
    df: pd.DataFrame,
    socs: np.ndarray,
    headers: Headers,
    direction: str,
    ignore_rests: bool = False,
) -> list[PulseDataset]:
    """
    Split the data up into pulses, based on when the cycler
    transitions between resting and (dis)charging. Tidy up the data by
    removing duplicate timestamps. Build a set of pulse datasets from
    each pulse, which can then be parameterised.
    """
    if direction not in ["charge", "discharge"]:
        raise ValueError(f"direction must be charge or discharge; received {direction}")
    active_command = headers.charging if direction == "charge" else headers.discharging
    wrong_command = headers.discharging if direction == "charge" else headers.charging
    warnings.warn("get_pulse_data may miss out the last pulse in a GITT")

    end_of_rests = df[
        df[headers.command].eq(headers.resting)
        & df.shift(-1)[headers.command].eq(active_command)
    ]
    ret = []
    for start, end in zip(end_of_rests.index, end_of_rests.index[1:]):
        pulse_df = df.iloc[start:end]
        if ignore_rests:
            pulse_df = pulse_df[pulse_df[headers.command].ne(headers.resting)]
        soclist = socs[pulse_df.index]

        unique_times = np.r_[True, np.diff(pulse_df[headers.time]) != 0]
        if not all(unique_times):
            warnings.warn(
                f"Skipping double-counted time-sample \n{pulse_df[np.logical_not(unique_times)]}"
            )
            pulse_df = pulse_df[unique_times]
            soclist = soclist[unique_times]

        ts = pulse_df[headers.time].to_numpy()
        dataset = PulseDataset(
            ts - ts[0],
            pulse_df[headers.voltage].to_numpy(),
            soclist,
            -pulse_df[headers.current].to_numpy(),
        )
        ret.append(dataset)
    return ret


def get_ocvs_from_pulsedataset_list(
    pulses: list[PulseDataset],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Record the final voltage from each pulse, and return as OCV.
    """
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
