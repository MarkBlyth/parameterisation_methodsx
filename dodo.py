#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path

RAW_DATA_DIR = Path("research/lab_data")
SCRIPTS_DIR = Path("research/scripts")
PROCESSED_DIR = Path("research/processed_data")
FIGURES_DIR = Path("research/figures")
PAPER_FIGURES_DIR = Path("doc/figs")
DOC_DIR = Path("doc")


def task_svg_to_pdf():
    for fig in ["circuit"]:
        figfile = str(PAPER_FIGURES_DIR / fig)
        yield {
            "name": fig,
            "actions": [f"inkscape {figfile+'.svg'} --export-pdf={figfile+'.pdf'} -z"],
            "file_dep": [figfile + ".svg"],
            "targets": [figfile + ".pdf"],
        }


def task_plot_figures():
    gitt_data = str(
        RAW_DATA_DIR
        / "2133Parameterisation_25degLGM50LT_HOT_1_END_14_11_2025_07_47_08.csv"
    )
    utils = str(SCRIPTS_DIR / "utils.py")
    for name in [
        "gitt_ocv",
        "gitt_soc",
        "nominal_capacity",
        "pulse_windows",
        "gitt_hyperparameters",
        "interpolations",
        "transient_fitting",
    ]:
        script = str(SCRIPTS_DIR / ("plot_" + name + ".py"))
        figname = FIGURES_DIR / (name + ".pgf")
        yield {
            "name": name,
            "actions": ["python " + " ".join((script, gitt_data, "5", str(figname)))],
            "file_dep": [script, utils, gitt_data],
            "targets": [figname],
        }


def task_plot_fitting_errors():
    script = str(SCRIPTS_DIR / "plot_multiple_fitting_losses.py")
    utils = str(SCRIPTS_DIR / "utils.py")
    loss_data_1 = str(PROCESSED_DIR / "mlp001_fitting_errors_25degC.csv")
    loss_data_2 = str(PROCESSED_DIR / "mlp001_full_pulse_fitting_errors_25degC.csv")
    figname = FIGURES_DIR / "fitting_loss.pgf"
    return {
        "actions": ["python " + " ".join((script, loss_data_1, loss_data_2, str(figname)))],
        "file_dep": [script, utils, loss_data_1, loss_data_2],
        "targets": [figname],
    }


def task_plot_wltp_validation():
    script = str(SCRIPTS_DIR / "plot_wltp_validation.py")
    utils = str(SCRIPTS_DIR / "utils.py")
    model = str(SCRIPTS_DIR / "model.py")
    param_file = str(PROCESSED_DIR / "mlp001_parameters_25degC.csv")
    ocv_file = str(PROCESSED_DIR / "mlp001_ocv_25degC.csv")
    cycler_file = str(RAW_DATA_DIR / "MLP001_wltp_25degC.xlsx")
    figname = FIGURES_DIR / "wltp_validation.pgf"
    return {
        "actions": [
            "python "
            + " ".join(
                (script, cycler_file, ocv_file, param_file, "2.132", str(figname))
            )
        ],
        "file_dep": [script, utils, model, param_file, ocv_file, cycler_file],
        "targets": [figname],
    }

def task_plot_parameters():
    script = str(SCRIPTS_DIR / "plot_parameters.py")
    param_file = str(PROCESSED_DIR / "MLP001_params.csv")
    ocv_file = str(PROCESSED_DIR / "MLP001_ocv.csv")
    figname = FIGURES_DIR / "parameters.pgf"
    return {
        "actions": [
            "python "
            + " ".join(
                (script, param_file, ocv_file, str(figname))
            )
        ],
        "file_dep": [script, param_file, ocv_file],
        "targets": [figname],
    }


def task_sync_figures_to_paper():
    fig_files = [action["targets"][0] for action in task_plot_figures()]
    fig_files.append(task_plot_fitting_errors()["targets"][0])
    fig_files.append(task_plot_wltp_validation()["targets"][0])
    fig_files.append(task_plot_parameters()["targets"][0])
    return {
        "actions": [f"rsync -avh {FIGURES_DIR}/ {PAPER_FIGURES_DIR}/"],
        "file_dep": fig_files,
        "targets": [PAPER_FIGURES_DIR / f.name for f in fig_files],
        "verbosity": 2,
    }


def task_pgf_to_pdf():
    all_synced_figs = task_sync_figures_to_paper()["targets"]
    pgf_targets = [fig for fig in all_synced_figs if fig.suffix == ".pgf"]
    for pgffile in pgf_targets:
        ret = {
            "name": pgffile.name,
            "actions": [f"{PAPER_FIGURES_DIR/'pgf2svg.sh'} {pgffile.stem}"],
            "file_dep": [pgffile, PAPER_FIGURES_DIR / "pgf2svg.sh"],
            "targets": [str(pgffile)[:-4] + ".svg"],
        }
        yield ret


def task_run_parameteriser_on_full_pulse():
    data_file = RAW_DATA_DIR / "MLP001_25degC.xlsx"
    par_file = PROCESSED_DIR / "mlp001_full_pulse_fitting_parameters_25degC.csv"
    ocv_file = PROCESSED_DIR / "mlp001_full_pulse_fitting_ocv_25degC.csv"
    mse_file = PROCESSED_DIR / "mlp001_full_pulse_fitting_errors_25degC.csv"
    return {
        "file_dep": [
            data_file,
            SCRIPTS_DIR / "fitter.py",
            SCRIPTS_DIR / "run_parameteriser_full_pulse.py",
        ],
        "targets": [par_file, ocv_file, mse_file],
        "actions": [
            f"python {SCRIPTS_DIR / 'run_parameteriser_full_pulse.py'} {data_file} -o {ocv_file} -p {par_file} -e {mse_file} -t 25"
        ],
        "verbosity": 2,
    }

def task_run_parameteriser():
    for temperature in [5, 10, 15, 25, 40]:
        data_file = RAW_DATA_DIR / f"MLP001_{temperature}degC.xlsx"
        par_file = PROCESSED_DIR / f"mlp001_parameters_{temperature}degC.csv"
        ocv_file = PROCESSED_DIR / f"mlp001_ocv_{temperature}degC.csv"
        mse_file = PROCESSED_DIR / f"mlp001_fitting_errors_{temperature}degC.csv"
        yield {
            "name": data_file.stem,
            "file_dep": [
                data_file,
                SCRIPTS_DIR / "fitter.py",
                SCRIPTS_DIR / "run_parameteriser.py",
            ],
            "targets": [par_file, ocv_file, mse_file],
            "actions": [
                f"python {SCRIPTS_DIR / 'run_parameteriser.py'} {data_file} -o {ocv_file} -p {par_file} -e {mse_file} -t {temperature}"
            ],
            "verbosity": 2,
        }

def task_combine_parameter_files():
    paramfiles = [str(task["targets"][0]) for task in task_run_parameteriser()]
    outfile = PROCESSED_DIR / "MLP001_params.csv"
    return {
        "file_dep": paramfiles + [SCRIPTS_DIR / "combine_paramfiles.sh"],
        "targets": [outfile],
        "actions": [f"bash {str(SCRIPTS_DIR / 'combine_paramfiles.sh')} {' '.join(paramfiles)} > {str(outfile)}"]
        }


def task_combine_ocv_files():
    paramfiles = [str(task["targets"][1]) for task in task_run_parameteriser()]
    outfile = PROCESSED_DIR / "MLP001_ocv.csv"
    return {
        "file_dep": paramfiles + [SCRIPTS_DIR / "combine_paramfiles.sh"],
        "targets": [outfile],
        "actions": [f"bash {str(SCRIPTS_DIR / 'combine_paramfiles.sh')} {' '.join(paramfiles)} > {str(outfile)}"]
        }


# def task_compile_doc():
#     fig_deps = task_sync_figures_to_paper()["targets"]
#     fig_deps += [task["targets"][0] for task in task_svg_to_pdf()]
#     return {
#         "actions": [
#             f"cd {DOC_DIR} && pdflatex -shell-escape main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && cd .."
#         ],
#         "file_dep": [
#             DOC_DIR / "main.tex",
#             DOC_DIR / "method.tex",
#             DOC_DIR / "details.tex",
#             DOC_DIR / "validation.tex",
#             DOC_DIR / "limitations.tex",
#             DOC_DIR / "background.tex",
#             DOC_DIR / "conclusions.tex",
#             DOC_DIR / "references.bib",
#         ]
#         + fig_deps,
#         "targets": [DOC_DIR / "main.pdf"],
#         "verbosity": 2,
#     }
