"""Run LPCMCI on the CSTR example CSV and plot the learned graph.

This script expects exactly one CSV file in the same directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

USE_SETPOINTS = True


def str2bool(value: str) -> bool:
    """Parse a string into a boolean value for argparse."""
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def find_single_csv(script_dir: Path) -> Path:
    """Return the only CSV in script_dir or raise a helpful error."""
    csv_files = sorted(path for path in script_dir.glob("*.csv") if not path.name.endswith("_links.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {script_dir}. Place exactly one .csv file next to this script."
        )
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise RuntimeError(
            f"Expected exactly one CSV file in {script_dir}, found {len(csv_files)}: {names}"
        )
    return csv_files[0]


def save_links_csv(results: dict, var_names: list[str], output_path: Path) -> None:
    """Save all discovered links to CSV with link type and metrics."""
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "target",
                "lag",
                "p_value",
                "effect_size",
                "abs_effect_size",
                "connection_type",
                "connection_string",
            ]
        )

        n_vars, _, n_lags = graph.shape
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                for lag in range(n_lags):
                    connection_type = graph[source_idx, target_idx, lag]
                    if connection_type == "":
                        continue

                    effect_size = float(val_matrix[source_idx, target_idx, lag])
                    writer.writerow(
                        [
                            var_names[source_idx],
                            var_names[target_idx],
                            lag,
                            float(p_matrix[source_idx, target_idx, lag]),
                            effect_size,
                            abs(effect_size),
                            connection_type,
                            f"({source_idx},{-lag}) {connection_type} ({target_idx},0)",
                        ]
                    )


def save_all_links_csv(results: dict, var_names: list[str], output_path: Path, pc_alpha: float) -> None:
    """Save all tested links (selected and non-selected) to CSV."""
    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "target",
                "lag",
                "selected",
                "connection_type",
                "p_value",
                "effect_size",
                "abs_effect_size",
                "p_gt_pc_alpha",
                "selection_note",
                "connection_string",
            ]
        )

        n_vars, _, n_lags = graph.shape
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                for lag in range(n_lags):
                    connection_type = graph[source_idx, target_idx, lag]
                    p_value = float(p_matrix[source_idx, target_idx, lag])
                    effect_size = float(val_matrix[source_idx, target_idx, lag])
                    selected = connection_type != ""
                    p_gt_alpha = p_value > pc_alpha

                    if selected:
                        selection_note = "selected"
                    elif p_gt_alpha:
                        selection_note = "not_selected_p_value_gt_pc_alpha"
                    else:
                        selection_note = "not_selected_other_or_constrained"

                    writer.writerow(
                        [
                            var_names[source_idx],
                            var_names[target_idx],
                            lag,
                            selected,
                            connection_type,
                            p_value,
                            effect_size,
                            abs(effect_size),
                            p_gt_alpha,
                            selection_note,
                            f"({source_idx},{-lag}) {connection_type if connection_type else '[none]'} ({target_idx},0)",
                        ]
                    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LPCMCI on CSTR CSV data and plot the learned DPAG."
    )
    parser.add_argument(
        "--use-setpoints",
        type=str2bool,
        default=USE_SETPOINTS,
        help="Include cA0 and theta0 variables (true/false).",
    )
    parser.add_argument("--tau-max", type=int, default=3, help="Maximum lag for LPCMCI.")
    parser.add_argument("--pc-alpha", type=float, default=0.01, help="LPCMCI significance level.")
    parser.add_argument("--verbosity", type=int, default=1, help="LPCMCI verbosity level.")
    parser.add_argument(
        "--plot",
        type=str2bool,
        default=True,
        help="Plot the learned DPAG using tigramite.plotting.plot_graph (true/false).",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Optional output path for the plot image (e.g. cstr_lpcmci_graph.png).",
    )
    parser.add_argument(
        "--save-links",
        type=str,
        default="",
        help="Optional output path for discovered links CSV. Defaults next to script with suffix _links.csv.",
    )
    parser.add_argument(
        "--save-all-links",
        type=str,
        default="",
        help="Optional output path for all links CSV (selected + not selected). Defaults to suffix _all_links.csv.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = find_single_csv(script_dir)

    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    columns = [column.strip() for column in header.split(",") if column.strip()]
    required_columns = ["cA", "cB", "theta", "thetaK", "F", "QK", "cA0", "theta0"]
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(
            f"CSV file {csv_path.name} is missing required columns: {missing_columns}. "
            "Expected columns: cA,cB,theta,thetaK,F,QK,cA0,theta0."
        )

    selected_columns = ["cA", "cB", "theta", "thetaK", "F", "QK"]
    if args.use_setpoints:
        selected_columns += ["cA0", "theta0"]

    usecols = [columns.index(column) for column in selected_columns]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    dataframe = pp.DataFrame(data=data, var_names=selected_columns)
    parcorr = ParCorr(significance="analytic")
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=args.verbosity)

    results = lpcmci.run_lpcmci(
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
    )

    print(f"CSV used: {csv_path}")
    print(f"Variables used: {selected_columns}")
    print("\nLPCMCI graph shape:", results["graph"].shape)
    print("LPCMCI graph:")
    print(results["graph"])

    links_output_path = Path(args.save_links) if args.save_links else csv_path.with_name(f"{csv_path.stem}_links.csv")
    save_links_csv(results=results, var_names=selected_columns, output_path=links_output_path)
    print(f"Saved links CSV to: {links_output_path}")

    all_links_output_path = Path(args.save_all_links) if args.save_all_links else csv_path.with_name(f"{csv_path.stem}_all_links.csv")
    save_all_links_csv(
        results=results,
        var_names=selected_columns,
        output_path=all_links_output_path,
        pc_alpha=args.pc_alpha,
    )
    print(f"Saved all-links CSV to: {all_links_output_path}")

    if args.plot:
        tp.plot_graph(
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            var_names=selected_columns,
        )
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {args.save_plot}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
