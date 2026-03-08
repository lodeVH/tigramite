"""Run LPCMCI on a CSV dataset located next to this script.

Usage examples:
    python lpcmci_from_local_csv.py
    python lpcmci_from_local_csv.py --use-h false --tau-max 3 --pc-alpha 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

# Toggle this default if you want to include/exclude the H column by default.
USE_H = False


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
    csv_files = sorted(script_dir.glob("*.csv"))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LPCMCI on the single CSV file that is in the same directory as this script."
        )
    )
    parser.add_argument("--use-h", type=str2bool, default=USE_H, help="Include column H (true/false).")
    parser.add_argument("--tau-max", type=int, default=2, help="Maximum lag for LPCMCI.")
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
        help="Optional output path for the plot image (e.g. lpcmci_graph.png).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = find_single_csv(script_dir)

    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()

    columns = [column.strip() for column in header.split(",") if column.strip()]
    required_columns = ["T1", "T2", "H"]
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(
            f"CSV file {csv_path.name} is missing required columns: {missing_columns}. "
            "Expected columns: T1, T2, H."
        )

    selected_columns = ["T1", "T2", "H"] if args.use_h else ["T1", "T2"]
    usecols = [columns.index(column) for column in selected_columns]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=usecols, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    dataframe = pp.DataFrame(data=data, var_names=selected_columns)
    parcorr = ParCorr(significance="analytic")
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=args.verbosity)

    results = lpcmci.run_lpcmci(tau_max=args.tau_max, pc_alpha=args.pc_alpha)

    print(f"CSV used: {csv_path}")
    print(f"Variables used: {selected_columns}")
    print("\nLPCMCI graph shape:", results["graph"].shape)
    print("LPCMCI graph:")
    print(results["graph"])

    if args.plot:
        tp.plot_graph(graph=results["graph"], val_matrix=results["val_matrix"])
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {args.save_plot}")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
