"""Plot an LPCMCI graph at a chosen p-value threshold from *_all_links.csv.

Expected input CSV format is the one produced by lpcmci_cstr_from_csv.py via
save_all_links_csv(...), including columns:
source,target,lag,p_value,effect_size
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tigramite import plotting as tp


def str2bool(value: str) -> bool:
    """Parse a string into a boolean value for argparse."""
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def find_single_all_links_csv(script_dir: Path) -> Path:
    """Return the only *_all_links.csv in script_dir or raise a helpful error."""
    csv_files = sorted(script_dir.glob("*_all_links.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_all_links.csv found in {script_dir}. Run the LPCMCI CSV script first."
        )
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise RuntimeError(
            f"Expected exactly one *_all_links.csv in {script_dir}, found {len(csv_files)}: {names}"
        )
    return csv_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a graph from an all-links CSV at a chosen p-value threshold."
    )
    parser.add_argument(
        "--all-links-csv",
        type=str,
        default="",
        help="Optional path to *_all_links.csv. Default: the only *_all_links.csv next to this script.",
    )
    parser.add_argument("--pc-alpha", type=float, default=0.05, help="P-value threshold for selecting links.")
    parser.add_argument("--plot", type=str2bool, default=True, help="Show interactive plot (true/false).")
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Optional output image path (e.g. graph_from_all_links.png).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    all_links_path = Path(args.all_links_csv) if args.all_links_csv else find_single_all_links_csv(script_dir)

    rows = []
    var_names = []
    var_set = set()
    max_lag = 0

    with all_links_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "lag", "p_value", "effect_size"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV {all_links_path.name} missing required columns: {sorted(missing)}")

        for row in reader:
            source = row["source"]
            target = row["target"]
            lag = int(row["lag"])
            p_value = float(row["p_value"])
            effect_size = float(row["effect_size"])

            rows.append((source, target, lag, p_value, effect_size))
            if source not in var_set:
                var_set.add(source)
                var_names.append(source)
            if target not in var_set:
                var_set.add(target)
                var_names.append(target)
            if lag > max_lag:
                max_lag = lag

    if not rows:
        raise ValueError(f"CSV {all_links_path.name} has no data rows.")

    name_to_index = {name: idx for idx, name in enumerate(var_names)}
    n_vars = len(var_names)

    p_matrix = np.ones((n_vars, n_vars, max_lag + 1), dtype=float)
    val_matrix = np.zeros((n_vars, n_vars, max_lag + 1), dtype=float)

    for source, target, lag, p_value, effect_size in rows:
        i = name_to_index[source]
        j = name_to_index[target]
        p_matrix[i, j, lag] = p_value
        val_matrix[i, j, lag] = effect_size

    graph_bool = p_matrix <= args.pc_alpha

    # Ensure lag-0 consistency for plotting by symmetrizing contemporaneous links.
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            present = bool(graph_bool[i, j, 0] or graph_bool[j, i, 0])
            graph_bool[i, j, 0] = present
            graph_bool[j, i, 0] = present

    print(f"All-links CSV used: {all_links_path}")
    print(f"Variables: {var_names}")
    print(f"pc_alpha threshold: {args.pc_alpha}")

    if args.plot or args.save_plot:
        tp.plot_graph(graph=graph_bool, val_matrix=val_matrix, var_names=var_names)
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {args.save_plot}")
            plt.close()
        elif args.plot:
            plt.show()


if __name__ == "__main__":
    main()
