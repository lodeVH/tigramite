from pathlib import Path

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import threading
import time
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI




# NumPy>=2.0 removed the ddof argument from np.corrcoef, while some Tigramite
# versions still call np.corrcoef(..., ddof=0) in shuffle significance internals.
# Patch only when needed so CMIknn(shuffle_test) remains usable across envs.
try:
    np.corrcoef(np.array([0.0, 1.0]), np.array([1.0, 2.0]), ddof=0)
except TypeError:
    _orig_corrcoef = np.corrcoef

    def _corrcoef_compat(*args, **kwargs):
        kwargs.pop("ddof", None)
        return _orig_corrcoef(*args, **kwargs)

    np.corrcoef = _corrcoef_compat

# ===========================================================
# User settings
# ===========================================================

# For deterministic/noiseless simulations, CMIknn is typically safer than GPDC.
use_test = "CMIknn"   # "CMIknn" or "GPDC"

# If you know the plant sample time, you can derive tau_max from dominant mode time constant.
sample_time_h = 0.01  # from data generation (0.01 h = 36 s)
sample_time_s = sample_time_h * 3600.0
# From your modal analysis screenshot, slowest mode tau ≈ 263.51 s.
dominant_tau_s = 263.51
settling_factor = 5       # 5*tau ~ 99% settling window

# Fallback tau settings (used when sample_time_s is None).
tau_min = 0
tau_max = 5

pc_alpha = 0.5    # significance level for PC phase; PCMCI+ uses this alpha for final link selection but not for PC phase pruning.
fdr_method = None  # None keeps theoretical links; use "fdr_bh" for noisy data.
pcmci_verbosity = 1  # >0 prints Tigramite progress details during PCMCI+

# CMIknn parameters (used only when use_test='CMIknn'):
# - significance='shuffle_test': p-values from permutation/shuffle significance testing.
# - knn: number of nearest neighbors for local CMI estimation; larger = smoother/less variance,
#   smaller = more local/more variance.
# - shuffle_neighbors: neighbors used in restricted shuffling during significance estimation;
#   should typically be in the same range as knn.
cmiknn_significance = "fixed_thres" # 'shuffle_test'
cmiknn_knn = 5
cmiknn_shuffle_neighbors = False #5

# Keep this False for perfect-theory runs; switch on only if CI estimation becomes singular.
add_tiny_jitter = False
# Keep False if you want to preserve physical units in outputs/plots.
standardize_data = False
jitter_std = 1e-8
random_seed = 7

# Optional: select warmup and used samples as percentages of the CSV length.
warmup_percent = 1.5
used_percent = 98.5

states = ["cA", "cB", "theta", "thetaK"]
inputs = ["F", "QK", "cA0", "theta0"]
var_names = states + inputs

# Keep paths robust when script is launched from another working directory.
DATA_DIR = Path(__file__).resolve().parent


def resolve_tau_max(default_tau_max: int) -> int:
    """Derive tau_max from process time scale when sample_time_s is provided."""
    if sample_time_s is None:
        return default_tau_max
    if sample_time_s <= 0:
        raise ValueError("sample_time_s must be > 0")

    settling_window_s = settling_factor * dominant_tau_s
    auto_tau_max = max(1, int(np.ceil(settling_window_s / sample_time_s)))
    print(
        f"Auto tau_max from dynamics: ceil({settling_factor}*{dominant_tau_s}/{sample_time_s}) = {auto_tau_max}"
    )
    return auto_tau_max


# ===========================================================
# Load and standardize data
# ===========================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {DATA_DIR}.")
if len(csv_files) > 1:
    raise ValueError(
        f"Expected exactly one CSV file in {DATA_DIR}, found {len(csv_files)}: "
        + ", ".join(path.name for path in csv_files)
    )

csv_path = csv_files[0]

print(f"Loading csv: {csv_path.name}")
df = pd.read_csv(csv_path)

missing_columns = [col for col in var_names if col not in df.columns]
if missing_columns:
    raise ValueError(f"CSV is missing required columns: {missing_columns}")

df = df[var_names]


if not (0.0 <= warmup_percent < 100.0):
    raise ValueError("warmup_percent must be in [0, 100).")
if not (0.0 < used_percent <= 100.0):
    raise ValueError("used_percent must be in (0, 100].")

total_samples = len(df)
warmup = int(np.floor((warmup_percent / 100.0) * total_samples))
T_used = max(1, int(np.floor((used_percent / 100.0) * total_samples)))
end_index = min(total_samples, warmup + T_used)

print(
    f"Data window: warmup={warmup} ({warmup_percent:.1f}%), "
    f"requested={T_used} ({used_percent:.1f}%), used={max(0, end_index - warmup)} / {total_samples}"
)

df = df.iloc[warmup:end_index].reset_index(drop=True)
if df.empty:
    raise ValueError("No samples left after applying warmup_percent/used_percent slice.")

raw_data = df.to_numpy(dtype=float)
if standardize_data:
    means = np.mean(raw_data, axis=0)
    stds = np.std(raw_data, axis=0)
    stds[stds < 1e-12] = 1e-12
    data = (raw_data - means) / stds
    print("Data preprocessing: standardized (zero-mean, unit-variance per variable)")
else:
    data = raw_data.copy()
    print("Data preprocessing: using raw values (no standardization)")

T, N = data.shape

tau_max = resolve_tau_max(tau_max)
if T <= tau_max:
    raise ValueError(f"Not enough time points (T={T}) for tau_max={tau_max}.")

if add_tiny_jitter:
    print(f"Adding tiny jitter (std={jitter_std}) for numerical stability")
    rng = np.random.default_rng(random_seed)
    data = data + rng.normal(0.0, jitter_std, size=data.shape)

print(f"Data shape: T={T}, N={N}")
print(f"Lag range: tau_min={tau_min}, tau_max={tau_max}")
print("Variables:", var_names)

dataframe = DataFrame(data=data, var_names=var_names)

# ===========================================================
# Choose CI test
# ===========================================================

if use_test.lower() == "cmiknn":
    print("Using CMIknn independence test")
    print(
        "CMIknn settings: "
        f"significance={cmiknn_significance}, knn={cmiknn_knn}, "
        f"shuffle_neighbors={cmiknn_shuffle_neighbors}"
    )
    ind_test = CMIknn(
        significance=cmiknn_significance,
        knn=cmiknn_knn,
        shuffle_neighbors=cmiknn_shuffle_neighbors,
    )
elif use_test.lower() == "gpdc":
    print("Using GPDC independence test")
    ind_test = GPDC(significance='analytic')
else:
    raise ValueError("use_test must be 'CMIknn' or 'GPDC'")

# ===========================================================
# Run PCMCI+
# ===========================================================

RUNTIME_HISTORY_PATH = DATA_DIR / "pcmci_runtime_history.json"


def load_runtime_history(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_runtime_history(path: Path, history: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def estimate_runtime_seconds(history: dict, T: int, N: int, tau_min: int, tau_max: int, test_name: str) -> float:
    lag_count = max(1, tau_max - tau_min + 1)
    key = f"{test_name.lower()}|T={T}|N={N}|lag={lag_count}"
    samples = history.get(key, [])
    if samples:
        return float(np.mean(samples))

    # Fallback heuristic estimate when no prior runs exist for this configuration.
    complexity = N * N * lag_count * np.log(max(T, 2))
    factor = 0.16 if test_name.lower() == "cmiknn" else 0.05
    return max(5.0, factor * complexity)


def update_runtime_history(history: dict, runtime_s: float, T: int, N: int, tau_min: int, tau_max: int, test_name: str) -> dict:
    lag_count = max(1, tau_max - tau_min + 1)
    key = f"{test_name.lower()}|T={T}|N={N}|lag={lag_count}"
    samples = history.get(key, [])
    samples.append(float(runtime_s))
    history[key] = samples[-10:]
    return history


def run_with_progress(task, label: str = "PCMCI+", total_seconds_estimate: float | None = None):
    """Run a long task while showing progress and optional rough ETA."""
    stop_event = threading.Event()
    start = time.time()

    def _ticker():
        tick = 0
        while not stop_event.wait(1.0):
            tick = (tick + 1) % 20
            bar = "#" * tick + "-" * (20 - tick)
            elapsed = time.time() - start
            if total_seconds_estimate and total_seconds_estimate > 0:
                remaining = max(0.0, total_seconds_estimate - elapsed)
                print(
                    f"\r{label} running [{bar}] {elapsed:6.1f}s elapsed | ~{remaining:6.1f}s ETA",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r{label} running [{bar}] {elapsed:6.1f}s elapsed", end="", flush=True)
        print("\r" + " " * 110 + "\r", end="", flush=True)

    ticker = threading.Thread(target=_ticker, daemon=True)
    ticker.start()
    try:
        result = task()
    finally:
        stop_event.set()
        ticker.join()

    return result, time.time() - start


pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test, verbosity=pcmci_verbosity)
runtime_history = load_runtime_history(RUNTIME_HISTORY_PATH)
rough_total_seconds = estimate_runtime_seconds(runtime_history, T, N, tau_min, tau_max, use_test)

print("\nRunning PCMCI+ ...")
print(f"Rough runtime estimate: ~{rough_total_seconds:.1f}s (improves after repeated runs)")
results, runtime_s = run_with_progress(
    lambda: pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha),
    label="PCMCI+",
    total_seconds_estimate=rough_total_seconds,
)

runtime_history = update_runtime_history(runtime_history, runtime_s, T, N, tau_min, tau_max, use_test)
save_runtime_history(RUNTIME_HISTORY_PATH, runtime_history)
print(f"PCMCI+ finished in {runtime_s:.1f}s.")

p_mat = results['p_matrix'].copy()
val_mat = results['val_matrix'].copy()

if fdr_method:
    print(f"Applying multiple-testing correction: {fdr_method}")
    sig_mat = pcmci.get_corrected_pvalues(p_matrix=p_mat, fdr_method=fdr_method)
else:
    sig_mat = p_mat

# ===========================================================
# Enforce "inputs exogenous" by post-filtering
# ===========================================================
input_indices = [var_names.index(x) for x in inputs]
for j in input_indices:
    sig_mat[:, j, :] = 1.0
    val_mat[:, j, :] = 0.0

print("\nPost-filter: removed all parents into inputs:", ", ".join(inputs))

# Remove self-links (same variable -> same variable at any lag) from output graph.
for idx, var_name in enumerate(var_names):
    sig_mat[idx, idx, :] = 1.0
    val_mat[idx, idx, :] = 0.0

print("Post-filter: removed all self-links:", ", ".join(var_names))

# ===========================================================
# Print significant links (after correction + cleaning)
# ===========================================================

print("\n===== Significant links after cleaning (alpha = {:.3f}) =====".format(pc_alpha))
pcmci.print_significant_links(p_matrix=sig_mat, val_matrix=val_mat, alpha_level=pc_alpha)

# ===========================================================
# Build graph and draw circular layout
# ===========================================================

print("\nBuilding graph for visualization ...")
G = nx.DiGraph()
G.add_nodes_from(var_names)

significant_links = []
for j, target in enumerate(var_names):
    for i, source in enumerate(var_names):
        for tau in range(tau_min, tau_max + 1):
            p_value = sig_mat[i, j, tau]
            if p_value < pc_alpha:
                effect_size = val_mat[i, j, tau]
                weight = abs(effect_size)
                significant_links.append(
                    {
                        "source": source,
                        "target": target,
                        "lag": tau,
                        "p_value": float(p_value),
                        "effect_size": float(effect_size),
                        "abs_effect_size": float(weight),
                    }
                )
                G.add_edge(source, target, weight=weight, lag=tau)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

links_output_path = DATA_DIR / "cstr_graph_links.csv"
links_df = pd.DataFrame(
    significant_links,
    columns=["source", "target", "lag", "p_value", "effect_size", "abs_effect_size"],
)
links_df.to_csv(links_output_path, index=False)
print(f"Significant link table saved to {links_output_path}")

graphml_output_path = DATA_DIR / "cstr_graph.graphml"
nx.write_graphml(G, graphml_output_path)
print(f"Graph structure saved to {graphml_output_path}")

pos = nx.circular_layout(G)
node_colors = ["#1f77b4" if v in states else "#ff7f0e" for v in var_names]
edge_widths = [2 * G[u][v]['weight'] for u, v in G.edges()]

plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")
nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=18,
    width=edge_widths,
    edge_color="black",
    connectionstyle="arc3,rad=0.05",
)
edge_labels = {(u, v): f"τ={G[u][v]['lag']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title(f"CSTR causal graph (PCMCI+ with {use_test})", fontsize=14)
plt.axis("off")
plt.tight_layout()
output_path = DATA_DIR / "cstr_graph.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nGraph saved to {output_path}")
print("Done.")
