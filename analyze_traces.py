import re
import pathlib
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# CONFIG
# ============================================================
TRACE_DIR = pathlib.Path("trace_analysis")
OUT_DIR = pathlib.Path("plots")
OUT_DIR.mkdir(exist_ok=True)

CSV_OUT = OUT_DIR / "summary_metrics.csv"

plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BACKEND_COLORS = {
    "GLOO": "#d62728",
    "NCCL": "#1f77b4",
}

BACKEND_LEGEND = [
    Patch(facecolor=BACKEND_COLORS[b], label=b)
    for b in BACKEND_COLORS
]

# ============================================================
# FILENAME PARSING
# ============================================================
def parse_filename(fname: str):
    backend = "NCCL" if fname.startswith("nccl") else "GLOO"

    def grab(pattern):
        m = re.search(pattern, fname)
        return int(m.group(1)) if m else None

    dp = grab(r"dp1x(\d+)")
    tp = grab(r"tp(\d+)")
    gpus = grab(r"gpu(\d+)")

    fname_lower = fname.lower()

    # --- robust model detection ---
    if re.search(r"llama.*8b", fname_lower):
        model = "LLaMA 3 8B"
    elif re.search(r"qwen.*32b", fname_lower):
        model = "Qwen 32B"
    else:
        model = "Unknown Model"

    label = f"{model} | {gpus} GPUs | DP={dp} TP={tp}"
    return backend, label, dp, tp, gpus


# ============================================================
# TRACE PARSING
# ============================================================
def parse_report(text: str):
    def grab(pattern, cast=float):
        m = re.search(pattern, text, re.DOTALL)
        return cast(m.group(1)) if m else np.nan

    return {
        "tokens_per_sec": grab(r"Tokens per Second:\s*([\d.]+)"),
        "iter_time_ms": grab(r"Time per Iteration:\s*([\d.]+)"),
        "overlap_pct": grab(r"Average Overlap:\s*([\d.]+)"),
        "bandwidth_gbps": grab(r"Average Collective Bandwidth:\s*([\d.]+)"),
        "total_traffic_gb": grab(r"Total Traffic:\s*([\d.]+)"),
    }

# ============================================================
# LOAD + SORT TRACE FILES
# ============================================================
records = []

for path in TRACE_DIR.glob("*.txt"):
    backend, label, dp, tp, gpus = parse_filename(path.name)

    with open(path) as f:
        metrics = parse_report(f.read())

    records.append({
        "file": path.name,
        "backend": backend,
        "label": label,
        "dp": dp,
        "tp": tp,
        "gpus": gpus,
        **metrics,
    })

assert records, "No trace files found."

records.sort(key=lambda r: (r["backend"], r["gpus"], r["tp"]))

# ============================================================
# WRITE CSV SUMMARY
# ============================================================
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"[OK] Wrote CSV summary → {CSV_OUT}")

# ============================================================
# PLOTTING HELPERS
# ============================================================
labels = [r["label"] for r in records]
colors = [BACKEND_COLORS[r["backend"]] for r in records]

def bar_plot(key, ylabel, title, fname):
    values = [r[key] for r in records]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, values, color=colors)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=BACKEND_LEGEND, frameon=False)

    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)

# ============================================================
# CORE METRIC PLOTS
# ============================================================
bar_plot(
    "tokens_per_sec",
    "Tokens / second",
    "End-to-End Training Throughput",
    "throughput.png",
)

bar_plot(
    "iter_time_ms",
    "Milliseconds",
    "Time per Training Iteration",
    "iteration_time.png",
)

bar_plot(
    "overlap_pct",
    "Overlap (%)",
    "Communication–Computation Overlap",
    "overlap.png",
)

bar_plot(
    "bandwidth_gbps",
    "GB / second",
    "Average Collective Bandwidth",
    "bandwidth.png",
)

bar_plot(
    "total_traffic_gb",
    "GB",
    "Total Network Traffic per Iteration",
    "total_traffic.png",
)

print(f"[OK] All plots saved to → {OUT_DIR.resolve()}")
