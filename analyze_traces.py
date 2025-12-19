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
plt.rcParams["axes.titlepad"] = 20   # default ~6, now moved higher

BACKEND_COLORS = {"GLOO": "#d62728", "NCCL": "#1f77b4"}
BACKEND_LEGEND = [Patch(facecolor=BACKEND_COLORS[b], label=b) for b in BACKEND_COLORS]

# ============================================================
# TITLE MAP (preserves original plot titles)
# ============================================================
TITLE_MAP = {
    "tokens_per_sec": "End-to-End Training Throughput",
    "iter_time_ms": "Time per Training Iteration",
    "overlap_pct": "Communication–Computation Overlap",
    "bandwidth_gbps": "Average Collective Bandwidth",
    "total_traffic_gb": "Total Network Traffic per Iteration",
    "allgather_bw": "Average AllGather Bandwidth",
    "reducescatter_bw": "Average ReduceScatter Bandwidth",
    "allreduce_bw": "Average AllReduce Bandwidth",
}

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

    if re.search(r"llama.*8b", fname_lower):
        model = "LLaMA 3 8B"
    elif re.search(r"qwen.*32b", fname_lower):
        model = "Qwen 32B"
    else:
        model = "Unknown Model"

    label = f"{model} | {gpus} GPUs | DP={dp} TP={tp}"
    return backend, label, dp, tp, gpus

# ============================================================
# REPORT PARSING
# ============================================================
def parse_report(text: str):
    def grab(pattern, cast=float):
        m = re.search(pattern, text, re.DOTALL)
        return cast(m.group(1)) if m else np.nan

    metrics = {
        "tokens_per_sec": grab(r"Tokens per Second:\s*([\d.]+)"),
        "iter_time_ms": grab(r"Time per Iteration:\s*([\d.]+)"),
        "overlap_pct": grab(r"Average Overlap:\s*([\d.]+)"),
        "bandwidth_gbps": grab(r"Average Collective Bandwidth:\s*([\d.]+)"),
        "total_traffic_gb": grab(r"Total Traffic:\s*([\d.]+)"),
        "allgather_bw": grab(r"AllGather:[\s\S]*?Avg BW:\s*([\d.]+)"),
        "reducescatter_bw": grab(r"ReduceScatter:[\s\S]*?Avg BW:\s*([\d.]+)"),
        "allreduce_bw": grab(r"AllReduce:[\s\S]*?Avg BW:\s*([\d.]+)"),
    }
    return metrics

# ============================================================
# LOAD TRACE FILES
# ============================================================
records = []

for path in TRACE_DIR.glob("*.txt"):
    backend, label, dp, tp, gpus = parse_filename(path.name)

    with open(path, "r") as f:
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
# BASE PLOTTING FUNCTION
# ============================================================
labels = [r["label"] for r in records]
colors = [BACKEND_COLORS[r["backend"]] for r in records]

def bar_plot(key, ylabel, title, fname):
    values = [r[key] for r in records]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, values, color=colors)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles=BACKEND_LEGEND, frameon=False)

    y_offset = 0.03 * max(values)

    for i, v in enumerate(values):
        if not np.isnan(v):
            if key == "tokens_per_sec":  # fix throughput label overlap
                ax.text(i, v + y_offset, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7, rotation=45)
            else:
                ax.text(i, v + y_offset, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)

# ============================================================
# CORE PLOTS
# ============================================================
bar_plot("tokens_per_sec", "Tokens / second",
         "End-to-End Training Throughput", "throughput.png")
bar_plot("iter_time_ms", "Milliseconds",
         "Time per Training Iteration", "iteration_time.png")
bar_plot("overlap_pct", "Overlap (%)",
         "Communication–Computation Overlap", "overlap.png")
bar_plot("bandwidth_gbps", "GB / second",
         "Average Collective Bandwidth", "bandwidth.png")
bar_plot("total_traffic_gb", "GB",
         "Total Network Traffic per Iteration", "total_traffic.png")
bar_plot("allgather_bw", "GB / second",
         "Average AllGather Bandwidth", "allgather_bandwidth.png")
bar_plot("reducescatter_bw", "GB / second",
         "Average ReduceScatter Bandwidth", "reducescatter_bandwidth.png")
bar_plot("allreduce_bw", "GB / second",
         "Average AllReduce Bandwidth", "allreduce_bandwidth.png")

# ============================================================
# NCCL-only + GLOO-only VERSIONS
# ============================================================
def backend_plot(backend_name, key, ylabel, fname_suffix):
    subset = [r for r in records if r["backend"] == backend_name]
    if not subset:
        return

    labs = [r["label"] for r in subset]
    vals = [r[key] for r in subset]
    cols = [BACKEND_COLORS[r["backend"]] for r in subset]

    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, vals, color=cols)

    base_title = TITLE_MAP.get(key, key)
    ax.set_title(f"{base_title} ({backend_name} only)", pad=20)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


    y_offset = 0.03 * max(vals) if vals else 0
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + y_offset, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{backend_name.lower()}_{fname_suffix}.png")
    plt.close(fig)


for key in TITLE_MAP.keys():

    if key == "tokens_per_sec":
        ylabel = "Tokens / second"
    elif key == "iter_time_ms":
        ylabel = "Milliseconds"
    elif key.endswith("_bw") or key == "bandwidth_gbps":
        ylabel = "GB / second"
    elif key == "total_traffic_gb":
        ylabel = "GB"
    elif key == "overlap_pct":
        ylabel = "Overlap (%)"
    else:
        ylabel = "Value"

    backend_plot("NCCL", key, ylabel, f"{key}")
    backend_plot("GLOO", key, ylabel, f"{key}")

print(f"[OK] All plots saved to → {OUT_DIR.resolve()}")
