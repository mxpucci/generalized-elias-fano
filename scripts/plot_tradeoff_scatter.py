#!/usr/bin/env python3
"""
Generate publication-quality scatter plots that compare compression ratio (%)
against decompression speed and random-access speed for both competitor codecs
and GEF variants. All statistics are averaged across datasets at partition size
2^20 (single-thread JSON results).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Preserving your import structure with fallback
try:
    from competitor_data import competitor_benchmarks, competitor_tradeoff_placeholders
except ImportError:
    # print("Warning: 'competitor_data' module not found. Using manual averages if available.")
    competitor_benchmarks = {}
    competitor_tradeoff_placeholders = {}

TARGET_PARTITION_SIZE = 32000

GEF_VARIANTS = {
    "RLE_GEF": "RLE-GEF",
    "U_GEF_Approximate": "U-GEF (Approximate)",
    "U_GEF_Optimal": "U-GEF (Optimal)",
    "B_GEF_Approximate": "B-GEF (Approximate)",
    "B_GEF_Optimal": "B-GEF (Optimal)",
    "B*_GEF_Approximate": "B*-GEF (Approximate)",
}

# ==========================================
# 1. PARSING LOGIC
# ==========================================

def parse_partition_size(benchmark_name: str) -> int | None:
    for part in benchmark_name.split("/"):
        if part.startswith("partition_size:"):
            return int(part.split(":")[1])
    return None


def parse_compressor_and_strategy(benchmark_name: str) -> str | None:
    parts = benchmark_name.split("/")
    compressor_type = None
    for part in parts:
        if "B_GEF_NO_RLE" in part or "B_GEF_STAR" in part:
            compressor_type = "B*_GEF"
            break
        if "B_GEF" in part:
            compressor_type = "B_GEF"
            break
        if "U_GEF" in part:
            compressor_type = "U_GEF"
            break
        if "RLE_GEF" in part:
            compressor_type = "RLE_GEF"
            break
    if compressor_type is None:
        return None
    if compressor_type == "RLE_GEF":
        return "RLE_GEF"
    for part in parts:
        if part.startswith("strategy:"):
            strategy_num = int(part.split(":")[1])
            strategy_name = "Approximate" if strategy_num == 0 else "Optimal"
            return f"{compressor_type}_{strategy_name}"
    return None


def calculate_compression_percentage(size_in_bytes: float, num_integers: float) -> float:
    if num_integers == 0:
        return 0.0
    return (size_in_bytes / (num_integers * 8)) * 100.0


def compute_competitor_averages() -> Dict[str, Dict[str, float]]:
    """Average the competitor metrics across datasets."""
    comp_ratio = competitor_benchmarks.get("compression_ratio", {})
    comp_decomp = competitor_benchmarks.get("decompression_speed", {})
    comp_access = competitor_benchmarks.get("random_access_speed", {})

    all_algorithms = set(comp_ratio) | set(comp_decomp) | set(comp_access)
    averages: Dict[str, Dict[str, float]] = {}

    for algo in all_algorithms:
        metrics = {}
        if algo in comp_ratio and comp_ratio[algo]:
            metrics["compression_ratio"] = float(np.mean(list(comp_ratio[algo].values())))
        if algo in comp_decomp and comp_decomp[algo]:
            metrics["decompression_speed"] = float(np.mean(list(comp_decomp[algo].values())))
        if algo in comp_access and comp_access[algo]:
            metrics["random_access_speed"] = float(np.mean(list(comp_access[algo].values())))
        if metrics:
            averages[algo] = metrics

    for algo, metrics in competitor_tradeoff_placeholders.items():
        averages.setdefault(algo, {})
        averages[algo].update(metrics)

    return averages


def load_gef_tradeoff_points(
    bench_output_dir: Path, partition_size: int
) -> Dict[str, Dict[str, float]]:
    """Extract averaged compression ratio, decompression, and access speeds for GEF variants."""
    stats = defaultdict(lambda: {"ratio": [], "comp_throughput": [], "decomp": [], "access": []})

    if not bench_output_dir.exists():
        # Silent fail or warning handled in main
        return {}

    json_files = list(bench_output_dir.glob("*_no_omp.json"))
    if not json_files:
        return {}

    for json_file in json_files:
        with open(json_file, "r") as fh:
            lines = fh.readlines()
        # Robust JSON start finding
        json_start = 0
        for idx, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = idx
                break
        try:
            data = json.loads("".join(lines[json_start:]))
        except json.JSONDecodeError:
            continue

        for bench in data.get("benchmarks", []):
            if bench.get("run_type") != "iteration":
                continue
            ps = parse_partition_size(bench.get("name", ""))
            if ps != partition_size:
                continue
            variant_key = parse_compressor_and_strategy(bench.get("name", ""))
            if variant_key not in GEF_VARIANTS:
                continue

            family = bench.get("name", "").split("/")[1]
            if family.endswith("_Compression"):
                size_in_bytes = bench.get("size_in_bytes", 0)
                num_integers = bench.get("num_integers", 0)
                compression_pct = calculate_compression_percentage(size_in_bytes, num_integers)
                if compression_pct > 0:
                    stats[variant_key]["ratio"].append(compression_pct)
                
                throughput = bench.get("compression_throughput_MBs", 0)
                if throughput > 0:
                    stats[variant_key]["comp_throughput"].append(throughput / 1e6)
            elif family.endswith("_Decompression"):
                throughput = bench.get("decompression_throughput_MBs", 0)
                if throughput > 0:
                    stats[variant_key]["decomp"].append(throughput / 1e6)
            elif family.endswith("_Lookup"):
                throughput = bench.get("lookup_throughput_MBs", 0)
                if throughput > 0:
                    stats[variant_key]["access"].append(throughput / 1e6)

    aggregates: Dict[str, Dict[str, float]] = {}
    for key, series in stats.items():
        if not series["ratio"]:
            continue
        label = GEF_VARIANTS[key]
        metrics: Dict[str, float] = {"compression_ratio": float(np.mean(series["ratio"]))}
        if series["comp_throughput"]:
            metrics["compression_throughput"] = float(np.mean(series["comp_throughput"]))
        if series["decomp"]:
            metrics["decompression_speed"] = float(np.mean(series["decomp"]))
        if series["access"]:
            metrics["random_access_speed"] = float(np.mean(series["access"]))
        if len(metrics) > 1:
            aggregates[label] = metrics
    return aggregates

# ==========================================
# 2. STYLING
# ==========================================

# Common shape for all GEF variants
GEF_SHAPE = 'P' # 'P' is a filled plus sign, distinct from competitors

STYLES = {
    # COMPETITORS
    "Xz": {"m": "*", "c": "#EA4335", "label": "Xz"},
    "Brotli": {"m": "h", "c": "#3B7C26", "label": "Brotli"},
    "Zstd": {"m": "X", "c": "#5BC0BE", "label": "Zstd"},
    "Lz4": {"m": "o", "c": "#E040FB", "label": "Lz4"},
    "Snappy": {"m": "o", "c": "#8D8E2C", "label": "Snappy"},
    "DAC": {"m": "d", "c": "#C62828", "label": "DAC"},
    "ALP": {"m": "D", "c": "#F8BBD0", "label": "ALP"},
    "Chimp": {"m": "<", "c": "#8D4E4B", "label": "Chimp"},
    "Chimp128": {"m": "^", "c": "#6A1B9A", "label": "Chimp128"},
    "Gorilla": {"m": ">", "c": "black", "label": "Gorilla"},
    "TSXor": {"m": "v", "c": "#C0CA33", "label": "TSXor"},
    "LeCo": {"m": "p", "c": "#FBC02D", "label": "LeCo"},
    "NeaTS": {"m": "X", "c": "blue", "label": "NeaTS"},
    "SNeaTS": {"m": "X", "c": "grey", "label": "SNeaTS"},
    "LeaTS": {"m": "p", "c": "#ff9800", "label": "LeaTS"},
    
    # GEF VARIANTS - All share GEF_SHAPE, but different colors
    "RLE-GEF":            {"m": GEF_SHAPE, "c": "#e377c2", "label": "RLE-GEF"},
    "U-GEF (Approximate)":{"m": GEF_SHAPE, "c": "#2ca02c", "label": "U-GEF (Approx.)"},
    "U-GEF (Optimal)":    {"m": GEF_SHAPE, "c": "#d62728", "label": "U-GEF (Optimal)"},
    "B-GEF (Approximate)":{"m": GEF_SHAPE, "c": "#1f77b4", "label": "B-GEF (Approx.)"},
    "B-GEF (Optimal)":    {"m": GEF_SHAPE, "c": "#ff7f0e", "label": "B-GEF (Optimal)"},
    "B*-GEF (Approximate)":{"m": GEF_SHAPE, "c": "#9467bd", "label": "B*-GEF (Approx.)"},
}

def prepare_data_points(
    gef_points: Dict[str, Dict[str, float]],
    competitor_avgs: Dict[str, Dict[str, float]],
    metric_key: str,
) -> Dict[str, Tuple[float, float]]:
    """Combine competitor and GEF averages for the requested metric."""
    data_points: Dict[str, Tuple[float, float]] = {}
    for name, metrics in competitor_avgs.items():
        ratio = metrics.get("compression_ratio")
        value = metrics.get(metric_key)
        if ratio is None or value is None:
            continue
        data_points[name] = (ratio, value)

    for name, metrics in gef_points.items():
        ratio = metrics.get("compression_ratio")
        value = metrics.get(metric_key)
        if ratio is None or value is None:
            continue
        data_points[name] = (ratio, value)

    return data_points

# ==========================================
# 3. PLOTTING
# ==========================================

def draw_better_arrow(ax):
    """
    Draws a blocky 'Better' arrow pointing Top-Left.
    Logic: 'larrow' boxstyle points to the left of the text.
    If we rotate text -45 (downhill reading), the 'Left' side of the text
    points to the Top-Left of the screen.
    """
    # Coordinates (0-1 relative to axes)
    x_pos = 0.45
    y_pos = 0.45 
    
    # Rotation -45 ensures the arrow points Top-Left
    rotation = -45 

    ax.text(x_pos, y_pos, "Better", 
            transform=ax.transAxes, 
            rotation=rotation,
            ha="center", va="center",
            size=13, color="dimgrey", alpha=0.9,
            family='serif',
            bbox=dict(boxstyle="larrow,pad=0.3", 
                      fc="silver", 
                      ec="none",   
                      alpha=0.4))

def create_benchmark_plot(
    data: Dict[str, Tuple[float, float]],
    title: str,
    y_label: str,
    log_scale_y: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort to make legend order predictable (optional)
    for algo_name in sorted(data.keys()):
        style = STYLES.get(algo_name)
        if style is None:
            continue
        ratio, speed = data[algo_name]
        if log_scale_y and speed <= 0:
            continue
        
        # Increase marker size slightly for visibility
        size = 90 if style['m'] == GEF_SHAPE else 80
        
        ax.scatter(
            ratio,
            speed,
            marker=style["m"],
            color=style["c"],
            s=size,
            label=style["label"],
            edgecolors="none",
            zorder=10,
        )

    if log_scale_y:
        ax.set_yscale("log")

    # Typography and Grid
    ax.set_xlabel("Compression ratio (%)", fontsize=14, family='serif')
    ax.set_ylabel(y_label, fontsize=14, family='serif')
    
    ax.grid(True, which="major", ls="-", color='#dddddd', linewidth=1.0, zorder=0)
    ax.grid(True, which="minor", ls=":", color='#eeeeee', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Thicken borders for publication quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    draw_better_arrow(ax)

    # Legend Configuration
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18), # Below the chart
            fancybox=False,
            frameon=True,
            edgecolor='lightgrey',
            shadow=False,
            ncol=5, # Wide layout
            fontsize=10,
            columnspacing=1.0
        )

    plt.title(title, fontsize=16, pad=20, family='serif')
    plt.tight_layout()
    return fig

def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{stem}.pdf"
    fig.savefig(dest, bbox_inches="tight")
    print(f"Saved: {dest}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot compression trade-off scatter charts.")
    parser.add_argument("bench_output_dir", nargs="?", default="benchmark_results")
    parser.add_argument("output_dir", nargs="?", default="plots")
    parser.add_argument("--partition_size", type=int, default=TARGET_PARTITION_SIZE)
    args = parser.parse_args()

    bench_dir = Path(args.bench_output_dir)
    output_dir = Path(args.output_dir)

    competitor_avgs = compute_competitor_averages()
    gef_points = load_gef_tradeoff_points(bench_dir, args.partition_size)

    if not competitor_avgs and not gef_points:
        print("Warning: No data found. Ensure 'benchmark_results' has JSON files or 'competitor_data' is available.")

    # 1. Compression Throughput Plot (Logarithmic)
    compression_points = prepare_data_points(
        gef_points, competitor_avgs, "compression_throughput"
    )
    if compression_points:
        fig_comp = create_benchmark_plot(
            compression_points,
            "Compression Ratio vs Compression Throughput",
            "Compression throughput (MB/s)",
            log_scale_y=True, # Changed to True
        )
        save_figure(fig_comp, output_dir, "compression_throughput_vs_ratio")

    # 2. Decompression Plot (Logarithmic)
    decompression_points = prepare_data_points(
        gef_points, competitor_avgs, "decompression_speed"
    )
    if decompression_points:
        fig_decomp = create_benchmark_plot(
            decompression_points,
            "Compression Ratio vs Decompression Speed",
            "Decompression speed (MB/s)",
            log_scale_y=True, # Changed to True
        )
        save_figure(fig_decomp, output_dir, "decompression_speed_vs_ratio")

    # 3. Random Access Plot (Logarithmic)
    access_points = prepare_data_points(
        gef_points, competitor_avgs, "random_access_speed"
    )
    if access_points:
        fig_log = create_benchmark_plot(
            access_points,
            "Compression Ratio vs Random Access Speed",
            "Random access speed (MB/s)",
            log_scale_y=True,
        )
        save_figure(fig_log, output_dir, "random_access_speed_vs_ratio")

if __name__ == "__main__":
    main()