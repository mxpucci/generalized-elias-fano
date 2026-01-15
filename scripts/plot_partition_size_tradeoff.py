import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuration
# ==========================================
DEFAULT_RESULTS_DIR = "benchmark_results"
OUTPUT_DIR = "plots"

# Mapping internal compressor names to Display Names
NAME_MAPPING = {
    "B_STAR_GEF": "B*-GEF (Optimal)",
    "B_STAR_GEF_APPROXIMATE": "B*-GEF (Approximate)",
    "B_GEF": "B-GEF (Optimal)",
    "B_GEF_APPROXIMATE": "B-GEF (Approximate)",
    "RLE_GEF": "RLE-GEF",
    "U_GEF": "U-GEF (Optimal)",
    "U_GEF_APPROXIMATE": "U-GEF (Approximate)"
}

# Styles matching your reference PDF
STYLE_MAPPING = {
    "B*-GEF (Approximate)":   {"color": "#8172b3", "marker": "D", "linestyle": "-."}, # Purple Diamond
    "B*-GEF (Optimal)":       {"color": "#8c564b", "marker": "p", "linestyle": "-."}, # Brown Pentagon
    "B-GEF (Approximate)":    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},  # Blue Circle
    "B-GEF (Optimal)":        {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},  # Orange Square
    "RLE-GEF":                {"color": "#e377c2", "marker": "h", "linestyle": ":"},  # Pink Hexagon
    "U-GEF (Approximate)":    {"color": "#2ca02c", "marker": "^", "linestyle": "--"}, # Green Triangle Up
    "U-GEF (Optimal)":        {"color": "#d62728", "marker": "v", "linestyle": "--"}, # Red Triangle Down
}

PARTITION_SIZES = [8000, 10000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 100000, 250000, 500000]

# Requested Legend Order
COMPRESSOR_ORDER = [
    "RLE-GEF",
    "U-GEF (Approximate)",
    "U-GEF (Optimal)",
    "B-GEF (Approximate)",
    "B-GEF (Optimal)",
    "B*-GEF (Approximate)",
    "B*-GEF (Optimal)"
]

plt.rcParams.update({
    # 1. Force Type 42 (TrueType) to avoid "Type 3" errors
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    
    # 2. Disable external LaTeX (Safe Mode)
    "text.usetex": False,
    
    # 3. Font Priority List
    # "Linux Libertine O": The actual VLDB font (if you have the .ttf installed)
    # "Palatino Linotype": The Windows/Office standard for Palatino
    # "Book Antiqua": A very common Palatino clone
    # "Palatino": The Apple/Unix standard
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Palatino Linotype", "Book Antiqua", "Palatino", "Times New Roman"],
    
    # 4. Math Font
    # 'stix' is the best match for Palatino-style math. 
    # 'cm' (Computer Modern) is too thin.
    "mathtext.fontset": "stix",
    
    # 5. Sizes (VLDB uses 9pt body, so 8-10pt figures fit best)
    "font.size": 14,
    "legend.fontsize": 10,
    "axes.labelsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

def load_benchmark_data(results_dir):
    data_rows = []
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    if not json_files:
        print(f"No .json files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(json_files)} result files. Parsing...")

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
                
            for bench in content.get('benchmarks', []):
                # Name format: "dataset/compressor/partition_size/mode"
                name_parts = bench['name'].split('/')
                
                if len(name_parts) < 4: continue
                
                dataset = name_parts[0]
                compressor_raw = name_parts[1]
                try:
                    partition_size = int(name_parts[2])
                except ValueError:
                    continue # Skip if partition size isn't an int
                mode = name_parts[3]
                
                # Filter out Disabled RA if desired (optional)
                if "RA_Disabled" in mode: continue

                # Metrics
                comp_ratio = bench.get('CompressionRatio')
                throughput_bps = bench.get('bytes_per_second', 0)
                
                if comp_ratio is not None:
                    data_rows.append({
                        "Compressor": NAME_MAPPING.get(compressor_raw, compressor_raw),
                        "Dataset": dataset,
                        "Partition Size": partition_size,
                        "Compression Ratio (%)": comp_ratio * 100,
                        "Throughput (MB/s)": throughput_bps / (1024 * 1024)
                    })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return pd.DataFrame(data_rows)

def format_xtick(val, pos):
    if val >= 1000: return f'{int(val/1000)}K'
    return str(val)

def setup_plot(ax, title, ylabel):
    # ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Partition Size", fontsize=16, weight='bold')
    ax.set_ylabel(ylabel, fontsize=16, weight='bold')
    # Use indices for x-axis ticks to ensure equal spacing
    x_indices = range(len(PARTITION_SIZES))
    ax.set_xticks(x_indices)
    ax.set_xticklabels([format_xtick(s, None) for s in PARTITION_SIZES])
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.minorticks_off()
    
    # loc='best' automatically finds the location with minimum overlap
    ax.legend(loc='best', frameon=False, fontsize=10)

def main():
    parser = argparse.ArgumentParser(description="Plot partition size tradeoff results.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR, help="Directory containing benchmark JSON results")
    args = parser.parse_args()

    results_dir = args.results_dir

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    df = load_benchmark_data(results_dir)
    if df.empty: return

    # Aggregate
    df_avg = df.groupby(['Compressor', 'Partition Size'])[['Compression Ratio (%)', 'Throughput (MB/s)']].mean().reset_index()

    # Map Partition Size to Index for plotting
    size_to_idx = {size: i for i, size in enumerate(PARTITION_SIZES)}
    df_avg['Partition Index'] = df_avg['Partition Size'].map(size_to_idx)
    
    # Filter out any sizes that are not in PARTITION_SIZES (optional safety check)
    df_avg = df_avg.dropna(subset=['Partition Index'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # Sort compressors for consistent plotting order
    available_compressors = set(df_avg['Compressor'].unique())
    sorted_compressors = [c for c in COMPRESSOR_ORDER if c in available_compressors] + \
                         [c for c in available_compressors if c not in COMPRESSOR_ORDER]

    # Plot 1: Ratio
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for compressor in sorted_compressors:
        subset = df_avg[df_avg['Compressor'] == compressor].sort_values("Partition Size")
        style = STYLE_MAPPING.get(compressor, {"color": "gray", "marker": "o", "linestyle": "-"})
        ax1.plot(subset['Partition Index'], subset['Compression Ratio (%)'], label=compressor, **style, linewidth=2, markersize=8)
    setup_plot(ax1, "Compression Ratio vs Partition Size", "Compression Ratio (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "compression_ratio.pdf"))
    print(f"Saved {OUTPUT_DIR}/compression_ratio.pdf")

    # Plot 2: Throughput
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for compressor in sorted_compressors:
        subset = df_avg[df_avg['Compressor'] == compressor].sort_values("Partition Size")
        style = STYLE_MAPPING.get(compressor, {"color": "gray", "marker": "o", "linestyle": "-"})
        ax2.plot(subset['Partition Index'], subset['Throughput (MB/s)'], label=compressor, **style, linewidth=2, markersize=8)
    setup_plot(ax2, "Compression Throughput vs Partition Size", "Compression Throughput (MB/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "compression_throughput.pdf"))
    print(f"Saved {OUTPUT_DIR}/compression_throughput.pdf")

if __name__ == "__main__":
    main()