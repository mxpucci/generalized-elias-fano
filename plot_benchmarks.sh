#!/bin/bash

# =============================================================================
# Benchmark Plotting Script
# =============================================================================
# This script generates plots from compression benchmark results.
#
# Usage:
#   ./plot_benchmarks.sh [bench_output_dir] [output_dir] [partition_size]
#
# Arguments:
#   bench_output_dir - Directory containing benchmark JSON files (default: ./bench-output)
#   output_dir       - Directory to save plots (default: ./plots)
#   partition_size   - Partition size to filter for tables (default: 1048576)
#
# Examples:
#   ./plot_benchmarks.sh
#   ./plot_benchmarks.sh bench-output my_plots
#   ./plot_benchmarks.sh bench-output my_plots 1048576
# =============================================================================

# Exit immediately if a command fails
set -e

# --- Parse Arguments ---
BENCH_OUTPUT_DIR="${1:-./bench-output}"
OUTPUT_DIR="${2:-./plots}"
PARTITION_SIZE="${3:-1048576}"

# --- Configuration ---
VENV_PATH="./mac-venv"
PLOT_SCRIPT="./scripts/plot_compression_analysis.py"
TRADEOFF_SCRIPT="./scripts/plot_tradeoff_scatter.py"
TABLES_SCRIPT="./scripts/bench_tables.py"
TABLES_OUTPUT_DIR="./latex_tables"
MPL_CACHE_DIR="$OUTPUT_DIR/.matplotlib-cache"
# ---------------------

echo "========================================================================="
echo "Benchmark Plotting & Tables Script"
echo "========================================================================="
echo "Input directory:  $BENCH_OUTPUT_DIR"
echo "Plots directory:  $OUTPUT_DIR"
echo "Tables directory: $TABLES_OUTPUT_DIR"
echo "Partition size:   $PARTITION_SIZE"
echo "========================================================================="
echo ""

# --- Validation ---
if [ ! -d "$BENCH_OUTPUT_DIR" ]; then
    echo "Error: Benchmark output directory '$BENCH_OUTPUT_DIR' does not exist."
    echo "Run ./run_benchmarks.sh first to generate benchmark data."
    exit 1
fi

if [ ! -f "$PLOT_SCRIPT" ]; then
    echo "Error: Plot script '$PLOT_SCRIPT' not found."
    exit 1
fi

if [ ! -f "$TABLES_SCRIPT" ]; then
    echo "Error: Tables script '$TABLES_SCRIPT' not found."
    exit 1
fi

# Check for Python virtual environment
if [ -d "$VENV_PATH" ]; then
    echo "Activating Python virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Attempting to use system Python..."
fi

# Check for required packages
echo "Checking Python dependencies..."
python3 -c "import matplotlib, numpy, pandas" 2>/dev/null || {
    echo "Error: Required Python packages not found."
    echo "Please install them with: pip install matplotlib numpy pandas"
    exit 1
}

echo "✓ All dependencies satisfied"
echo ""

# --- Run Plot Script ---
echo "Preparing Matplotlib cache directory..."
mkdir -p "$MPL_CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"

echo "Generating plots..."
echo ""
python3 "$PLOT_SCRIPT" "$BENCH_OUTPUT_DIR" "$OUTPUT_DIR"
python3 "$TRADEOFF_SCRIPT" "$BENCH_OUTPUT_DIR" "$OUTPUT_DIR"

echo ""
# --- Run Tables Script ---
echo "Generating tables..."
echo ""
python3 "$TABLES_SCRIPT" "$BENCH_OUTPUT_DIR" "$TABLES_OUTPUT_DIR" --partition_size "$PARTITION_SIZE"

echo ""
echo "========================================================================="
echo "✓ Plotting and Tables generation complete!"
echo "========================================================================="
echo "Plots saved in:  $OUTPUT_DIR"
echo "Tables saved in: $TABLES_OUTPUT_DIR"
echo ""
echo "Generated plots:"
echo "  1. compression_ratio_by_compressor_single_thread.(png/pdf)"
echo "  2. compression_ratio_by_compressor_multi_thread.(png/pdf)"
echo "  3. compression_throughput_by_compressor.(png/pdf)"
echo "  4. decompression_throughput_by_compressor.(png/pdf)"
echo "  5. multi_metric_comparison.(png/pdf)"
echo "  6. compression_ratio_vs_speed_linear.(png/pdf)"
echo "  7. compression_ratio_vs_speed_log.(png/pdf)"
echo ""
echo "Generated tables:"
echo "  1. table_compression_ratio.tex"
echo "  2. table_compression_throughput.tex"
echo "  3. table_random_access.tex"
echo "  4. competitors.tex (if available)"
echo ""
echo "For more details, see PLOTTING_GUIDE.md"
echo "========================================================================="


