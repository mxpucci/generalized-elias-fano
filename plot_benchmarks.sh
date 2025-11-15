#!/bin/bash

# =============================================================================
# Benchmark Plotting Script
# =============================================================================
# This script generates plots from compression benchmark results.
#
# Usage:
#   ./plot_benchmarks.sh [bench_output_dir] [output_dir]
#
# Arguments:
#   bench_output_dir - Directory containing benchmark JSON files (default: ./bench-output)
#   output_dir       - Directory to save plots (default: ./plots)
#
# Examples:
#   ./plot_benchmarks.sh
#   ./plot_benchmarks.sh bench-output my_plots
# =============================================================================

# Exit immediately if a command fails
set -e

# --- Parse Arguments ---
BENCH_OUTPUT_DIR="${1:-./bench-output}"
OUTPUT_DIR="${2:-./plots}"

# --- Configuration ---
VENV_PATH="./mac-venv"
PLOT_SCRIPT="./scripts/plot_compression_analysis.py"
# ---------------------

echo "========================================================================="
echo "Benchmark Plotting Script"
echo "========================================================================="
echo "Input directory:  $BENCH_OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
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
python3 -c "import matplotlib, numpy" 2>/dev/null || {
    echo "Error: Required Python packages not found."
    echo "Please install them with: pip install matplotlib numpy"
    exit 1
}

echo "✓ All dependencies satisfied"
echo ""

# --- Run Plot Script ---
echo "Generating plots..."
echo ""
python3 "$PLOT_SCRIPT" "$BENCH_OUTPUT_DIR" "$OUTPUT_DIR"

echo ""
echo "========================================================================="
echo "✓ Plotting complete!"
echo "========================================================================="
echo "Plots saved in: $OUTPUT_DIR"
echo ""
echo "Generated plots:"
echo "  1. compression_percentage_vs_partition_size.png"
echo "  2. compression_throughput_vs_partition_size.png"
echo "  3. compression_comparison.png"
echo ""
echo "For more details, see PLOTTING_GUIDE.md"
echo "========================================================================="


