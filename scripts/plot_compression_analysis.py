#!/usr/bin/env python3
"""
Compression Analysis Plotting Script
=====================================
This script analyzes compression benchmark results and generates plots showing:
1. Average compression percentage across datasets for different partition sizes
2. Average compression throughput across datasets for different partition sizes
3. Decompression throughput comparisons (multi-thread vs single-thread)

Both metrics are plotted separately for OpenMP-enabled and OpenMP-disabled runs.

Usage:
    python3 plot_compression_analysis.py [bench_output_dir] [output_dir]

Arguments:
    bench_output_dir - Directory containing benchmark JSON files (default: ./bench-output)
    output_dir       - Directory to save plots (default: ./plots)
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, output_path, filename_stem):
    """Save figure as PDF for publication-ready assets."""
    output_file = output_path / f"{filename_stem}.pdf"
    fig.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")


def parse_partition_size(benchmark_name):
    """Extract partition size from benchmark name."""
    # Example: "FileBasedCompressionBenchmark/B_GEF_Compression/file_idx:0/strategy:0/partition_size:512"
    parts = benchmark_name.split('/')
    for part in parts:
        if part.startswith('partition_size:'):
            return int(part.split(':')[1])
    return None


def parse_compressor_and_strategy(benchmark_name):
    """
    Extract compressor type and strategy from benchmark name.
    Returns a string like "B_GEF_APPROXIMATE", "U_GEF_OPTIMAL", "RLE_GEF", etc.
    """
    parts = benchmark_name.split('/')
    
    # Extract compressor type
    compressor_type = None
    for part in parts:
        if 'B_GEF_NO_RLE' in part or 'B_GEF_STAR' in part:
            compressor_type = 'B*_GEF'
            break
        elif 'B_GEF' in part:
            compressor_type = 'B_GEF'
            break
        elif 'U_GEF' in part:
            compressor_type = 'U_GEF'
            break
        elif 'RLE_GEF' in part:
            compressor_type = 'RLE_GEF'
            break
    
    if compressor_type is None:
        return None
    
    # Extract strategy if applicable
    if compressor_type == 'RLE_GEF':
        return 'RLE_GEF'  # No strategy for RLE_GEF
    
    for part in parts:
        if part.startswith('strategy:'):
            strategy_num = int(part.split(':')[1])
            strategy_name = 'Approximate' if strategy_num == 0 else 'Optimal'
            return f'{compressor_type}_{strategy_name}'
    
    return None


def calculate_compression_percentage(size_in_bytes, num_integers):
    """
    Calculate compression percentage as defined:
    compression_percentage = (size_in_bytes / (num_integers * 8)) * 100
    
    This represents the percentage of the original uncompressed size.
    Lower is better (more compression).
    """
    if num_integers == 0:
        return 0
    return (size_in_bytes / (num_integers * 8)) * 100


def format_partition_size(value, _pos=None):
    """Convert raw partition sizes to human-readable labels (e.g., 16K, 1M)."""
    if value <= 0:
        return ""
    units = ["", "K", "M", "G", "T"]
    idx = 0
    while value >= 1000 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    value_int = int(value)
    label_value = value_int if abs(value - value_int) < 1e-9 else round(value, 1)
    return f"{label_value}{units[idx]}"


def collect_partition_ticks(metric_dict):
    """Gather every partition size present in the metric dictionary."""
    ticks = set()
    for comp_data in metric_dict.values():
        for ps in comp_data.get('partition_sizes', []):
            if ps is not None:
                ticks.add(ps)
    return sorted(ticks)


def _create_nested_metric_dict():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


def load_benchmark_data(bench_output_dir):
    """
    Load all benchmark JSON files and extract relevant data.
    
    Returns:
        dict: Nested dictionary structure:
              {openmp_status: {dataset: {partition_size: {strategy: [data_points]}}}}
    """
    compression_data = _create_nested_metric_dict()
    decompression_data = _create_nested_metric_dict()
    lookup_data = _create_nested_metric_dict()
    
    bench_path = Path(bench_output_dir)
    json_files = list(bench_path.glob("*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {bench_output_dir}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        print(f"  Processing: {json_file.name}")
        
        # Determine OpenMP status and dataset from filename
        # Format: <DATASET>_with_omp.json or <DATASET>_no_omp.json
        filename = json_file.stem  # Remove .json
        if filename.endswith('_with_omp'):
            openmp_status = 'with_omp'
            dataset = filename.replace('_with_omp', '')
        elif filename.endswith('_no_omp'):
            openmp_status = 'no_omp'
            dataset = filename.replace('_no_omp', '')
        else:
            print(f"    Warning: Skipping {json_file.name} - unknown format")
            continue
        
        # Load JSON
        try:
            with open(json_file, 'r') as f:
                # Skip any leading text lines that are not JSON
                lines = f.readlines()
                json_start = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        json_start = i
                        break
                json_content = ''.join(lines[json_start:])
                content = json.loads(json_content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    Error: Failed to parse {json_file.name}: {e}")
            continue
        
        # Extract benchmark data
        benchmarks = content.get('benchmarks', [])
        for bench in benchmarks:
            # Skip aggregate results (like mean, median, etc.)
            if bench.get('run_type') != 'iteration':
                continue
            
            # Extract fields
            name = bench.get('name', '')
            name_parts = name.split('/')
            if len(name_parts) < 2:
                continue

            benchmark_family = name_parts[1]
            partition_size = parse_partition_size(name)
            compressor_key = parse_compressor_and_strategy(name)
            if partition_size is None or compressor_key is None:
                continue

            if benchmark_family.endswith('_Compression'):
                size_in_bytes = bench.get('size_in_bytes', 0)
                num_integers = bench.get('num_integers', 0)
                throughput = bench.get('compression_throughput_MBs', 0)
                
                if (size_in_bytes <= 0 or
                        num_integers <= 0 or
                        throughput <= 0):
                    continue
                
                compression_pct = calculate_compression_percentage(size_in_bytes, num_integers)
                
                compression_data[openmp_status][dataset][partition_size][compressor_key].append({
                    'compression_percentage': compression_pct,
                    'throughput': throughput,
                    'size_in_bytes': size_in_bytes,
                    'num_integers': num_integers
                })
            elif benchmark_family.endswith('_Decompression'):
                throughput = bench.get('decompression_throughput_MBs', 0)
                if throughput <= 0:
                    continue
                decompression_data[openmp_status][dataset][partition_size][compressor_key].append({
                    'throughput': throughput
                })
            elif benchmark_family.endswith('_Lookup'):
                throughput = bench.get('lookup_throughput_MBs', 0)
                if throughput <= 0:
                    continue
                lookup_data[openmp_status][dataset][partition_size][compressor_key].append({
                    'throughput': throughput
                })
    
    return compression_data, decompression_data, lookup_data


def aggregate_by_compressor(dataset_map):
    """
    Calculate average compression percentage and throughput for each compressor type
    and strategy combination across all datasets.
    
    Returns:
        dict: {compressor_key: {'partition_sizes': [...], 'compression_pct': [...], 'throughput': [...]}}
    """
    # Collect data by compressor type and partition size
    compressor_data = defaultdict(lambda: defaultdict(lambda: {'compression_pct': [], 'throughput': []}))
    
    for dataset, dataset_data in dataset_map.items():
        for partition_size, compressor_dict in dataset_data.items():
            for compressor_key, points in compressor_dict.items():
                for point in points:
                    compression_value = point.get('compression_percentage')
                    if compression_value is not None:
                        compressor_data[compressor_key][partition_size]['compression_pct'].append(
                            compression_value
                        )
                    throughput_value = point.get('throughput')
                    if throughput_value is not None:
                        # Convert bytes/s to MB/s (decimal units)
                        compressor_data[compressor_key][partition_size]['throughput'].append(
                            throughput_value / 1000000.0
                        )
    
    # Calculate averages for each compressor
    results = {}
    for compressor_key, partition_dict in compressor_data.items():
        partition_sizes = []
        avg_compression_pct = []
        avg_throughput_mbs = []
        
        for ps in sorted(partition_dict.keys()):
            comp_values = partition_dict[ps]['compression_pct']
            throughput_values = partition_dict[ps]['throughput']
            
            comp_mean = float(np.mean(comp_values)) if comp_values else None
            throughput_mean = float(np.mean(throughput_values)) if throughput_values else None
            
            if comp_mean is None and throughput_mean is None:
                continue
            
            partition_sizes.append(ps)
            avg_compression_pct.append(comp_mean)
            avg_throughput_mbs.append(throughput_mean)
        
        if partition_sizes:
            results[compressor_key] = {
                'partition_sizes': partition_sizes,
                'compression_pct': avg_compression_pct,
                'throughput': avg_throughput_mbs
            }
    
    return results


def build_status_results(metric_data):
    results = {}
    for status, dataset_map in metric_data.items():
        if dataset_map:
            aggregated = aggregate_by_compressor(dataset_map)
            if aggregated:
                results[status] = aggregated
    return results


def plot_metric_lines(ax, metric_dict, value_key, compressor_styles, y_label, title):
    plotted = False
    partition_ticks = collect_partition_ticks(metric_dict)
    
    # Create mapping from partition size to index for equally spaced x-axis
    ps_to_idx = {ps: i for i, ps in enumerate(partition_ticks)} if partition_ticks else {}
    
    for compressor_key, comp_data in sorted(metric_dict.items()):
        if compressor_key not in compressor_styles:
            continue
        values = comp_data[value_key]
        partition_sizes = comp_data['partition_sizes']
        filtered_indices = []
        filtered_values = []
        for ps, val in zip(partition_sizes, values):
            if val is None or ps not in ps_to_idx:
                continue
            filtered_indices.append(ps_to_idx[ps])
            filtered_values.append(val)
        if not filtered_indices:
            continue
        style = compressor_styles[compressor_key]
        ax.plot(filtered_indices, filtered_values,
                marker=style['marker'], linestyle=style['linestyle'],
                linewidth=2, markersize=7, color=style['color'],
                label=style['label'], alpha=0.85)
        plotted = True
    
    ax.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if plotted:
        # Use categorical (equally spaced) x-axis with formatted labels
        if partition_ticks:
            ax.set_xticks(range(len(partition_ticks)))
            ax.set_xticklabels([format_partition_size(ps) for ps in partition_ticks], rotation=45, ha='right')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_axis_off()


def plot_status_subplot(ax, status_data, compressor_styles, y_label, title, legend_loc='best'):
    """
    Plot throughput values for a specific OpenMP status on a provided axis.
    """
    if not status_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_axis_off()
        return

    plotted = False
    partition_ticks = collect_partition_ticks(status_data)
    
    # Create mapping from partition size to index for equally spaced x-axis
    ps_to_idx = {ps: i for i, ps in enumerate(partition_ticks)} if partition_ticks else {}
    
    for compressor_key, comp_data in sorted(status_data.items()):
        if compressor_key not in compressor_styles:
            continue
        style = compressor_styles[compressor_key]
        partition_sizes = comp_data.get('partition_sizes', [])
        throughputs = comp_data.get('throughput', [])
        if not partition_sizes or not throughputs:
            continue
        
        # Convert partition sizes to indices for equally spaced plotting
        indices = []
        filtered_throughputs = []
        for ps, tp in zip(partition_sizes, throughputs):
            if ps in ps_to_idx and tp is not None:
                indices.append(ps_to_idx[ps])
                filtered_throughputs.append(tp)
        
        if not indices:
            continue
        
        ax.plot(indices, filtered_throughputs,
                marker=style['marker'], linestyle=style['linestyle'],
                linewidth=2, markersize=7, color=style['color'],
                label=style['label'], alpha=0.8)
        plotted = True

    if plotted:
        ax.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        # Use categorical (equally spaced) x-axis with formatted labels
        if partition_ticks:
            ax.set_xticks(range(len(partition_ticks)))
            ax.set_xticklabels([format_partition_size(ps) for ps in partition_ticks], rotation=45, ha='right')
        # Add top margin so legend doesn't overlap with high values
        ax.margins(y=0.15)
        ax.legend(fontsize=10, loc=legend_loc)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_axis_off()


def create_plots(compression_data, decompression_data, lookup_data, output_dir):
    """Create and save the plots showing each compressor separately."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    compressor_styles = {
        'B_GEF_Approximate': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'B-GEF (Approximate)'},
        'B_GEF_Optimal': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'B-GEF (Optimal)'},
        'U_GEF_Approximate': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'U-GEF (Approximate)'},
        'U_GEF_Optimal': {'color': '#d62728', 'marker': 'v', 'linestyle': '--', 'label': 'U-GEF (Optimal)'},
        'B*_GEF_Approximate': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-.', 'label': 'B*-GEF (Approximate)'},
        'B*_GEF_Optimal': {'color': '#8c564b', 'marker': 'p', 'linestyle': '-.', 'label': 'B*-GEF (Optimal)'},
        'RLE_GEF': {'color': '#e377c2', 'marker': 'h', 'linestyle': ':', 'label': 'RLE-GEF'},
    }
    
    status_titles = {
        'with_omp': 'Multi-thread',
        'no_omp': 'Single-thread'
    }
    
    compression_results = build_status_results(compression_data)
    decompression_results = build_status_results(decompression_data)
    lookup_results = build_status_results(lookup_data)
    
    if not compression_results:
        print("Error: No data to plot")
        return
    
    # Compression ratio plot (single-thread data only)
    ratio_status = 'no_omp'
    ratio_data = compression_results.get(ratio_status)
    if ratio_data:
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_metric_lines(
            ax,
            ratio_data,
            'compression_pct',
            compressor_styles,
            'Compression Ratio (%)',
            'Compression Ratio vs Partition Size'
        )
        plt.tight_layout()
        save_figure(fig, output_path, 'compression_ratio_by_compressor')
        plt.close(fig)
    else:
        print("Warning: No single-thread compression-ratio data available for plotting.")
    
    # Compression throughput plots (saved separately per threading mode)
    with_omp_data = compression_results.get('with_omp')
    if with_omp_data:
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_status_subplot(
            ax,
            with_omp_data,
            compressor_styles,
            'Compression Throughput (MB/s)',
            f"Compression Throughput vs Partition Size ({status_titles['with_omp']})"
        )
        plt.tight_layout()
        save_figure(fig, output_path, 'compression_throughput_by_compressor_mt')
        plt.close(fig)
    else:
        print("Warning: No multi-thread compression throughput data available for plotting.")

    no_omp_data = compression_results.get('no_omp')
    if no_omp_data:
        fig, ax = plt.subplots(figsize=(9, 7))
        plot_status_subplot(
            ax,
            no_omp_data,
            compressor_styles,
            'Compression Throughput (MB/s)',
            f"Compression Throughput vs Partition Size ({status_titles['no_omp']})",
            legend_loc='upper right'
        )
        plt.tight_layout()
        save_figure(fig, output_path, 'compression_throughput_by_compressor_st')
        plt.close(fig)
    else:
        print("Warning: No single-thread compression throughput data available for plotting.")

    # Decompression throughput plots
    if decompression_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        plot_status_subplot(
            ax1,
            decompression_results.get('with_omp'),
            compressor_styles,
            'Decompression Throughput (MB/s)',
            f"Decompression Throughput vs Partition Size ({status_titles['with_omp']})"
        )
        plot_status_subplot(
            ax2,
            decompression_results.get('no_omp'),
            compressor_styles,
            'Decompression Throughput (MB/s)',
            f"Decompression Throughput vs Partition Size ({status_titles['no_omp']})"
        )
        plt.tight_layout()
        save_figure(fig, output_path, 'decompression_throughput_by_compressor')
        plt.close(fig)
    else:
        print("Warning: No decompression throughput data available for plotting.")

    # Random access throughput plots (lookup benchmarks)
    if lookup_results:
        plotted_lookup = False
        for status_key in ('with_omp', 'no_omp'):
            status_data = lookup_results.get(status_key)
            if not status_data:
                continue
            fig, ax = plt.subplots(figsize=(9, 7))
            plot_status_subplot(
                ax,
                status_data,
                compressor_styles,
                'Random Access Throughput (MB/s)',
                f"Random Access Throughput vs Partition Size ({status_titles.get(status_key, status_key)})"
            )
            plt.tight_layout()
            suffix = 'mt' if status_key == 'with_omp' else 'st'
            save_figure(fig, output_path, f'random_access_throughput_by_compressor_{suffix}')
            plt.close(fig)
            plotted_lookup = True
        if not plotted_lookup:
            print("Warning: Random access data exists but no usable series were found for plotting.")
    else:
        print("Warning: No random access throughput data available for plotting.")
    
    # Print summary statistics by compressor
    print("\n" + "=" * 80)
    print("SUMMARY BY COMPRESSOR")
    print("=" * 80)
    
    for openmp_status in ['with_omp', 'no_omp']:
        if openmp_status not in compression_results:
            continue
            
        status_label = status_titles.get(openmp_status, openmp_status).upper()
        print(f"\n{'='*80}")
        print(f"{status_label}")
        print(f"{'='*80}")
        
        for compressor_key in sorted(compression_results[openmp_status].keys()):
            comp_data = compression_results[openmp_status][compressor_key]
            print(f"\n{compressor_key}:")
            print("-" * 80)
            print(f"{'Partition':<12} {'Compression %':<18} {'Throughput (MB/s)':<20}")
            print("-" * 80)
            
            for ps, comp_pct, throughput in zip(
                comp_data['partition_sizes'],
                comp_data['compression_pct'],
                comp_data['throughput']
            ):
                print(f"{ps:<12} {comp_pct:<18.2f} {throughput:<20.2f}")
    
    print("\n" + "=" * 80)
    if lookup_results:
        print("RANDOM ACCESS SUMMARY (lookup throughput)")
        print("=" * 80)
        for openmp_status in ('with_omp', 'no_omp'):
            if openmp_status not in lookup_results:
                continue
            status_label = status_titles.get(openmp_status, openmp_status).upper()
            print(f"\n{'='*80}")
            print(f"{status_label}")
            print(f"{'='*80}")
            for compressor_key in sorted(lookup_results[openmp_status].keys()):
                comp_data = lookup_results[openmp_status][compressor_key]
                print(f"\n{compressor_key}:")
                print("-" * 80)
                print(f"{'Partition':<12} {'Lookup Throughput (MB/s)':<26}")
                print("-" * 80)
                for ps, throughput in zip(
                    comp_data['partition_sizes'],
                    comp_data['throughput']
                ):
                    value = throughput if throughput is not None else float('nan')
                    print(f"{ps:<12} {value:<26.2f}")


def main():
    """Main entry point."""
    # Parse arguments
    bench_output_dir = sys.argv[1] if len(sys.argv) > 1 else './bench-output'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './plots'
    
    print("=" * 80)
    print("Compression Analysis Plotting Script")
    print("=" * 80)
    print(f"Benchmark directory: {bench_output_dir}")
    print(f"Output directory:    {output_dir}")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading benchmark data...")
    compression_data, decompression_data, lookup_data = load_benchmark_data(bench_output_dir)
    
    if (not compression_data and not decompression_data and not lookup_data):
        print("Error: No data loaded")
        sys.exit(1)
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(compression_data, decompression_data, lookup_data, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

