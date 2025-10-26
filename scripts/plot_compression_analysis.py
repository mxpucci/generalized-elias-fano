#!/usr/bin/env python3
"""
Compression Analysis Plotting Script
=====================================
This script analyzes compression benchmark results and generates plots showing:
1. Average compression percentage across datasets for different partition sizes
2. Average compression throughput across datasets for different partition sizes

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


def load_benchmark_data(bench_output_dir):
    """
    Load all benchmark JSON files and extract relevant data.
    
    Returns:
        dict: Nested dictionary structure:
              {openmp_status: {dataset: {partition_size: {strategy: [data_points]}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
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
            partition_size = parse_partition_size(name)
            compressor_key = parse_compressor_and_strategy(name)
            size_in_bytes = bench.get('size_in_bytes', 0)
            num_integers = bench.get('num_integers', 0)
            throughput = bench.get('compression_throughput_MBs', 0)
            
            if partition_size is None or compressor_key is None:
                continue
            
            # Calculate compression percentage
            compression_pct = calculate_compression_percentage(size_in_bytes, num_integers)
            
            # Store data grouped by compressor type
            data[openmp_status][dataset][partition_size][compressor_key].append({
                'compression_percentage': compression_pct,
                'throughput': throughput,
                'size_in_bytes': size_in_bytes,
                'num_integers': num_integers
            })
    
    return data


def aggregate_by_compressor(data, openmp_status):
    """
    Calculate average compression percentage and throughput for each compressor type
    and strategy combination across all datasets.
    
    Returns:
        dict: {compressor_key: {'partition_sizes': [...], 'compression_pct': [...], 'throughput': [...]}}
    """
    # Collect data by compressor type and partition size
    compressor_data = defaultdict(lambda: defaultdict(lambda: {'compression_pct': [], 'throughput': []}))
    
    for dataset, dataset_data in data[openmp_status].items():
        for partition_size, compressor_dict in dataset_data.items():
            for compressor_key, points in compressor_dict.items():
                for point in points:
                    compressor_data[compressor_key][partition_size]['compression_pct'].append(
                        point['compression_percentage']
                    )
                    # Convert bytes/s to MB/s (decimal units)
                    compressor_data[compressor_key][partition_size]['throughput'].append(
                        point['throughput'] / 1000000.0
                    )
    
    # Calculate averages for each compressor
    results = {}
    for compressor_key, partition_dict in compressor_data.items():
        partition_sizes = sorted(partition_dict.keys())
        avg_compression_pct = []
        avg_throughput_mbs = []
        
        for ps in partition_sizes:
            avg_compression_pct.append(np.mean(partition_dict[ps]['compression_pct']))
            avg_throughput_mbs.append(np.mean(partition_dict[ps]['throughput']))
        
        results[compressor_key] = {
            'partition_sizes': partition_sizes,
            'compression_pct': avg_compression_pct,
            'throughput': avg_throughput_mbs
        }
    
    return results


def create_plots(data, output_dir):
    """Create and save the plots showing each compressor separately."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define colors and markers for each compressor type
    compressor_styles = {
        'B_GEF_Approximate': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'B-GEF (Approximate)'},
        'B_GEF_Optimal': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'B-GEF (Optimal)'},
        'U_GEF_Approximate': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'U-GEF (Approximate)'},
        'U_GEF_Optimal': {'color': '#d62728', 'marker': 'v', 'linestyle': '--', 'label': 'U-GEF (Optimal)'},
        'B*_GEF_Approximate': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-.', 'label': 'B*-GEF (Approximate)'},
        'B*_GEF_Optimal': {'color': '#8c564b', 'marker': 'p', 'linestyle': '-.', 'label': 'B*-GEF (Optimal)'},
        'RLE_GEF': {'color': '#e377c2', 'marker': 'h', 'linestyle': ':', 'label': 'RLE-GEF'},
    }
    
    # Aggregate data for both OpenMP statuses
    results = {}
    for openmp_status in ['with_omp', 'no_omp']:
        if openmp_status in data:
            results[openmp_status] = aggregate_by_compressor(data, openmp_status)
    
    if not results:
        print("Error: No data to plot")
        return
    
    # =========================================================================
    # Plot 1: Compression Percentage vs Partition Size (With OpenMP)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot with OpenMP
    if 'with_omp' in results:
        for compressor_key, comp_data in sorted(results['with_omp'].items()):
            if compressor_key not in compressor_styles:
                continue
            style = compressor_styles[compressor_key]
            ax1.plot(comp_data['partition_sizes'], comp_data['compression_pct'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2, markersize=7, color=style['color'],
                    label=style['label'], alpha=0.8)
        
        ax1.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Compression Ratio (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Compression Ratio vs Partition Size (With OpenMP)',
                     fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
    
    # Plot without OpenMP
    if 'no_omp' in results:
        for compressor_key, comp_data in sorted(results['no_omp'].items()):
            if compressor_key not in compressor_styles:
                continue
            style = compressor_styles[compressor_key]
            ax2.plot(comp_data['partition_sizes'], comp_data['compression_pct'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2, markersize=7, color=style['color'],
                    label=style['label'], alpha=0.8)
        
        ax2.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Compression Ratio (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Compression Ratio vs Partition Size (Without OpenMP)',
                     fontsize=14, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    output_file = output_path / 'compression_ratio_by_compressor.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # =========================================================================
    # Plot 2: Compression Throughput vs Partition Size
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot with OpenMP
    if 'with_omp' in results:
        for compressor_key, comp_data in sorted(results['with_omp'].items()):
            if compressor_key not in compressor_styles:
                continue
            style = compressor_styles[compressor_key]
            ax1.plot(comp_data['partition_sizes'], comp_data['throughput'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2, markersize=7, color=style['color'],
                    label=style['label'], alpha=0.8)
        
        ax1.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Compression Throughput (MB/s)', fontsize=13, fontweight='bold')
        ax1.set_title('Compression Throughput vs Partition Size (With OpenMP)',
                     fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
    
    # Plot without OpenMP
    if 'no_omp' in results:
        for compressor_key, comp_data in sorted(results['no_omp'].items()):
            if compressor_key not in compressor_styles:
                continue
            style = compressor_styles[compressor_key]
            ax2.plot(comp_data['partition_sizes'], comp_data['throughput'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2, markersize=7, color=style['color'],
                    label=style['label'], alpha=0.8)
        
        ax2.set_xlabel('Partition Size', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Compression Throughput (MB/s)', fontsize=13, fontweight='bold')
        ax2.set_title('Compression Throughput vs Partition Size (Without OpenMP)',
                     fontsize=14, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    output_file = output_path / 'compression_throughput_by_compressor.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Print summary statistics by compressor
    print("\n" + "=" * 80)
    print("SUMMARY BY COMPRESSOR")
    print("=" * 80)
    
    for openmp_status in ['with_omp', 'no_omp']:
        if openmp_status not in results:
            continue
            
        status_label = "WITH OpenMP" if openmp_status == 'with_omp' else "WITHOUT OpenMP"
        print(f"\n{'='*80}")
        print(f"{status_label}")
        print(f"{'='*80}")
        
        for compressor_key in sorted(results[openmp_status].keys()):
            comp_data = results[openmp_status][compressor_key]
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
    data = load_benchmark_data(bench_output_dir)
    
    if not data:
        print("Error: No data loaded")
        sys.exit(1)
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(data, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

