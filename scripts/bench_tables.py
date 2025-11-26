#!/usr/bin/env python3
import json
import os
import glob
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

# Try to import the competitor data, handle if the file doesn't exist.
try:
    from competitor_data import competitor_benchmarks
    COMPETITOR_DATA_AVAILABLE = True
except ImportError:
    COMPETITOR_DATA_AVAILABLE = False
    print("Warning: 'competitor_data.py' not found. Skipping generation of combined table.")


def parse_partition_size(benchmark_name):
    """Extract partition size from benchmark name."""
    parts = benchmark_name.split('/')
    for part in parts:
        if part.startswith('partition_size:'):
            return int(part.split(':')[1])
    return None


def parse_gef_data(directory, target_partition_size=None):
    """
    Parses the GEF-specific benchmark data from JSON files.
    This function is for the GEF-only tables.
    """
    compressor_name_map = {
        "RLE_GEF_Compression": "RLE-GEF", "RLE_GEF_Lookup": "RLE-GEF",
        "U_GEF_Compression": "U-GEF", "U_GEF_Lookup": "U-GEF",
        "B_GEF_Compression": "B-GEF", "B_GEF_Lookup": "B-GEF",
        "B_GEF_NO_RLE_Compression": r"$\mathrm{B^*-GEF}$", "B_GEF_NO_RLE_Lookup": r"$\mathrm{B^*-GEF}$",
    }
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    json_files = glob.glob(os.path.join(directory, '*.json'))
    if not json_files:
        print(f"Error: No JSON files found in directory '{directory}'")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for file_path in json_files:
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        # Skip multi-threaded (OpenMP) results. Tables should use single-threaded data.
        if dataset_name.endswith('_with_omp'):
            continue

        # Handle _no_omp suffix in dataset name
        if dataset_name.endswith('_no_omp'):
            dataset_name = dataset_name.replace('_no_omp', '')

        with open(file_path, 'r') as f:
            try:
                # Some benchmark files have a preamble line before the JSON object.
                # We need to find the start of the JSON object.
                file_content = f.read()
                json_start_idx = file_content.find('{')
                if json_start_idx != -1:
                    content = json.loads(file_content[json_start_idx:])
                else:
                    print(f"Warning: No JSON object found in {file_path}")
                    continue
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue

        for bench in content.get('benchmarks', []):
            # Filter by partition size if specified
            if target_partition_size is not None:
                p_size = parse_partition_size(bench['name'])
                # If we can't find a partition size in the name, it might be one of the aggregate runs or differently named.
                # However, for standard compression/lookup benchmarks in this suite, partition size is in the name.
                if p_size is not None and p_size != target_partition_size:
                    continue

            compressor_raw = bench['name'].split('/')[1]
            # Map raw compressor name to the base name (e.g. "B_GEF_Compression" -> "B-GEF")
            compressor_base = compressor_name_map.get(compressor_raw, compressor_raw)
            
            # Strategy extraction needs to be robust.
            # Example label: "AP.bin/SDSL/APPROXIMATE/8192" -> strategy is "APPROXIMATE"
            # Example label: "AP.bin/SDSL/8192" -> no strategy (RLE) -> default to "Standard" or handle appropriately.
            label_parts = bench.get('label', '').split('/')
            strategy = 'Standard'
            if len(label_parts) > 2:
                # Check if the 3rd part is a known strategy or a number (partition size)
                potential_strategy = label_parts[2]
                if potential_strategy == "BRUTE_FORCE":
                    strategy = "Optimal"
                elif potential_strategy in ["APPROXIMATE", "OPTIMAL", "Standard"]:
                     strategy = potential_strategy if potential_strategy != "OPTIMAL" else "Optimal"
                else:
                     # Likely RLE or similar where strategy is not explicit in the same position
                     strategy = 'Standard'
            
            # Force RLE-GEF to have 'Optimal' strategy label as requested
            if compressor_base == "RLE-GEF":
                strategy = "Optimal"

            # Debugging print (can be removed later)
            # print(f"Found: {dataset_name}, {compressor_base}, {strategy}")

            if 'bpi' in bench:
                data['compression_ratio'][dataset_name][compressor_base][strategy] = (bench['bpi'] / 64) * 100
            if 'compression_throughput_MBs' in bench:
                # Note: The JSON key says _MBs, but in these files (see values like 1.24e+08), it is actually Bytes/s.
                # We divide by 1e6 to get MB/s.
                data['comp_throughput'][dataset_name][compressor_base][strategy] = bench['compression_throughput_MBs'] / 1e6
            if 'items_per_second' in bench:
                # Calculate MB/s: items_per_second * 8 bytes / 1e6 (for MB)
                # If the input is already in MB/s, we need to check.
                # The benchmark code outputs items_per_second. 
                # Assuming each item is 64-bit (8 bytes).
                data['random_access_speed'][dataset_name][compressor_base][strategy] = (bench['items_per_second'] * 8) / 1e6

    # --- Create DataFrames for each metric using pivot_table for robustness ---
    ratio_records = [{'dataset': ds, 'compressor': c, 'strategy': s, 'value': v} for ds, comps in data['compression_ratio'].items() for c, strats in comps.items() for s, v in strats.items()]
    ratio_df = pd.DataFrame(ratio_records).pivot_table(index='dataset', columns=['compressor', 'strategy'], values='value') if ratio_records else pd.DataFrame()

    comp_records = [{'dataset': ds, 'compressor': c, 'strategy': s, 'value': v} for ds, comps in data['comp_throughput'].items() for c, strats in comps.items() for s, v in strats.items()]
    comp_df = pd.DataFrame(comp_records).pivot_table(index='dataset', columns=['compressor', 'strategy'], values='value') if comp_records else pd.DataFrame()

    access_records = [{'dataset': ds, 'compressor': c, 'strategy': s, 'value': v} for ds, comps in data['random_access_speed'].items() for c, strats in comps.items() for s, v in strats.items()]
    access_df = pd.DataFrame(access_records).pivot_table(index='dataset', columns=['compressor', 'strategy'], values='value') if access_records else pd.DataFrame()

    return ratio_df, comp_df, access_df

def generate_gef_table(df, caption, label, unit_name, highlight_best=None, compressor_order=None):
    """Generates a more compact LaTeX table for GEF variants using table* with font and spacing adjustments."""
    if df.empty: return f"% No data for {caption}\n"
    compressors_from_data = df.columns.get_level_values(0).unique()
    compressors = [c for c in compressor_order if c in compressors_from_data] if compressor_order else compressors_from_data
    strategy_order = ['APPROXIMATE', 'Optimal']

    latex_string = (
        f"\\begin{{table*}}[htbp]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\centering\n"
        f"\\small\n"  # Use smaller font
        f"\\setlength{{\\tabcolsep}}{{4pt}} % Reduce space between columns\n"
    )
    column_format = "l" + "c" * len(df.columns)
    latex_string += f"\\begin{{tabular}}{{{column_format}}}\n\\toprule\n"

    # Header Row 1: Compressor Names
    latex_string += "\\textbf{Dataset}"
    for comp in compressors:
        present_strategies = [s for s in strategy_order if (comp, s) in df.columns]
        num_strategies = len(present_strategies)
        if num_strategies == 0: continue
        comp_display = comp.replace('_', ' ').replace('Compression', '').replace('Lookup', '').strip()
        if num_strategies > 1:
            latex_string += f" & \\multicolumn{{{num_strategies}}}{{c}}{{\\textbf{{{comp_display}}}}}"
        else:
            latex_string += f" & \\textbf{{{comp_display}}}"
    latex_string += " \\\\\n"

    # Header Row 2: Strategy Names
    cmidrules = ""
    latex_string += f"\\textbf{{({unit_name})}}"
    col_idx = 1
    for comp in compressors:
        present_strategies = [s for s in strategy_order if (comp, s) in df.columns]
        num_strategies = len(present_strategies)
        if num_strategies == 0: continue
        for strat in present_strategies:
            strat_display = strat.replace('_', ' ').title() if strat != 'Standard' else ' '
            # If strat is 'Optimal', title() keeps it 'Optimal'.
            latex_string += f" & {strat_display}"
        if num_strategies > 1:
            cmidrules += f"\\cmidrule(lr){{{col_idx+1}-{col_idx+num_strategies}}} "
        col_idx += num_strategies
    latex_string += f" \\\\\n{cmidrules}\n\\midrule\n"

    # Table Body
    for index, row in df.iterrows():
        best_value = row.min() if highlight_best == 'min' else (row.max() if highlight_best == 'max' else None)
        safe_index = index.replace('_', '\\_')
        latex_string += f"{safe_index}"
        for comp in compressors:
            present_strategies = [s for s in strategy_order if (comp, s) in df.columns]
            for strat in present_strategies:
                if (comp, strat) in row.index and pd.notna(row[(comp, strat)]):
                    value = row[(comp, strat)]
                    if best_value is not None and abs(value - best_value) < 1e-9:
                        latex_string += f" & \\textbf{{{value:.2f}}}"
                    else:
                        latex_string += f" & {value:.2f}"
                else:
                    latex_string += " & -"
        latex_string += " \\\\\n"
    latex_string += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    return latex_string

def generate_combined_table_section(df, title, highlight_mode):
    """Generates a section of the combined LaTeX table, escaping dataset names."""
    general_purpose_cols = ['Xz', 'Brotli', 'Zstd', 'Lz4', 'Snappy']
    special_purpose_cols = ['Chimp128', 'Chimp', 'TSXor', 'DAC', 'Gorilla', 'LeCo', 'ALP', 'NeaTS', 'Best GEF']
    gp_cols_present = [c for c in general_purpose_cols if c in df.columns]
    sp_cols_present = [c for c in special_purpose_cols if c in df.columns]

    latex_string = f"\\multicolumn{{{1 + len(gp_cols_present) + len(sp_cols_present)}}}{{l}}{{\\textbf{{{title}}}}} \\\\\n\\midrule\n"

    for index, row in df.iterrows():
        safe_index = str(index).replace('_', '\\_')
        latex_string += f"{safe_index}"

        row_valid = row.dropna()
        if row_valid.empty:
            for _ in gp_cols_present + sp_cols_present:
                latex_string += " & -"
            latex_string += " \\\\\n"
            continue

        overall_best = row_valid.min() if highlight_mode == 'min' else row_valid.max()
        gp_valid = row_valid.reindex(gp_cols_present).dropna()
        sp_valid = row_valid.reindex(sp_cols_present).dropna()

        gp_best = gp_valid.min() if highlight_mode == 'min' and not gp_valid.empty else (gp_valid.max() if not gp_valid.empty else None)
        sp_best = sp_valid.min() if highlight_mode == 'min' and not sp_valid.empty else (sp_valid.max() if not sp_valid.empty else None)

        for col in gp_cols_present + sp_cols_present:
            value = row[col]
            if pd.isna(value):
                latex_string += " & -"
                continue

            formatted_value = f"{value:.2f}"
            is_overall_best = abs(value - overall_best) < 1e-9
            is_gp_best = gp_best is not None and abs(value - gp_best) < 1e-9
            is_sp_best = sp_best is not None and abs(value - sp_best) < 1e-9

            if is_overall_best:
                formatted_value = f"\\underline{{\\textbf{{{formatted_value}}}}}"
            elif is_gp_best or is_sp_best:
                formatted_value = f"\\textbf{{{formatted_value}}}"

            latex_string += f" & {formatted_value}"
        latex_string += " \\\\\n"
    return latex_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate benchmark tables.')
    parser.add_argument('benchmark_dir', nargs='?', default='benchmark_results', help='Directory containing benchmark JSON files')
    parser.add_argument('output_dir', nargs='?', default='latex_tables', help='Directory to save LaTeX tables')
    parser.add_argument('--partition_size', type=int, default=32000, help='Partition size to filter (default: 32000)')
    
    args = parser.parse_args()
    
    benchmark_dir = args.benchmark_dir
    gef_output_dir = args.output_dir
    combined_output_dir = args.output_dir

    if not os.path.exists(gef_output_dir):
        os.makedirs(gef_output_dir)
        print(f"Created output directory: {gef_output_dir}")
    # combined_output_dir is same as gef_output_dir

    TABLE_PARTITION_SIZE = 32000
    if args.partition_size != TABLE_PARTITION_SIZE:
        print(f"Overriding requested partition size {args.partition_size} with {TABLE_PARTITION_SIZE} for table generation.")

    target_partition_size = TABLE_PARTITION_SIZE

    print(f"Processing benchmarks from: {benchmark_dir}")
    print(f"Outputting tables to:     {gef_output_dir}")
    print(f"Filtering partition size: {target_partition_size}")

    # --- PART 1: Generate GEF-only tables ---
    # All GEF tables should rely on partition size 32000 for consistency across metrics.
    gef_ratio_df, gef_comp_df, gef_access_df = parse_gef_data(benchmark_dir, target_partition_size)
    
    gef_compressor_order = ["RLE-GEF", "U-GEF", "B-GEF", r"$\mathrm{B^*-GEF}$"]

    print("Generating GEF-only tables with compact formatting...")
    with open(os.path.join(gef_output_dir, "table_compression_ratio.tex"), 'w') as f:
        f.write(generate_gef_table(gef_ratio_df, "GEF Variants: Compression ratio (\\%). Lower is better.", "tab:gef_ratio", "\\%", 'min', gef_compressor_order))
    with open(os.path.join(gef_output_dir, "table_compression_throughput.tex"), 'w') as f:
        f.write(generate_gef_table(gef_comp_df, "GEF Variants: Compression throughput (MB/s). Higher is better.", "tab:gef_comp_throughput", "MB/s", 'max', gef_compressor_order))
    with open(os.path.join(gef_output_dir, "table_random_access.tex"), 'w') as f:
        f.write(generate_gef_table(gef_access_df, "GEF Variants: Random access speed (MB/s). Higher is better.", "tab:gef_access", "MB/s", 'max', gef_compressor_order))
    print(f"-> GEF-only tables saved in '{gef_output_dir}'.")

    # --- PART 2: Generate combined competitors.tex file ---
    if COMPETITOR_DATA_AVAILABLE:
        print("\nGenerating combined competitors.tex file using landscape and rotated headers...")

        # Find the GEF variant that yields the best compression ratio for each dataset.
        # We will use this variant for ALL metrics (decompression, random access) to ensure consistency.
        best_gef_variants = gef_ratio_df.idxmin(axis=1)
        
        best_gef_ratio = []
        best_gef_access = []
        
        for dataset in gef_ratio_df.index:
            if dataset in best_gef_variants.index:
                variant = best_gef_variants[dataset]
                # variant is a tuple: (CompressorName, Strategy)
                
                # Get Ratio
                val_ratio = gef_ratio_df.loc[dataset, variant]
                best_gef_ratio.append(val_ratio)
                
                # Get Access Speed
                # Ensure we look up the exact same variant in the access dataframe
                if variant in gef_access_df.columns and dataset in gef_access_df.index:
                     val_access = gef_access_df.loc[dataset, variant]
                else:
                     val_access = np.nan
                best_gef_access.append(val_access)
            else:
                best_gef_ratio.append(np.nan)
                best_gef_access.append(np.nan)

        best_gef_ratio = pd.Series(best_gef_ratio, index=gef_ratio_df.index)
        best_gef_access = pd.Series(best_gef_access, index=gef_ratio_df.index)
        best_gef_decomp = pd.Series(np.nan, index=gef_ratio_df.index) # Decompression not available yet

        comp_ratio_df = pd.DataFrame(competitor_benchmarks['compression_ratio'])
        comp_decomp_df = pd.DataFrame(competitor_benchmarks['decompression_speed'])
        comp_access_df = pd.DataFrame(competitor_benchmarks['random_access_speed'])

        comp_ratio_df['Best GEF'] = best_gef_ratio
        comp_decomp_df['Best GEF'] = best_gef_decomp
        comp_access_df['Best GEF'] = best_gef_access

        dataset_order = ["IT", "US", "ECG", "WD", "AP", "UK", "GE", "LON", "LAT", "DP", "CT", "DU", "BT", "BW", "BM", "BP"]
        gp_cols = ['Xz', 'Brotli', 'Zstd', 'Lz4', 'Snappy']
        sp_cols = ['Chimp128', 'Chimp', 'TSXor', 'DAC', 'Gorilla', 'LeCo', 'ALP', 'NeaTS', 'Best GEF']

        final_ratio_df = comp_ratio_df.reindex(index=dataset_order, columns=gp_cols + sp_cols)
        final_decomp_df = comp_decomp_df.reindex(index=dataset_order, columns=gp_cols + sp_cols)
        final_access_df = comp_access_df.reindex(index=dataset_order, columns=gp_cols + sp_cols)

        # Helper for rotating column headers
        rotated_headers = " & ".join([f"\\adjustbox{{angle=45,lap=\\width-1em}}{{{col}}}" for col in gp_cols + sp_cols])

        # Assemble the final LaTeX file string using table* and rotated headers with resizebox
        header = (
                "% NOTE: Use \\usepackage{adjustbox} and \\usepackage{graphicx} in your LaTeX preamble.\n"
                "\\begin{table*}[htbp]\n"
                "\\caption{Compression ratio (top), decompression speed (middle), and random access speed (bottom) achieved by general-purpose and special-purpose compressors. Best in family is bold, best overall is underlined.}\n"
                "\\label{tab:combined_benchmarks}\n"
                "\\centering\n"
                "\\resizebox{\\textwidth}{!}{%\n"
                "\\setlength{\\tabcolsep}{2pt}\n"
                f"\\begin{{tabular}}{{@{{}}l *{{{len(gp_cols) + len(sp_cols)}}}{{c}}@{{}}}}\n\\toprule\n"
                f"\\textbf{{Dataset}} & \\multicolumn{{{len(gp_cols)}}}{{c}}{{\\textbf{{General-purpose}}}} & \\multicolumn{{{len(sp_cols)}}}{{c}}{{\\textbf{{Special-purpose}}}} \\\\\n"
                f"\\cmidrule(lr){{2-{1+len(gp_cols)}}} \\cmidrule(l){{{2+len(gp_cols)}-{1+len(gp_cols)+len(sp_cols)}}}\n"
                " & " + rotated_headers + " \\\\\n\\midrule\n"
        )

        ratio_tex = generate_combined_table_section(final_ratio_df, "Compression ratio (\\%)", "min")
        decomp_tex = generate_combined_table_section(final_decomp_df, "Decompression speed (MB/s)", "max")
        access_tex = generate_combined_table_section(final_access_df, "Random access speed (MB/s)", "max")

        footer = "\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n"

        full_tex_file = header + ratio_tex + "\\midrule\n" + decomp_tex + "\\midrule\n" + access_tex + footer

        competitor_filename = os.path.join(combined_output_dir, "competitors.tex")
        with open(competitor_filename, 'w') as f:
            f.write(full_tex_file)
        print(f"-> Combined table saved to {competitor_filename}")

    print(f"\n✅ All tasks complete.")
