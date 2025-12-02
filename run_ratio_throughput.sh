#!/bin/bash

set -u
set -o pipefail
# removed 'set -e' globally so we can trap errors manually

# ... (Configuration unchanged) ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/compression_metrics_results"
BENCH_NO_OMP="${REPO_ROOT}/build/benchmarks/compression_benchmark_no_omp"

# Check for python3 availability
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found." >&2
    exit 1
fi

abs_path() {
    python3 - "$1" <<'PY'
import os, sys
if len(sys.argv) > 1:
    print(os.path.abspath(sys.argv[1]))
else:
    sys.exit(1)
PY
}

usage() {
    cat <<EOF
Usage:
  $(basename "$0") <input_directory> [output_directory] [--filter REGEX] [--partition SIZE]

Options:
  --filter REGEX     Benchmark name filter (default: _Compression)
  --partition SIZE   Run only benchmarks with specific partition size (e.g., 32000)
                     If not specified, all partition sizes will be benchmarked.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

INPUT_DIR_RAW="$1"
shift

if [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; then
    OUTPUT_DIR_RAW="$1"
    shift
else
    OUTPUT_DIR_RAW="${DEFAULT_OUTPUT_DIR}"
fi

BENCH_FILTER="_Compression"
PARTITION_SIZE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter) BENCH_FILTER="$2"; shift 2 ;;
        --partition) PARTITION_SIZE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown option '$1'" >&2; usage; exit 1 ;;
    esac
done

INPUT_DIR="$(abs_path "${INPUT_DIR_RAW}")"
OUTPUT_DIR="$(abs_path "${OUTPUT_DIR_RAW}")"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

if [[ ! -x "$BENCH_NO_OMP" ]]; then
    echo "Error: Benchmark executable '$BENCH_NO_OMP' not found or not executable."
    exit 1
fi

# Find files
echo "Scanning for .bin files in $INPUT_DIR..."
BIN_FILES=()
while IFS= read -r bin_path; do
    [[ -z "$bin_path" ]] && continue
    BIN_FILES+=("$bin_path")
done < <(python3 - "$INPUT_DIR" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
if root.is_dir():
    files = sorted(p.resolve() for p in root.glob("*.bin"))
    print("\n".join(str(p) for p in files))
PY
)

if (( ${#BIN_FILES[@]} == 0 )); then
    echo "Error: No .bin files found in '$INPUT_DIR'."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================================================="
echo "Single-thread Compression Metrics"
echo "========================================================================="
echo "Datasets found  : ${#BIN_FILES[@]}"
echo "Benchmark exec  : $BENCH_NO_OMP"
echo "Benchmark filter: $BENCH_FILTER"
if [[ -n "$PARTITION_SIZE" ]]; then
    echo "Partition size  : $PARTITION_SIZE (will filter results after benchmarking)"
else
    echo "Partition size  : ALL"
fi
echo "========================================================================="
echo ""

declare -a SUMMARY_ARGS=()
dataset_index=0

for dataset_path in "${BIN_FILES[@]}"; do
    ((dataset_index++))
    dataset_basename="$(basename "$dataset_path")"
    dataset_name="${dataset_basename%.bin}"
    
    # Use temporary file if partition filtering is needed
    if [[ -n "$PARTITION_SIZE" ]]; then
        output_path_temp="${OUTPUT_DIR}/${dataset_name}_no_omp_temp.json"
        output_path="${OUTPUT_DIR}/${dataset_name}_no_omp.json"
    else
        output_path_temp="${OUTPUT_DIR}/${dataset_name}_no_omp.json"
        output_path="${output_path_temp}"
    fi

    echo "[$dataset_index/${#BIN_FILES[@]}] Processing '${dataset_basename}'..."

    # 1. Print the command so you can copy-paste it to debug if it fails
    CMD=(
        "$BENCH_NO_OMP" "$dataset_path"
        --benchmark_filter="${BENCH_FILTER}"
        --benchmark_format=json
        --benchmark_out="${output_path_temp}"
        --benchmark_out_format=json
        --benchmark_context=openmp=disabled
        --benchmark_context=variant=no_omp
        --benchmark_context=bitvector=pasta
        --benchmark_context=dataset="${dataset_name}"
        --benchmark_context=threads=1
    )
    
    # echo "DEBUG: Running: ${CMD[*]}"

    # 2. Run command and capture exit code
    "${CMD[@]}"
    EXIT_CODE=$?

    # 3. Check for failure
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Benchmark failed for dataset '$dataset_basename'"
        echo "Exit Code: $EXIT_CODE"
        echo "Command attempted:"
        echo "${CMD[*]}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi

    # 4. Filter by partition size if specified
    if [[ -n "$PARTITION_SIZE" ]]; then
        python3 - "$output_path_temp" "$output_path" "$PARTITION_SIZE" <<'PY'
import json
import sys
from pathlib import Path

if len(sys.argv) != 4:
    sys.exit(1)

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
partition_size = sys.argv[3]

data = json.loads(input_file.read_text())
filtered_benchmarks = []

for bench in data.get("benchmarks", []):
    label = bench.get("label", "")
    # Label format: basename/factory/strategy/partition_size
    # Check if label ends with the partition size
    if label.endswith("/" + partition_size):
        filtered_benchmarks.append(bench)

data["benchmarks"] = filtered_benchmarks
output_file.write_text(json.dumps(data, indent=2))

# Clean up temp file
input_file.unlink()

print(f"Filtered: kept {len(filtered_benchmarks)} benchmarks for partition size {partition_size}")
PY
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to filter JSON by partition size"
            exit 1
        fi
    fi

    echo "     Saved JSON: $(basename "$output_path")"
    echo ""

    SUMMARY_ARGS+=("$dataset_name" "$output_path")
done

# Run summary script
python3 - "${SUMMARY_ARGS[@]}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

ARGS = sys.argv[1:]
if not ARGS: sys.exit(0)

def parse_entry(label: str, bench_name: str):
    if not label: return None
    name_parts = bench_name.split('/')
    if len(name_parts) < 2: return None
    compressor = name_parts[1].replace("_Compression", "")
    label_parts = label.split('/')
    
    if compressor == "RLE_GEF":
        if len(label_parts) < 3: return None
        strategy = None
        partition = label_parts[2]
    else:
        if len(label_parts) < 4: return None
        strategy = label_parts[2].upper()
        partition = label_parts[3]
        
    try:
        partition = int(partition)
    except ValueError:
        return None
    return compressor, strategy, partition

def strategy_display(strategy):
    if strategy == "APPROXIMATE": return "approximate"
    if strategy == "BRUTE_FORCE": return "optimal"
    return str(strategy).lower() if strategy else None

stats = defaultdict(lambda: {"ratio_sum": 0.0, "ratio_count": 0, "thru_sum": 0.0, "thru_count": 0})

for idx in range(0, len(ARGS), 2):
    dataset = ARGS[idx]
    json_path = Path(ARGS[idx + 1])
    if not json_path.is_file(): continue
    try:
        payload = json.loads(json_path.read_text())
    except: continue

    for bench in payload.get("benchmarks", []):
        if "_Compression" not in bench.get("name", ""): continue
        parsed = parse_entry(bench.get("label", ""), bench.get("name", ""))
        if not parsed: continue
        
        compressor, strategy, partition = parsed
        key = (compressor, strategy, partition)
        
        bpi = bench.get("bpi")
        thru = bench.get("compression_throughput_MBs")

        if isinstance(bpi, (int, float)):
            stats[key]["ratio_sum"] += (float(bpi) / 64.0) * 100.0
            stats[key]["ratio_count"] += 1
        if isinstance(thru, (int, float)):
            stats[key]["thru_sum"] += float(thru)
            stats[key]["thru_count"] += 1

if not stats:
    print("No data found matching filter.")
    sys.exit(0)

print("=========================================================================")
print("Compressor averages by strategy & partition")
print("=========================================================================")

for (compressor, strategy, partition), values in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1] or "", x[0][2])):
    r_avg = (values["ratio_sum"] / values["ratio_count"]) if values["ratio_count"] else None
    t_avg = (values["thru_sum"] / values["thru_count"]) if values["thru_count"] else None
    
    r_str = f"{r_avg:.2f} %" if r_avg else "n/a"
    t_str = f"{t_avg:,.2f} MB/s" if t_avg else "n/a"
    s_str = strategy_display(strategy)
    lbl = f"{compressor} ({s_str})" if s_str else compressor
    
    print(f"- {lbl}, partition {partition}: ratio={r_str}, throughput={t_str}")
print("=========================================================================")
PY