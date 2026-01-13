#!/bin/bash
# Download VNN-COMP 2021-2024 benchmarks
#
# Usage: ./download_benchmarks.sh [years...]
# Examples:
#   ./download_benchmarks.sh           # Download all years
#   ./download_benchmarks.sh 2021      # Download only 2021
#   ./download_benchmarks.sh 2023 2024 # Download 2023 and 2024

set -e

cd "$(dirname "$0")"

download_year() {
    local year=$1
    local dir="vnncomp${year}"

    if [ -d "$dir" ]; then
        echo "[$year] Already exists: $dir"
        return
    fi

    echo "[$year] Downloading..."

    case $year in
        2021)
            git clone --depth 1 https://github.com/stanleybak/vnncomp2021.git "$dir"
            ;;
        2022)
            git clone --depth 1 https://github.com/stanleybak/vnncomp2022.git "$dir"
            ;;
        2023)
            git clone --depth 1 https://github.com/ChristopherBrix/vnncomp2023_benchmarks.git "$dir"
            ;;
        2024)
            git clone --depth 1 https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git "$dir"
            ;;
        2025)
            git clone --depth 1 https://github.com/VNN-COMP/vnncomp2025_benchmarks.git "$dir"
            ;;
        *)
            echo "Unknown year: $year"
            return 1
            ;;
    esac

    # Decompress gzipped files
    echo "[$year] Decompressing files..."
    find "$dir" -name "*.gz" -exec gunzip -k {} \; 2>/dev/null || true

    echo "[$year] Done"
}

# Default: download all years
YEARS="${@:-2021 2023 2024 2025}"

echo "=== VNN-COMP Benchmark Downloader ==="
echo "Years: $YEARS"
echo ""

for year in $YEARS; do
    download_year "$year"
done

echo ""
echo "=== Summary ==="
for dir in vnncomp*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.onnx" | wc -l | tr -d ' ')
        echo "$dir: $count ONNX files"
    fi
done

echo ""
echo "Run tests with: pytest -v --timeout=10"
