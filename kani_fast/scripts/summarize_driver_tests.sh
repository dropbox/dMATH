#!/usr/bin/env bash
# Summarize kani-fast-driver end-to-end tests defined in tests/*.sh files.
# Counts expected-pass, expected-fail, and soundness test entries.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/summarize_driver_tests.sh [--dir PATH] [--list pass|fail|soundness|all]

Options:
  --dir PATH      Path to the test directory (default: crates/kani-fast-compiler/tests)
  --list MODE     Print test names for MODE: pass, fail, soundness, or all
  -h, --help      Show this help message
EOF
}

test_dir="crates/kani-fast-compiler/tests"
list_mode=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            test_dir="$2"
            shift 2
            ;;
        --list)
            list_mode="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "$test_dir" ]]; then
    echo "Test directory not found: $test_dir" >&2
    exit 1
fi

pass_count=0
fail_count=0
soundness_count=0
declare -a pass_names=()
declare -a fail_names=()
declare -a soundness_names=()

# Count tests from all .sh files in the test directory
for test_file in "$test_dir"/*.sh; do
    [[ -f "$test_file" ]] || continue
    while IFS= read -r line; do
        # Match run_test "name" "file" "unsound" (soundness bugs - 4th arg "soundness" optional)
        if [[ $line =~ run_test\ \"([^\"]+)\"[[:space:]]+\"([^\"]+)\"[[:space:]]+\"unsound\" ]]; then
            name="${BASH_REMATCH[1]}"
            ((soundness_count++))
            soundness_names+=("$name")
        # Match run_test "name" "file" "pass|fail"
        elif [[ $line =~ run_test\ \"([^\"]+)\"[[:space:]]+\"([^\"]+)\"[[:space:]]+\"(pass|fail)\" ]]; then
            name="${BASH_REMATCH[1]}"
            status="${BASH_REMATCH[3]}"
            if [[ $status == "pass" ]]; then
                ((pass_count++))
                pass_names+=("$name")
            else
                ((fail_count++))
                fail_names+=("$name")
            fi
        fi
    done < "$test_file"
done

total=$((pass_count + fail_count + soundness_count))

echo "Driver test summary from $test_dir/*.sh (static counts from test definitions)"
echo "  pass:      $pass_count"
echo "  fail:      $fail_count (expected-fail + expected-timeout)"
echo "  soundness: $soundness_count (tracked soundness bugs)"
echo "  total:     $total"
echo ""
echo "Note: Fast mode skips some tests. Run ./test_driver.sh for actual execution counts."

if [[ -n "$list_mode" ]]; then
    echo ""
    case "$list_mode" in
        pass)
            printf "%s\n" "${pass_names[@]}"
            ;;
        fail)
            printf "%s\n" "${fail_names[@]}"
            ;;
        soundness)
            printf "%s\n" "${soundness_names[@]}"
            ;;
        all)
            printf "%s\n" "${pass_names[@]}"
            printf "%s\n" "${fail_names[@]}"
            printf "%s\n" "${soundness_names[@]}"
            ;;
        *)
            echo "Unknown list mode: $list_mode" >&2
            exit 1
            ;;
    esac
fi
