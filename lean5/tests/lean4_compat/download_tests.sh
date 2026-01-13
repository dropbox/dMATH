#!/bin/bash
# Download Lean 4 test files for compatibility testing
# Run from tests/lean4_compat/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p lean4_tests

echo "Fetching file list from Lean 4 repo..."

# Get list of .lean files from tests/lean directory
curl -sL "https://api.github.com/repos/leanprover/lean4/contents/tests/lean?ref=master" \
    -H "Accept: application/vnd.github.v3+json" | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data:
    if item['name'].endswith('.lean') and item['type'] == 'file':
        print(item['download_url'])
" > /tmp/lean4_test_urls.txt

TOTAL=$(wc -l < /tmp/lean4_test_urls.txt)
echo "Found $TOTAL test files"

# Download files (limit to first 100 for initial testing)
COUNT=0
MAX=100

while IFS= read -r url && [ $COUNT -lt $MAX ]; do
    FILENAME=$(basename "$url")
    if [ ! -f "lean4_tests/$FILENAME" ]; then
        echo -ne "\rDownloading [$COUNT/$MAX] $FILENAME"
        curl -sL "$url" -o "lean4_tests/$FILENAME"
    fi
    COUNT=$((COUNT + 1))
done < /tmp/lean4_test_urls.txt

echo ""
echo "Downloaded $COUNT test files to lean4_tests/"
ls lean4_tests/*.lean | wc -l | xargs -I{} echo "{} files ready for testing"
