#!/usr/bin/env bash
# Run the full pipeline on the 7.2 glycolysis file.
# Place data/7.2_glycolysis.pdf or data/7.2_glycolysis.txt in the repo first.
# If your PDF was downloaded into src/ with spaces (OpenStax), this script will also detect it.

set -e
cd "$(dirname "$0")/.."

if [[ -f data/7.2_glycolysis.pdf ]]; then
  python src/run_all.py data/7.2_glycolysis.pdf
elif [[ -f data/7.2_glycolysis.txt ]]; then
  python src/run_all.py data/7.2_glycolysis.txt
elif [[ -f "src/7.2 Glycolysis - Biology 2e _ OpenStax.pdf" ]]; then
  python src/run_all.py "src/7.2 Glycolysis - Biology 2e _ OpenStax.pdf"
else
  echo "No 7.2 glycolysis file found."
  echo "Add one of:"
  echo "  - data/7.2_glycolysis.pdf"
  echo "  - data/7.2_glycolysis.txt"
  echo "  - src/7.2 Glycolysis - Biology 2e _ OpenStax.pdf"
  exit 1
fi
