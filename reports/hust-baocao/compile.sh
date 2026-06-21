#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ENGINE="${1:-xelatex}"
$ENGINE -interaction=nonstopmode main.tex
$ENGINE -interaction=nonstopmode main.tex
echo "Done: $(pwd)/main.pdf"
