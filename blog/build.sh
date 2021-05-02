#!/bin/bash

OUT_DIR="."

mkdir -p "$OUT_DIR"

for md in *.md; do
    out_file="$OUT_DIR/${md%.md}.html"

    pandoc "$md" \
        -o "$out_file" \
        --standalone \
        --mathjax
done

