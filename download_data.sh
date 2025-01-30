#!/bin/bash

URL="https://zenodo.org/api/records/14680012/files-archive"
OUTPUT_DIR="./data/subtask-2"
OUTPUT_FILE="$OUTPUT_DIR/full.zip"

mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_FILE"
unzip "$OUTPUT_FILE" -d "$OUTPUT_DIR"
rm "$OUTPUT_FILE"


URL="https://zenodo.org/api/records/14699130/files-archive"
OUTPUT_DIR="./data/subtask-1"
OUTPUT_FILE="$OUTPUT_DIR/full.zip"

mkdir -p "$OUTPUT_DIR"
curl -L "$URL" -o "$OUTPUT_FILE"
unzip "$OUTPUT_FILE" -d "$OUTPUT_DIR"
rm "$OUTPUT_FILE"

gunzip ./data/subtask-1/*.gz