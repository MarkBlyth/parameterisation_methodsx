#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 file1.csv [file2.csv ...]" >&2
  exit 1
fi

cat "$1"
shift
for csv in "$@"; do
  tail -n +2 "$csv"
done
