#!/usr/bin/env sh
# Remove macOS AppleDouble files (._*) that break pip/importlib on exFAT/FAT external drives.
# Run from repo root: sh scripts/clean_appledouble.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
find "$ROOT" -name '._*' -delete
echo "Removed AppleDouble ._ files under $ROOT"
