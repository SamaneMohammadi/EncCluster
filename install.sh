#!/usr/bin/env bash
set -e
# Build the Binary Fuse filter (C/CFFI) and install the DMCFE backend.
echo "Building pyfilters (Binary Fuse filter)..."
( cd modules/pyfilters && python pyfilters/ffibuild.py && cp _xorfilter*.so pyfilters/ )
echo "Installing mife (DMCFE backend)..."
pip install ./modules/mife
echo "Done."
