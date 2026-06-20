#!/usr/bin/env bash
# Build the Binary Fuse filter (C/CFFI) and install the DMCFE library.
set -e
echo "Building pyfilters (Binary Fuse filter)..."
cd modules/pyfilters && python pyfilters/ffibuild.py && cp _xorfilter*.so pyfilters/ && cd ../..
echo "Installing mife (DMCFE backend)..."
pip install ./modules/mife
echo "Done. Both vendored modules are ready."
