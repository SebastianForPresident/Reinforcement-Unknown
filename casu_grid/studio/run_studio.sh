#!/usr/bin/env bash
# Start Research Studio from this checkout (uses the checkout's code and revision for every run it launches).
cd "$(dirname "$0")/../.." || exit 1
exec python3 -m casu_grid.studio "$@"
