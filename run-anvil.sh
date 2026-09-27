#!/bin/bash
# Anvil launcher — for service managers or manual runs.
cd "$(dirname "$0")"

[ -d venv ] || python3 -m venv venv
source venv/bin/activate

# Ensure deps (quiet; only acts when missing)
python3 -c "import flask, httpx" 2>/dev/null || pip install -q flask httpx

export ANVIL_PORT="${ANVIL_PORT:-8590}"
exec python3 app.py
