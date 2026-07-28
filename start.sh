#!/usr/bin/env sh
set -eu

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"
export APP_MODE="${APP_MODE:-public}"
export ANKIT_APP_DATA_ROOT="${ANKIT_APP_DATA_ROOT:-deployment_data}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYVISTA_OFF_SCREEN="${PYVISTA_OFF_SCREEN:-true}"

exec python -m streamlit run tracking_app/research/shot_analysis_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-10000}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=true \
  --browser.gatherUsageStats=false
