#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PATH="${TRAJECTORY_VENV:-/workspace/venvs/trajectory-prediction}"
python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$VENV_PATH/bin/python" -m pip install -r requirements-cpu.lock
"$VENV_PATH/bin/python" -m pip install --no-deps --no-build-isolation -e .
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/trajectory-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
"$VENV_PATH/bin/python" -m pip check
"$VENV_PATH/bin/python" -m vehtraj download
"$VENV_PATH/bin/python" -m pytest -q
"$VENV_PATH/bin/python" -m vehtraj smoke --device cpu
