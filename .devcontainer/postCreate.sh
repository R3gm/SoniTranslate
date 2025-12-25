#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

# Base dependencies for most local development tasks.
if [[ -f requirements_base.txt ]]; then
  python -m pip install -r requirements_base.txt
fi

if [[ -f requirements_extra.txt ]]; then
  python -m pip install -r requirements_extra.txt
fi

# Optional: XTTS extras (set INSTALL_XTTS=1 in devcontainer.json or your env).
if [[ "${INSTALL_XTTS:-}" == "1" ]] && [[ -f requirements_xtts.txt ]]; then
  python -m pip install -r requirements_xtts.txt
  python -m pip install TTS==0.21.1 --no-deps
fi

# Optional: full CUDA requirements (set INSTALL_CUDA_REQS=1).
if [[ "${INSTALL_CUDA_REQS:-}" == "1" ]] && [[ -f requirements.txt ]]; then
  python -m pip install -r requirements.txt
fi
