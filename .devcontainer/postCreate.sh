#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade "pip<24.1" setuptools wheel

# Pin build-time Cython for packages like PyAV (av==10.*) that are not
# compatible with Cython 3 in build isolation.
CONSTRAINTS_FILE="$(pwd)/.devcontainer/constraints.txt"
if [[ -f "${CONSTRAINTS_FILE}" ]]; then
  export PIP_CONSTRAINT="${CONSTRAINTS_FILE}"
fi
python -m pip install "Cython<3"

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

# Seed Codex CLI config from the host when available.
CODEX_HOST_DIR="/tmp/.codex-host"
CODEX_TARGET_DIR="${HOME}/.codex"
mkdir -p "${CODEX_TARGET_DIR}"
if [[ -f "${CODEX_HOST_DIR}/auth.json" ]]; then
  install -m 600 "${CODEX_HOST_DIR}/auth.json" "${CODEX_TARGET_DIR}/auth.json"
fi
if [[ -f "${CODEX_HOST_DIR}/config.toml" ]]; then
  install -m 600 "${CODEX_HOST_DIR}/config.toml" "${CODEX_TARGET_DIR}/config.toml"
fi

# Seed SSH keys/config from the host when available.
SSH_HOST_DIR="/tmp/.ssh-host"
SSH_TARGET_DIR="${HOME}/.ssh"
mkdir -p "${SSH_TARGET_DIR}"
chmod 700 "${SSH_TARGET_DIR}"
if [[ -d "${SSH_HOST_DIR}" ]]; then
  for ssh_file in "${SSH_HOST_DIR}/"id_* "${SSH_HOST_DIR}/"known_hosts "${SSH_HOST_DIR}/"config; do
    if [[ -f "${ssh_file}" ]]; then
      install -m 600 "${ssh_file}" "${SSH_TARGET_DIR}/$(basename "${ssh_file}")"
    fi
  done
  if [[ -d "${SSH_HOST_DIR}/"config.d ]]; then
    mkdir -p "${SSH_TARGET_DIR}/config.d"
    chmod 700 "${SSH_TARGET_DIR}/config.d"
    for cfg in "${SSH_HOST_DIR}/"config.d/*; do
      if [[ -f "${cfg}" ]]; then
        install -m 600 "${cfg}" "${SSH_TARGET_DIR}/config.d/$(basename "${cfg}")"
      fi
    done
  fi
fi
