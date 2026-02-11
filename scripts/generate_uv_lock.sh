#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/generate_uv_lock.sh [output-file]
# Default output file: requirements-lock-uv.txt

OUT=${1:-requirements-lock-uv.txt}

# Ensure we run pip from the currently active environment
PIP_CMD=${PIP_CMD:-pip}

# List installed packages and exclude local editable packages
# Filters out packages installed with `pip install -e .`
$PIP_CMD list --format=freeze | grep -v "^speech-vad-diarization" | grep -v "^speech_vad_diarization_transcription" > "$OUT.tmp"

# Replace nemo-toolkit rc version with git URL (rc versions don't exist on PyPI)
# This ensures the lockfile can be installed without errors
sed -i 's|^nemo-toolkit==.*$|nemo-toolkit @ git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]|g' "$OUT.tmp"

mv "$OUT.tmp" "$OUT"

# Inform the user
printf "Generated lock file: %s (%d packages)\n" "$OUT" "$(wc -l < "$OUT")"
printf "Note: nemo-toolkit rc version replaced with git URL\n"

# Note: If you prefer uv to resolve dependencies from `pyproject.toml`,
# use `uv lock` (requires a valid [project] table). The generated ~requirements
# file above captures the installed environment and may be used with `uv pip install -r`.
