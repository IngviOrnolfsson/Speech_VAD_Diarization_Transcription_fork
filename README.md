# Speech VAD, Diarization & Transcription Pipeline

End-to-end processing of conversation recordings with support for both pre-separated and mixed audio. Produces cleaned, transcribed, and labeled segments ready for ELAN or manual inspection.

![Pipeline Flowchart](docs/figures/Protocol.png)
*Pipeline architecture: splits at Stage 1 (VAD vs Diarization) based on input type, then converges for unified processing.*

## Features

- **Multiple VAD methods**: rVAD (energy-based) or Pyannote.audio (neural)
- **Speaker separation**: SpeechBrain SepFormer for mixed audio
- **Diarization**: Pyannote.audio for single-channel recordings
- **Transcription**: Whisper with GPU acceleration and batching
- **Smart processing**: Turn merging, entropy-based labeling, context-aware annotation

---

## Installation

### Quick Install with Make

```bash
# RECOMMENDED: Hybrid approach (Conda for FFmpeg + UV for packages)
# Works on HPC clusters without sudo, fastest Python package installation
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

make install-hybrid  # or just: make install
conda activate wp1
```

**Alternative methods:**

```bash
# LOCAL MACHINE (with sudo): Pure UV
sudo apt install -y ffmpeg  # Ubuntu/Debian
make install-uv
source .venv/bin/activate

# ALL FROM CONDA: Slower but fully self-contained
make install-conda
conda activate wp1
```

### Manual Installation

#### Option 1: Hybrid - Conda (FFmpeg) + UV (Packages) ⭐ RECOMMENDED

**Best of both worlds:** Conda for system deps, UV for fast Python packages

```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create minimal Conda environment with FFmpeg
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

# Create environment (auto-detects mamba if available)
conda env create -f environment-minimal.yml -n wp1  # or: mamba env create -f environment-minimal.yml -n wp1
conda activate wp1

# Install Python packages with UV (fast!)
uv pip install -r requirements-lock-uv.txt
pip install -e .
```

**Advantages:**
- ✅ No sudo required (FFmpeg from Conda)
- ✅ Fast package installation (UV is 10-100x faster than pip)
- ✅ Reproducible (exact versions from lockfile)
- ✅ Perfect for HPC clusters

#### Option 2: Pure UV (Local Machine with FFmpeg)

**Requirements:** System FFmpeg (requires sudo or admin access)

```bash
# Install system dependencies first (FFmpeg required for audio processing)
# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# macOS:
brew install ffmpeg

# Then install UV and Python packages
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

# Create environment with Python 3.10+ (UV will download it if needed)
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements-lock-uv.txt
pip install -e .
```

#### Option 3: Pure Conda/Mamba (Fully Self-Contained)

**Includes everything** - No UV needed, but slower

```bash
git clone https://github.com/haraldsr/Speech_VAD_Diarization_Transcription.git
cd Speech_VAD_Diarization_Transcription

# Create environment
conda env create -f environment.yml
conda activate wp1

pip install -e .
```

**Note:** Use `mamba` instead of `conda` for faster installation: `mamba env create -f environment.yml`

---

## Quick Start

### 1. Pre-separated Audio (Two Channels)

```python
from speech_vad_diarization import process_conversation

results = process_conversation(
    speakers_audio={
        "P1": "path/to/speaker1.wav",
        "P2": "path/to/speaker2.wav"
    },
    output_dir="outputs/my_experiment",
    vad_type="rvad",  # or "pyannote"
    whisper_language="da",
    batch_size=60.0
)
```

### 2. Single Mixed Audio (Diarization)

```python
results = process_conversation(
    speakers_audio="path/to/mixed_audio.wav",  # Single file
    output_dir="outputs/diarization",
    vad_type="pyannote",  # Required for diarization
    whisper_language="da"
)
```

### 3. Speaker Separation + Pipeline

```python
# First separate speakers, then process
# See speech_separation_chunked.py and run_separation_and_pipeline.py
from speech_separation_chunked import separator, separate_audio_with_smart_chunking
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix")
separated = separate_audio_with_smart_chunking(model, "mixed.wav")
# Then use process_conversation() with the separated audio paths
```

### 4. Advanced: Resume from Existing VAD

```python
# If you already have VAD files, skip VAD and start from transcription
# See run_pipeline_only.py for complete example
from run_pipeline_only import run_pipeline_with_existing_vad
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad_type` | `"rvad"` | VAD method: `"rvad"` or `"pyannote"` |
| `vad_min_duration` | `0.07` | Minimum segment duration (seconds) |
| `energy_margin_db` | `10.0` | Energy threshold for filtering |
| `gap_thresh` | `0.2` | Max gap for merging segments |
| `whisper_model_name` | `"large-v3"` | Whisper model (or custom like `"CoRal-project/roest-whisper-large-v1"`) |
| `whisper_language` | `"en"` | Target language code |
| `whisper_device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `batch_size` | `60.0` | Batch size in seconds |
| `export_elan` | `True` | Export ELAN-compatible tab-delimited file |

---

## Output Files

![Output Structure](docs/figures/Protocol-2.png)
*Each speaker track consists of discrete, timestamped speech intervals (Turns or Backchannels).*

```
outputs/
└── experiment_name/
    ├── P1/                            # Speaker-specific folder
    │   └── speaker1_vad.txt           # VAD timestamps
    ├── P2/
    │   └── speaker2_vad.txt
    ├── merged_turns.txt               # Merged conversation turns
    ├── raw_transcriptions.txt         # Raw Whisper output
    ├── classified_transcriptions.txt  # With entropy labels
    ├── final_labels.txt               # Context-merged annotations (TSV)
    └── final_labels_elan.txt          # ELAN-compatible format
```

**Pipeline output format (`final_labels.txt`):**
```
speaker	start_sec	end_sec	transcription	entropy	type
P1	0.50	2.30	Hello there	2.31	turn
P2	2.45	3.10	Mm-hmm	0.00	backchannel
```

**ELAN import format (`final_labels_elan.txt`):**
```
tier	begin	end	annotation
P1_turn	500	2300	Hello there
P2_backchannel	2450	3100	Mm-hmm
```

To import in ELAN: **File → Import → Tab-delimited Text...** (skip first line: Yes)

---

## Speaker Separation (Mixed Audio)

For audio with overlapping speakers:

```python
from speech_separation_chunked import (
    separator,
    separate_audio_with_smart_chunking,
    save_vad_timestamps
)

# 1. Load SepFormer model
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir='pretrained_models/sepformer-wsj02mix',
    run_opts={"device": "cuda"}
)

# 2. Separate speakers
separated = separate_audio_with_smart_chunking(
    model=model,
    audio_path='mixed_audio.wav',
    chunk_duration_sec=60
)

# 3. Extract VAD and audio segments
vad_paths = save_vad_timestamps(separated, output_dir='outputs/vad')
```

**Parameters:**
- `chunk_duration_sec`: Reduce to 30 or 20 if OOM errors
- Smart chunking respects speech boundaries
- See `speech_separation.ipynb` for interactive workflow

---

## Troubleshooting

### FFmpeg Not Found / Torchcodec Error
```bash
# The error "Could not load libtorchcodec" means FFmpeg is missing
# Install FFmpeg at system level:

# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# macOS:
brew install ffmpeg

# Verify installation:
ffmpeg -version

# Then reinstall torchcodec:
pip install --force-reinstall torchcodec==0.8.1
```

**Note:** Conda environments include FFmpeg automatically. UV/pip environments require manual FFmpeg installation.

### Out of Memory
```python
# Reduce batch sizes
batch_size=30.0,
whisper_transformers_batch_size=50
```

### GPU Not Detected
```python
# Check PyTorch CUDA
import torch
print(torch.cuda.is_available())

# Force CPU if needed
whisper_device="cpu"
```

### Package Version Conflicts
```bash
# Use UV for better dependency resolution
uv pip install -r requirements-lock-uv.txt --upgrade
```

---

## Repository Structure

```
.
├── Makefile                          # Build automation (install, lint, clean)
├── environment-minimal.yml           # Minimal Conda env (Python + FFmpeg only)
├── environment.yml                   # Full Conda environment
├── requirements.txt                  # Flexible dependencies (development)
├── requirements-lock-uv.txt          # Exact dependencies (reproducibility)
├── conversation_pipeline.py          # Example CLI usage
├── run_pipeline_only.py              # Advanced: Resume from existing VAD files
├── run_separation_and_pipeline.py    # Full workflow with separation
├── speech_separation_chunked.py      # Speaker separation utilities
└── src/                              # Package source (installed as speech_vad_diarization)
    ├── __init__.py                   # Exports process_conversation, load_whisper_model, transcribe_segments
    ├── conversation.py               # Main API: process_conversation()
    ├── vad.py                        # VAD wrappers (rVAD, Pyannote)
    ├── postprocess_vad.py            # Energy filtering, segment cleaning
    ├── merge_turns.py                # Turn merging logic
    ├── transcription.py              # Whisper transcription
    └── labeling.py                   # Entropy-based labeling
```

---

## Development

### Make Commands

```bash
make help          # Show all available commands
make install       # Install with hybrid approach (recommended)
make install-uv    # Install with UV only (requires system FFmpeg)
make install-conda # Install with Conda/Mamba
make lint          # Run linting (flake8, mypy, isort, black)
make format        # Format code (isort + black)
make clean         # Remove build artifacts
```

### Git Hooks (Automatic Linting)

The repository includes a pre-commit hook that automatically runs `make format` and `make lint` before each commit:

```bash
# Install git hooks (done automatically, but can be re-run)
./scripts/install_hooks.sh
```

When you commit in VS Code (or via command line):
1. Code is automatically formatted
2. Linting checks run
3. Commit is blocked if linting fails

### View Installed Packages
```bash
pip list
conda list  # If using conda
```

### Update Environment to Match Working Setup
```bash
# Using UV (recommended)
conda activate wp1
pip list --format=freeze | sed 's/grpcio==1.74.1/grpcio>=1.74.0/; s/matplotlib==3.10.8/matplotlib>=3.10.0/' > requirements-lock-uv.txt

# Then apply to another environment
conda activate other_env
uv pip install -r requirements-lock-uv.txt --upgrade
```

---

## Files

- **`environment-minimal.yml`**: Minimal Conda environment (Python 3.10 + FFmpeg only)
- **`environment.yml`**: Full Conda environment (all dependencies from Conda)
- **`requirements.txt`**: Flexible versions (`>=`) for development
- **`requirements-lock-uv.txt`**: Exact versions for reproducibility (UV-compatible)

**Recommended workflow:** Use `environment-minimal.yml` + `requirements-lock-uv.txt` for best performance.

---

## Credits

- **Pyannote.audio**: https://github.com/pyannote/pyannote-audio
- **SpeechBrain**: https://speechbrain.github.io/
- **Whisper**: https://github.com/openai/whisper
- **rVAD**: https://github.com/zhenghuatan/rVADfast
- **UV**: https://github.com/astral-sh/uv

---

## License

**TODO: Temporary** - All Rights Reserved - Copyright (c) 2025 Harald Skat-Rørdam, Hanlu He

No license is currently granted for use, modification, or distribution of this software. An open-source license will be applied once determined by the copyright holders. See [LICENSE](LICENSE) file for details.
