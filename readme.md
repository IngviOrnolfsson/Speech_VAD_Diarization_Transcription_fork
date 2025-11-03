# Conversational Speech Labeler Pipeline

**conversational_speech_labeler** is a Python package for automated processing of dyadic conversation recordings. The pipeline includes:  

- Voice activity detection (VAD)  
- Segment filtering (short segments & low-energy filtering)  
- Turn merging and backchannel detection  
- WhisperX-based transcription per segmented turn  
- Turn classification and final labeling with context-aware merging  
- Configurable via `config.yaml`  

This package is designed for reproducible analysis of conversational audio data, suitable for experiments with multiple participants and trials.

---

## Features

- **Automated VAD** using rVAD  
- **Segment post-processing** (filter short or low-energy segments)  
- **Turn detection and merging** based on flexible heuristics  
- **WhisperX transcription** applied per segmented turn for consistency  
- **Backchannel classification** using entropy-based heuristics  
- **Final labeling output** in tab-delimited `.txt` format compatible with downstream analysis  

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd conversation_vad_labeler_package
```

2. Create a Python 3.10 virtual environment and activate it:

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Install the package in editable mode:

```bash
pip install -e .
```

---

## Directory Structure

```
conversation_vad_labeler_package/
├── conversation_vad_labeler/      # Core package modules
│   ├── io_helpers.py             # Audio file search, loading helpers
│   ├── vad.py                     # Voice activity detection wrapper
│   ├── postprocess_vad.py         # Segment filtering
│   ├── merge_turns.py             # Turn merging logic
│   ├── transcription.py           # WhisperX transcription helpers
│   ├── labeling.py                # Entropy-based backchannel detection & labeling
│   └── pipeline.py                # Full pipeline orchestration
├── scripts/
│   └── run_pipeline.py            # CLI entrypoint to run pipeline
├── config.yaml                     # Pipeline configuration
├── requirements.txt                # Dependencies
├── setup.py                        # Package setup
└── README.md
```

---

## Configuration

Pipeline behavior is controlled via `config.yaml`. Example:

```yaml
paths:
  input_audio_dir: "/path/to/audio_files"
  output_root: "./outputs"
 
LANGUAGE: da                  # WhisperX language code
VAD_MIN_DURATION: 0.07        # Minimum VAD segment duration (seconds)
ENERGY_MARGIN_DB: 10.0        # RMS energy margin for filtering
GAP_THRESH: 0.2               # Merge turns if gap < threshold (seconds)
SHORT_UTT_THRESH: 0.7         # Threshold for short utterances (seconds)
MERGE_SHORT_AFTER_LONG: true  # Merge short segments after long ones
CACHE: true                   # Cache intermediate transcription outputs
ENTROPY_THRESHOLD: 1.5        # Threshold for backchannel detection
```

---

## Usage

### Run the pipeline via script

```bash
python scripts/run_pipeline.py --experiment 9 --trial 2 --language da
```

### Run via Python API

```python
from conversation_vad_labeler.pipeline import run_pipeline

outputs = run_pipeline(
    experiment=9,
    trial=2,
    cfg_path="config.yaml",
    overwrite=False,
    device="cpu",
    language="da"
)

print(outputs)
```

Output dictionary:

```python
{
    'vad_p1': 'path/to/p1_vad.txt',
    'vad_p2': 'path/to/p2_vad.txt',
    'merged_turns': 'path/to/merged_turns.txt',
    'classified': 'path/to/classified.txt',
    'final_labels': 'path/to/final_labels.txt'
}
```

---

## Notes

- **Audio file naming:** The pipeline assumes audio files follow the format:

```
EXP{number}_{Noise/None}_p{1|2}_trial{number}.wav
```

- **Transcription consistency:** Each turn is transcribed individually to ensure identical results across reruns.  
- **Debugging:** Use the debug prints in `io_helpers.find_audio_files` to check which files are being picked.


## Contact

For questions or support, contact **[Hanlu He]** at **[hahea@dtu.dk]**.

