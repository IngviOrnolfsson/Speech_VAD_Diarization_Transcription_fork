"""CLI helper for the conversation VAD labeler package."""

from __future__ import annotations

import os
import time
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables

from speech_vad_diarization import process_conversation

# Optional: CarbonTracker for energy monitoring
try:
    from carbontracker.tracker import CarbonTracker

    CARBONTRACKER_AVAILABLE = True
except ImportError:
    CARBONTRACKER_AVAILABLE = False

# ============================================
# USER CONFIGURATION
# ============================================
ENABLE_CARBON_TRACKING = True  # Set to False to disable carbon tracking


def _default_example_inputs() -> tuple[dict[str, str], str, str]:
    base = Path("examples/recordings")
    vad_type = "rvad"
    output_directory = "outputs/diad"
    return (
        {
            "P1": str(base / "EXP9_None_p1_trial2.wav"),
            "P2": str(base / "EXP9_None_p2_trial2.wav"),
        },
        vad_type,
        output_directory,
    )


def _diarize_example_inputs() -> tuple[str, str, str]:
    base = Path("examples/coral")
    vad_type = "pyannote"
    output_directory = "outputs/diarize"
    return (
        str(base / "conv_0cbf895a2078529eb4a9d8b212e710c9.wav"),
        vad_type,
        output_directory,
    )


def _triad_example_inputs() -> tuple[dict[str, str], str, str]:
    base = Path("examples/Triad")
    vad_type = "rvad"
    output_directory = "outputs/triad_rvad"
    return (
        {
            "P1": str(base / "EXP2_None_P1_T1.wav"),
            "P2": str(base / "EXP2_None_P2_T1.wav"),
            "P3": str(base / "EXP2_None_P3_T1.wav"),
        },
        vad_type,
        output_directory,
    )


def main() -> None:
    speakers_audio, vad_type, output_directory = _triad_example_inputs()

    # HuggingFace token for pyannote - can be set via:
    # 1. HF_TOKEN environment variable (or in .env file)
    # 2. huggingface-cli login (stored in ~/.cache/huggingface/token)
    # If not set, pyannote will fail with a clear error message.
    hf_token = os.environ.get("HF_TOKEN")

    # Initialize CarbonTracker if enabled and available
    tracker = None
    if ENABLE_CARBON_TRACKING and CARBONTRACKER_AVAILABLE:
        api_key = os.environ.get("ELECTRICITYMAPS_API_KEY")
        tracker_kwargs = {
            "epochs": 1,
            "monitor_epochs": 1,
            "log_dir": "logs/carbon",
            "decimal_precision": 3,
            "ignore_errors": True,
            "sim_cpu": "AMD EPYC 7302",  # Name (for logging)
            "sim_cpu_tdp": int(155 * 2 / 16),  # TDP in Watts - 2/16 cores used
            "sim_cpu_util": 0.2,  # Estimated utilization (0-1)
        }
        if api_key:
            tracker_kwargs["api_keys"] = {"electricitymaps": api_key}
        else:
            print(
                "Note: ELECTRICITYMAPS_API_KEY not set. "
                "Carbon tracking will run without CO2 intensity data."
            )
        tracker = CarbonTracker(**tracker_kwargs)
        tracker.epoch_start()
    elif ENABLE_CARBON_TRACKING and not CARBONTRACKER_AVAILABLE:
        print("CarbonTracker not installed. Run: pip install carbontracker")

    start_time = time.time()
    process_conversation(
        speakers_audio=speakers_audio,
        output_dir=output_directory,
        vad_type=vad_type,
        auth_token=hf_token,  # Pass HF token for pyannote
        energy_margin_db=20.0,
        whisper_device="auto",
        interactive_energy_filter=False,
        batch_size=30.0,  # Total seconds per batch
        skip_vad_if_exists=True,
        skip_transcription_if_exists=True,
    )
    end_time = time.time()

    # Stop CarbonTracker and output results
    if tracker is not None:
        tracker.epoch_end()
        tracker.stop()

    print(f"Processing took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
