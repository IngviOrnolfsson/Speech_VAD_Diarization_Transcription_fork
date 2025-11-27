"""Speech VAD, Diarization & Transcription Pipeline."""

__all__ = [
    "__version__",
    "process_conversation",
    "load_whisper_model",
    "transcribe_segments",
]

__version__ = "0.1.0"

from .conversation import process_conversation
from .transcription import load_whisper_model, transcribe_segments
