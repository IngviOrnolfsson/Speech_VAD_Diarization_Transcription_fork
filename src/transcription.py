"""
Based on Conversational_speech_labeling_pipeline by Hanlu He.

https://github.com/hanlululu/Conversational_speech_labeling_pipeline
"""

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.base import Pipeline

# Set PyTorch CUDA memory configuration for better fragmentation handling
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class TransformersASRModel:
    pipeline: Pipeline
    language: Optional[str]


def load_whisper_model(
    transcription_model_name: str = "openai/whisper-large-v3",
    device: str = "cpu",
    language: Optional[str] = "da",
    cache_dir: Optional[str] = None,
    transformers_batch_size: int = 100,
) -> TransformersASRModel:
    """Initialise and return a Whisper ASR model via the Transformers pipeline.

    Parameters
    ----------
    transcription_model_name
        Model identifier (e.g., 'openai/whisper-large-v3')
    device
        'cpu' or 'cuda' for GPU inference
    language
        Target language code (e.g., 'da' for Danish)
    cache_dir
        Optional directory for model caching
    transformers_batch_size
        Maximum number of clips the transformers pipeline should batch internally.

    Returns
    -------
    TransformersASRModel
        Wrapper containing the configured transformers pipeline
    """

    # Convert device string to torch.device
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif device == "cuda":
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        torch_device = torch.device("cpu")
        torch_dtype = torch.float32

    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transcription_model_name,
        dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    model.to(torch_device)

    processor = AutoProcessor.from_pretrained(
        transcription_model_name, cache_dir=cache_dir
    )

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=transformers_batch_size,
        dtype=torch_dtype,
        device=torch_device,
        chunk_length_s=30.0,
    )

    return TransformersASRModel(
        pipeline=pipe,
        language=language,
    )


def _save_segment_wav(
    out_path: str, audio_array: np.ndarray, sr: int = 16000, compress: bool = True
) -> None:
    """Persist a speech segment to disk as 16-bit PCM WAV."""

    if compress:
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        audio_array = np.clip(audio_array, -1.0, 1.0)
        sf.write(out_path, audio_array, samplerate=sr, subtype="PCM_16")
    else:
        sf.write(out_path, audio_array, samplerate=sr)


def _transcribe_single_with_retry(
    seg_file: str,
    txt_cache: str,
    model: TransformersASRModel,
    generate_kwargs: Dict[str, Any],
    max_retries: int = 3,
) -> str:
    """Transcribe a single file with OOM retry logic.

    Parameters
    ----------
    seg_file : str
        Path to the audio segment file
    txt_cache : str
        Path to cache the transcription
    model : TransformersASRModel
        The ASR model
    generate_kwargs : Dict[str, Any]
        Generation parameters
    max_retries : int
        Maximum number of retry attempts with increasing memory clearing

    Returns
    -------
    str
        Transcribed text
    """
    pipe = model.pipeline
    original_device = pipe.model.device
    original_dtype = pipe.model.dtype

    for attempt in range(max_retries):
        try:
            # Aggressive memory and cache clearing before attempt
            if torch.cuda.is_available():
                # Force clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()

            # CRITICAL: Clear model's past_key_values cache
            if hasattr(pipe.model, "model") and hasattr(pipe.model.model, "decoder"):
                # Force recreation of decoder cache to avoid accumulation
                pipe.model.model.decoder.past_key_values = None

            # Clear Python garbage
            gc.collect()

            result = pipe(
                seg_file,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
            text = result.get("text", "").strip()

            # Save to cache
            with open(txt_cache, "w", encoding="utf-8") as cache_file:
                cache_file.write(text)

            # Clear memory after success
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return str(text)

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(
                    f"\n⚠️  OOM on file {os.path.basename(seg_file)} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                gc.collect()

                if attempt == max_retries - 1:
                    # Last GPU attempt failed - try CPU as final fallback
                    print(
                        "    → All GPU attempts failed. " "Trying CPU as last resort..."
                    )
                    try:
                        # Move model to CPU and convert to float32 for CPU
                        cpu_dev = "cpu"
                        pipe.model.to(cpu_dev, dtype=torch.float32)  # type: ignore
                        pipe.dtype = torch.float32  # type: ignore[misc]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                        result = pipe(
                            seg_file,
                            generate_kwargs=generate_kwargs,
                            return_timestamps=True,
                        )
                        text = result.get("text", "").strip()

                        # Save to cache
                        with open(txt_cache, "w", encoding="utf-8") as cache_file:
                            cache_file.write(text)

                        print("    ✓ Successfully transcribed on CPU")

                        # Move model back to original device and dtype
                        dev, dt = original_device, original_dtype
                        pipe.model.to(dev, dtype=dt)  # type: ignore
                        pipe.dtype = original_dtype  # type: ignore[misc]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                        return str(text)

                    except Exception as cpu_error:
                        print(f"    ❌ CPU fallback also failed: {cpu_error}")
                        # Move model back to original device and dtype
                        try:
                            dev, dt = original_device, original_dtype
                            pipe.model.to(dev, dtype=dt)  # type: ignore
                            pipe.dtype = original_dtype  # type: ignore[misc]
                        except Exception as restore_error:
                            print(
                                f"    ⚠️  Could not restore model to GPU: "
                                f"{restore_error}"
                            )
                        # Save error marker to cache so we know it failed
                        error_text = f"[TRANSCRIPTION_FAILED: {cpu_error}]"
                        with open(txt_cache, "w", encoding="utf-8") as cache_file:
                            cache_file.write(error_text)
                        return error_text
                else:
                    print("    → Retrying with aggressive memory cleanup...")
                    import time

                    time.sleep(2)  # Brief pause to let GPU stabilize
            else:
                raise

    return ""


def _transcribe_batch(
    batch_files: List[str],
    batch_caches: List[str],
    model: TransformersASRModel,
    cache: bool = False,
) -> List[str]:
    """Transcribe a batch of segment files using pipeline batching.

    Returns list of transcribed texts (in same order as input).
    """
    # Check which files need transcription (not cached)
    files_to_transcribe = []
    file_indices = []
    results = [""] * len(batch_files)

    for i, (seg_file, txt_cache) in enumerate(zip(batch_files, batch_caches)):
        if cache and os.path.exists(txt_cache):
            # Load from cache
            with open(txt_cache, "r", encoding="utf-8") as cache_file:
                results[i] = cache_file.read().strip()
        else:
            # Needs transcription
            files_to_transcribe.append(seg_file)
            file_indices.append(i)

    # If no files need transcription, return cached results
    if not files_to_transcribe:
        return results

    pipe = model.pipeline
    language = model.language

    # Collect metadata for logging without keeping large arrays in memory
    durations = []
    for seg_file in files_to_transcribe:
        try:
            info = sf.info(seg_file)
            durations.append(float(info.duration))
        except RuntimeError:
            durations.append(0.0)

    total_duration = sum(durations)

    generate_kwargs: Dict[str, Any] = {
        "task": "transcribe"
    }  # , "return_timestamps": True
    if language:
        generate_kwargs["language"] = language

    # Try to transcribe batch, if OOM or batching error then split and retry
    try:
        batch_results = pipe(
            files_to_transcribe,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
            batch_size=len(files_to_transcribe),
        )

        if isinstance(batch_results, dict):
            batch_results = [batch_results]

        for batch_idx, result in zip(file_indices, batch_results):
            text = result.get("text", "").strip()
            results[batch_idx] = text

            cache_path = batch_caches[batch_idx]
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(text)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        error_msg = str(e).lower()
        if (
            "out of memory" in error_msg
            or "cuda" in error_msg
            or "different keys" in error_msg
        ):
            # OOM or batching error - process with recursive splitting strategy
            print(
                f"\n⚠️  Error with batch ({len(files_to_transcribe)} files, "
                f"{total_duration:.1f}s total): {type(e).__name__}"
            )

            # Aggressive memory cleanup before splitting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # If batch has more than 1 file, split in half and retry
            if len(files_to_transcribe) > 1:
                print("    → Splitting batch into 2 smaller batches...")
                mid = len(files_to_transcribe) // 2

                # Process first half
                # first_half_files = [files_to_transcribe[i] for i in range(mid)]
                first_half_caches = [batch_caches[file_indices[i]] for i in range(mid)]
                first_half_indices = [file_indices[i] for i in range(mid)]

                first_results = _transcribe_batch(
                    [batch_files[idx] for idx in first_half_indices],
                    first_half_caches,
                    model,
                    cache,
                )
                for local_idx, batch_idx in enumerate(first_half_indices):
                    results[batch_idx] = first_results[local_idx]

                # Process second half
                # second_half_files = [
                #     files_to_transcribe[i]
                #     for i in range(mid, len(files_to_transcribe))
                # ]
                second_half_caches = [
                    batch_caches[file_indices[i]]
                    for i in range(mid, len(files_to_transcribe))
                ]
                second_half_indices = [
                    file_indices[i] for i in range(mid, len(files_to_transcribe))
                ]

                second_results = _transcribe_batch(
                    [batch_files[idx] for idx in second_half_indices],
                    second_half_caches,
                    model,
                    cache,
                )
                for local_idx, batch_idx in enumerate(second_half_indices):
                    results[batch_idx] = second_results[local_idx]
            else:
                # Single file OOM - use retry with aggressive cleanup
                print("    → Processing single file with retry logic...")
                batch_idx = file_indices[0]
                seg_file = files_to_transcribe[0]
                txt_cache = batch_caches[batch_idx]

                text = _transcribe_single_with_retry(
                    seg_file, txt_cache, model, generate_kwargs, max_retries=3
                )
                results[batch_idx] = text
        else:
            raise  # Re-raise non-OOM/batching errors

    return results


def transcribe_segments(
    model: TransformersASRModel,
    segments: pd.DataFrame,
    audio_path: str,
    output_dir: str,
    speaker: str,
    *,
    file_prefix: Optional[str] = None,
    cache: bool = True,
    min_duration_samples: int = 1600,
    batch_size: float | None = 30.0,
    compress: bool = True,
) -> List[Dict[str, Any]]:
    """Run ASR on a set of time-stamped segments extracted from ``audio_path``.

    Parameters
    ----------
    model
        A loaded Whisper model obtained via :func:`load_whisper_model`.
    segments
        DataFrame with ``start_sec`` and ``end_sec`` columns that describe the
        regions to transcribe. A ``speaker`` column is optional and overrides
        the supplied ``speaker`` argument per row when present.
    audio_path
        Source waveform on disk from which to slice the segments.
    output_dir
        Directory where per-segment WAV and cached transcripts are written.
    speaker
        Identifier tagged on each transcription record.
    file_prefix
        Optional custom stem for generated filenames; defaults to ``speaker``.
    cache
        When ``True`` reuses cached transcripts when present.
    min_duration_samples
        Segments shorter than this many samples are skipped to avoid unstable
        recognitions.
    batch_size
        Maximum total audio duration (in seconds) to process in parallel.
        Default: 240 seconds. The pipeline will batch audio segments up to
        this total duration. Use ``None`` or <= 0 to process all remaining
        segments in one go.
    compress
        If ``True``, saves segment WAV files as 16-bit PCM to reduce disk usage
    """

    os.makedirs(output_dir, exist_ok=True)
    audio, sr = sf.read(audio_path)
    prefix = file_prefix or speaker

    # Step 1: Extract all segments to WAV files with progress bar
    segment_info = []  # List of segment metadata dicts

    for idx, seg in tqdm(
        segments.iterrows(), total=len(segments), desc="Extracting segments"
    ):
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        row_speaker = seg.get("speaker", speaker)
        seg_filename = os.path.join(
            output_dir,
            f"{prefix}_seg_{idx}_{start:.2f}_{end:.2f}.wav",
        )
        txt_cache = seg_filename.replace(".wav", ".txt")

        start_samp = int(start * sr)
        end_samp = int(end * sr)
        segment_audio = audio[start_samp:end_samp]

        # Skip segments that are too short
        if len(segment_audio) < min_duration_samples:
            segment_info.append(
                {
                    "idx": idx,
                    "speaker": row_speaker,
                    "start_sec": start,
                    "end_sec": end,
                    "transcription": "",
                    "skip": True,
                }
            )
            continue

        # Save segment to WAV file (even if cached, for consistency)
        if not os.path.exists(seg_filename):
            _save_segment_wav(seg_filename, segment_audio, sr=sr, compress=compress)

        segment_info.append(
            {
                "idx": idx,
                "speaker": row_speaker,
                "start_sec": start,
                "end_sec": end,
                "seg_filename": seg_filename,
                "txt_cache": txt_cache,
                "skip": False,
            }
        )

    # Step 2: Transcribe segments in batches with progress bar
    valid_segments = [s for s in segment_info if not s["skip"]]

    # Create results list matching segment_info order
    transcriptions = {}  # Maps seg_filename -> transcription text

    # Determine maximum batch duration
    max_batch_duration = (
        float(batch_size) if batch_size and batch_size > 0 else float("inf")
    )

    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_duration = 0.0

    for seg in valid_segments:
        seg_duration = float(seg["end_sec"] - seg["start_sec"])

        # If this segment alone exceeds the cap, process it alone
        if seg_duration > max_batch_duration:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0
            batches.append([seg])
            continue

        # Start a new batch if adding would exceed the cap
        if current_batch and current_duration + seg_duration > max_batch_duration:
            batches.append(current_batch)
            current_batch = [seg]
            current_duration = seg_duration
        else:
            current_batch.append(seg)
            current_duration += seg_duration

    if current_batch:
        batches.append(current_batch)

    # Process batches
    for batch in tqdm(batches, desc=f"Transcribing {len(batches)} batches"):
        # Extract batch file paths and caches
        batch_files = [s["seg_filename"] for s in batch]
        batch_caches = [s["txt_cache"] for s in batch]

        # Transcribe batch
        batch_texts = _transcribe_batch(batch_files, batch_caches, model, cache)

        # Store results
        for seg_info, text in zip(batch, batch_texts):
            transcriptions[seg_info["seg_filename"]] = text

        # Clear GPU memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Assemble final results in original order
    results = []
    for seg_info in segment_info:
        if seg_info["skip"]:
            results.append(
                {
                    "speaker": seg_info["speaker"],
                    "start_sec": seg_info["start_sec"],
                    "end_sec": seg_info["end_sec"],
                    "transcription": "",
                }
            )
        else:
            results.append(
                {
                    "speaker": seg_info["speaker"],
                    "start_sec": seg_info["start_sec"],
                    "end_sec": seg_info["end_sec"],
                    "transcription": transcriptions[seg_info["seg_filename"]],
                }
            )

    return results
