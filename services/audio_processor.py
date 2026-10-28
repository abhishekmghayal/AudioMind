
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TARGET_SAMPLE_RATE = 16_000
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_DURATION_SECONDS = 30 * 60     # 30 minutes


def process_audio_bytes(file_bytes: bytes, filename: str) -> dict:
    """Validate uploaded audio and convert it to mono, 16 kHz PCM WAV."""

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 100 MB size limit.")

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {extension}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / f"input{extension}"
        output_path = temp / "processed.wav"

        # Work on a temporary copy; preserve the original upload.
        input_path.write_bytes(file_bytes)

        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(input_path),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio conversion timed out.") from exc
        except subprocess.CalledProcessError as exc:
            lines = (exc.stderr or "").strip().splitlines()
            detail = f" ({lines[-1]})" if lines else ""
            raise ValueError(
                f"Could not decode this file. Check that it is valid audio.{detail}"
            ) from exc

        if not output_path.exists():
            raise ValueError("FFmpeg did not produce an output file.")

        try:
            audio, sample_rate = sf.read(
                str(output_path),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise ValueError("Could not read processed audio.") from exc

        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio is empty or contains invalid samples.")

        duration = len(audio) / sample_rate

        if duration <= 0:
            raise ValueError("Audio duration must be positive.")

        if duration > MAX_DURATION_SECONDS:
            raise ValueError("Audio exceeds the 30-minute duration limit.")

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("Unexpected output sample rate.")

        if audio.shape[1] != 1:
            raise ValueError("Expected mono audio.")

        audio = audio[:, 0]

        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "channels": 1,
            "original_filename": filename,
            "processed_format": "WAV PCM 16-bit",
            "num_samples": len(audio),
        }
