
import io

import numpy as np
import soundfile as sf

from services.audio_processor import process_audio_bytes


def test_audio_preprocessing():
    # Generate a 2-second, 440 Hz test signal at 44.1 kHz.
    original_sr = 44_100
    duration = 2.0

    t = np.arange(
        int(original_sr * duration),
        dtype=np.float32,
    ) / original_sr

    audio = 0.2 * np.sin(2 * np.pi * 440 * t)

    buffer = io.BytesIO()
    sf.write(buffer, audio, original_sr, format="WAV")

    result = process_audio_bytes(buffer.getvalue(), "test.wav")

    assert result["sample_rate"] == 16_000
    assert result["channels"] == 1
    assert abs(result["duration_seconds"] - duration) < 0.02
    assert result["audio"].ndim == 1
    assert result["num_samples"] == len(result["audio"])

    print("PASS: Audio preprocessing")
    print("Sample rate:", result["sample_rate"])
    print("Duration:", round(result["duration_seconds"], 2), "seconds")
    print("Samples:", result["num_samples"])


if __name__ == "__main__":
    test_audio_preprocessing()
