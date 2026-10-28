import numpy as np
from services.sound_service import SoundService


def test_silence():
    service = SoundService()
    result = service.detect(np.zeros(16_000 * 3, dtype=np.float32))
    assert result["has_sound_events"] is False
    assert result["windows_classified"] == 0
    print("PASS: silence handled")


def test_bad_sample_rate():
    service = SoundService()
    try:
        service.detect(np.zeros(16_000, dtype=np.float32), sample_rate=44_100)
    except ValueError:
        print("PASS: wrong sample rate rejected")
        return
    raise AssertionError("Expected ValueError")


def run_real_file(path="data/sample_audio/environment.wav"):
    from pathlib import Path
    from services.audio_processor import process_audio_bytes

    p = Path(path)
    audio = process_audio_bytes(p.read_bytes(), p.name)
    result = SoundService().detect(audio["audio"], audio["sample_rate"])
    print("\nOverall:")
    for item in result["overall_predictions"]:
        print(f'  {item["label"]}: mean={item["mean_score"]} max={item["max_score"]}')


if __name__ == "__main__":
    test_silence()
    test_bad_sample_rate()
    run_real_file()