from services.fusion_service import FusionService


def test_fusion():
    speech = {
        "has_speech": True, "language": "en", "language_probability": 0.9,
        "transcript": "Hello there", "model": "base",
        "segments": [{"start": 1.0, "end": 3.0, "text": "Hello there",
                      "avg_logprob": -0.3, "no_speech_prob": 0.05}],
    }
    sound = {
        "model": "ast", "duration_seconds": 6.0, "window_seconds": 3.0,
        "overall_predictions": [
            {"label": "Speech", "mean_score": 0.7, "max_score": 0.8, "windows_present": 2},
            {"label": "Bicycle bell", "mean_score": 0.4, "max_score": 0.9, "windows_present": 1},
        ],
        "chunks": [
            {"start": 0.0, "end": 3.0, "skipped_silence": False,
             "predictions": [{"label": "Speech", "score": 0.8},
                             {"label": "Bicycle bell", "score": 0.9}]},
            {"start": 3.0, "end": 6.0, "skipped_silence": True, "predictions": []},
        ],
    }
    result = FusionService().fuse(speech, sound)

    assert result["background_sounds"][0]["label"] == "Bicycle bell"
    assert result["speech_related_sound_labels"][0]["label"] == "Speech"
    nearby = result["alignments"][0]["overlapping_background_sounds"]
    assert nearby[0]["label"] == "Bicycle bell"
    assert len(result["timeline"]) == 2          # silent window excluded
    assert result["duration_seconds"] == 6.0
    print("PASS: Fusion")


def test_bad_input():
    try:
        FusionService().fuse("x", {})
    except ValueError:
        print("PASS: bad input rejected")
        return
    raise AssertionError("Expected ValueError")


if __name__ == "__main__":
    test_fusion()
    test_bad_input()