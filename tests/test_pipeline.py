import sys
from pathlib import Path

from services.audio_pipeline import AudioPipeline


def main():
    audio_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample_audio/speech.wav")
    output_path = Path("outputs") / f"{audio_path.stem}_analysis.json"

    if not audio_path.is_file():
        raise FileNotFoundError(f"Test recording not found: {audio_path}")

    print("Initializing the audio pipeline...")
    pipeline = AudioPipeline()

    print("\nAnalyzing recording...")
    result = pipeline.analyze_file(audio_path)
    fused = result["fused"]

    print("\n--- AUDIO SUMMARY ---")
    print("Filename:", result["filename"])
    print("Duration:", result["audio"]["duration_seconds"])
    print("Transcript:", result["speech"]["transcript"] or "(no speech detected)")

    print("\n--- OVERALL SOUND PREDICTIONS ---")
    for item in result["sound"]["overall_predictions"]:
        print(item)

    print("\n--- FUSION SUMMARY ---")
    print("Speech detected:", fused["has_speech"])
    print("Background sounds:", [b["label"] for b in fused["background_sounds"]])
    print("Timeline entries:", len(fused["timeline"]))
    print("Warnings:", fused["warnings"])
    print("Timings (s):", result["timings"])

    # Structural checks: the pipeline produced a complete, consistent result.
    assert result["audio"]["sample_rate"] == 16_000
    assert fused["duration_seconds"] > 0
    assert "alignments" in fused and "limitations" in fused
    assert len(fused["speech_segments"]) == len(fused["alignments"])

    pipeline.save_json(result, output_path)
    print("\nSaved results to:", output_path)
    print("PASS: end-to-end pipeline completed.")


if __name__ == "__main__":
    main()