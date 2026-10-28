
from pathlib import Path

from services.audio_processor import process_audio_bytes
from services.speech_service import SpeechService


def main():
    audio_path = Path("data/sample_audio/speech.wav")

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Recording not found: {audio_path}\n"
            "Save a real speech recording at this path first."
        )

    print("1. Preprocessing audio...")

    processed = process_audio_bytes(
        audio_path.read_bytes(),
        audio_path.name,
    )

    print("Duration:", processed["duration_seconds"])
    print("Sample rate:", processed["sample_rate"])

    print("\n2. Loading Faster-Whisper...")
    speech_service = SpeechService(
        model_size="small",
        beam_size=5,
    )

    print("\n3. Transcribing...")
    result = speech_service.transcribe(
        processed["audio"],
        sample_rate=processed["sample_rate"],
    )

    print("\n--- TRANSCRIPTION RESULTS ---")
    print("Has speech segments:", result["has_speech"])
    print("Detected language:", result["language"])
    print("Language probability:", result["language_probability"])
    print("Transcript:", result["transcript"])
    print("Model:", result["model"])

    print("\n--- TIMESTAMPED SEGMENTS ---")
    for segment in result["segments"]:
        print(
            f'[{segment["start"]:.2f}s - '
            f'{segment["end"]:.2f}s] '
            f'{segment["text"]}'
        )

    print("\nSpeech service test completed.")


if __name__ == "__main__":
    main()
