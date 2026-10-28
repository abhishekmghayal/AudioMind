import json
import time
from pathlib import Path

from services.audio_processor import process_audio_bytes
from services.speech_service import SpeechService
from services.sound_service import SoundService
from services.fusion_service import FusionService


class AudioPipeline:
    def __init__(
        self,
        speech_model_size="base",
        speech_language=None,
        sound_top_k=5,
        sound_window_seconds=3.0,
        sound_hop_seconds=1.5,
        sound_min_score=0.1,
        sound_batch_size=4,
        device="cpu",
    ):
        self.speech_language = speech_language

        print("Loading speech model...")
        self.speech_service = SpeechService(model_size=speech_model_size, beam_size=5)

        print("Loading environmental sound model...")
        self.sound_service = SoundService(
            top_k=sound_top_k,
            window_seconds=sound_window_seconds,
            hop_seconds=sound_hop_seconds,
            min_score=sound_min_score,
            batch_size=sound_batch_size,
            device=device,
        )
        self.fusion_service = FusionService()

    @staticmethod
    def find_gaps(segments, duration, min_gap=2.0, margin=0.2):
        """Time ranges with no speech. `margin` keeps away from speech edges."""
        gaps, cursor = [], 0.0
        for s in sorted(segments, key=lambda x: x["start"]):
            end = s["start"] - margin
            if end - cursor >= min_gap:
                gaps.append((round(cursor, 2), round(end, 2)))
            cursor = max(cursor, s["end"] + margin)
        if duration - cursor >= min_gap:
            gaps.append((round(cursor, 2), round(duration, 2)))
        return gaps

    def analyze_bytes(self, file_bytes, filename, speech_language="__default__"):
        language = self.speech_language if speech_language == "__default__" else speech_language
        timings = {}

        t = time.perf_counter()
        processed = process_audio_bytes(file_bytes, filename)
        timings["preprocess_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        speech_result = self.speech_service.transcribe(
            processed["audio"], sample_rate=processed["sample_rate"], language=language
        )
        timings["speech_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        sound_result = self.sound_service.detect(
            processed["audio"], sample_rate=processed["sample_rate"]
        )
        gaps = self.find_gaps(speech_result["segments"], processed["duration_seconds"],min_gap=2.0)
        pause_result = self.sound_service.detect_in_ranges(
            processed["audio"], gaps, sample_rate=processed["sample_rate"]
        )
        timings["sound_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        fused_result = self.fusion_service.fuse(
            speech_result, sound_result,
            duration_seconds=processed["duration_seconds"],
            pause_result=pause_result,
        )
        timings["fusion_s"] = round(time.perf_counter() - t, 2)

        return {
            "filename": filename,
            "audio": {
                "duration_seconds": round(processed["duration_seconds"], 2),
                "sample_rate": processed["sample_rate"],
                "channels": processed["channels"],
                "processed_format": processed["processed_format"],
            },
            "speech": speech_result,
            "sound": sound_result,
            "pause_sound": pause_result,
            "fused": fused_result,
            "timings": timings,
        }

    def analyze_file(self, audio_path, speech_language="__default__"):
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return self.analyze_bytes(path.read_bytes(), path.name, speech_language)

    @staticmethod
    def save_json(result, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return path