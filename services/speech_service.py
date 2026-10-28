import numpy as np
from faster_whisper import WhisperModel


class SpeechService:
    def __init__(self, model_size="base", beam_size=5):
        self.model_size = model_size
        self.beam_size = beam_size
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio, sample_rate=16_000, language=None, vad_filter=True):
        if sample_rate != 16_000:
            raise ValueError("Speech service expects 16 kHz audio.")
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise ValueError("Audio must be a non-empty 1-D NumPy array.")

        audio = audio.astype(np.float32, copy=False)

        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=self.beam_size,
                language=language,                  # None = auto-detect
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400},
                condition_on_previous_text=False,   # reduces drift / fake timestamps
            )
            results = [
                {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text.strip(),
                    "avg_logprob": round(s.avg_logprob, 3),
                    "no_speech_prob": round(s.no_speech_prob, 3),
                }
                for s in segments
            ]
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        has_speech = len(results) > 0
        return {
            "has_speech": has_speech,
            "language": info.language if has_speech else None,
            "language_probability": round(info.language_probability, 3) if has_speech else None,
            "language_forced": language is not None,
            "vad_filter": vad_filter,
            "transcript": " ".join(r["text"] for r in results).strip(),
            "segments": results,
            "model": self.model_size,
        }