import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class SoundService:
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE = 16_000
    SILENCE_RMS = 1e-4
    SPEECH_WORDS = ("speech", "conversation", "narration", "monologue", "babbling", "speaking")
    BG_WINDOW_SCORE = 0.08   # a window "contains" a sound at this score

    def __init__(self, top_k=5, window_seconds=10.0, hop_seconds=None,
                 min_score=0.1, batch_size=4, device="cpu"):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if hop_seconds is not None and hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive.")

        self.top_k = top_k
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds or window_seconds
        self.min_score = min_score
        self.batch_size = batch_size
        self.device = torch.device(device)

        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
        except Exception as exc:
            raise RuntimeError(f"Could not load sound model: {exc}") from exc

        self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label
        self.speech_idx = np.array(
            [i for i, l in self.labels.items()
             if any(w in l.lower() for w in self.SPEECH_WORDS)], dtype=int
        )

    # ---------- helpers ----------
    @staticmethod
    def _normalize(chunk, target_rms=0.05, max_gain=10.0):
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < 1e-6:
            return chunk
        gain = float(np.clip(target_rms / rms, 1.0, max_gain))   # boost only, never attenuate
        return np.clip(chunk * gain, -1.0, 1.0).astype(np.float32)

    def _probabilities(self, chunks):
        """Return an (n_chunks, n_classes) array of sigmoid scores."""
        outputs = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            inputs = self.extractor(batch, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _top(self, probs, k):
        return np.argsort(probs)[::-1][:k]

    def _remove_speech(self, probs):
        bg = probs.copy()
        if len(bg) and len(self.speech_idx):
            bg[:, self.speech_idx] = 0.0
        return bg

    def _background_summary(self, bg):
        """Per-label mean/max/persistence over windows, speech classes already zeroed."""
        if len(bg) == 0:
            return []
        b_mean, b_max = bg.mean(axis=0), bg.max(axis=0)
        flags = bg >= self.BG_WINDOW_SCORE
        persist, wins = flags.mean(axis=0), flags.sum(axis=0)
        picked = list(dict.fromkeys(
            [int(j) for j in self._top(b_mean, 8)] + [int(j) for j in self._top(b_max, 8)]
        ))
        out = []
        for j in picked:
            if b_max[j] < 0.05:
                continue
            out.append({
                "label": self.labels[j],
                "mean_score": round(float(b_mean[j]), 4),
                "max_score": round(float(b_max[j]), 4),
                "persistence": round(float(persist[j]), 3),
                "windows_present": int(wins[j]),
                "max_run": self._max_run(flags[:, j]),
            })
        out.sort(key=lambda x: x["max_score"], reverse=True)
        return out

    def _overall(self, probs):
        overall = []
        if len(probs) > 0:
            mean_p, max_p = probs.mean(axis=0), probs.max(axis=0)
            present = (probs >= self.min_score).sum(axis=0)
            for j in self._top(mean_p, self.top_k):
                if max_p[j] < self.min_score:
                    continue
                overall.append({
                    "label": self.labels[int(j)],
                    "mean_score": round(float(mean_p[j]), 4),
                    "max_score": round(float(max_p[j]), 4),
                    "windows_present": int(present[j]),
                })
        return overall

    def _classify(self, windows, normalize):
        """windows: list of (start, end, chunk). Returns (active_idx, probs)."""
        active = [i for i, (_, _, c) in enumerate(windows)
                  if float(np.sqrt(np.mean(c ** 2))) >= self.SILENCE_RMS]
        try:
            batch = [self._normalize(windows[i][2]) if normalize else windows[i][2]
                     for i in active]
            probs = (self._probabilities(batch) if active
                     else np.zeros((0, len(self.labels)), dtype=np.float32))
        except Exception as exc:
            raise RuntimeError(f"Sound classification failed: {exc}") from exc
        return active, probs

    def _check(self, audio, sample_rate):
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Sound service expects 16 kHz audio.")
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise ValueError("Audio must be a non-empty 1-D NumPy array.")
        audio = audio.astype(np.float32, copy=False)
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid samples.")
        return audio

    @staticmethod
    def _max_run(flags_col):
        best = cur = 0
        for v in flags_col:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best
    
    # ---------- whole-file detection ----------
    def detect(self, audio, sample_rate=16_000, normalize=True):
        audio = self._check(audio, sample_rate)

        win = int(self.window_seconds * self.SAMPLE_RATE)
        hop = int(self.hop_seconds * self.SAMPLE_RATE)
        last_start = max(0, len(audio) - win)
        starts = list(range(0, last_start + 1, hop))
        if starts[-1] != last_start:
            starts.append(last_start)

        windows = [(s, s + len(audio[s:s + win]), audio[s:s + win]) for s in starts]
        active, probs = self._classify(windows, normalize)
        bg = self._remove_speech(probs)
        row_of = {i: n for n, i in enumerate(active)}

        chunk_results = []
        for i, (s, e, _) in enumerate(windows):
            entry = {
                "start": round(s / self.SAMPLE_RATE, 2),
                "end": round(e / self.SAMPLE_RATE, 2),
                "skipped_silence": i not in row_of,
                "predictions": [],
                "background_predictions": [],
            }
            if i in row_of:
                p, b = probs[row_of[i]], bg[row_of[i]]
                entry["predictions"] = [
                    {"label": self.labels[int(j)], "score": round(float(p[j]), 4)}
                    for j in self._top(p, self.top_k) if p[j] >= self.min_score
                ]
                entry["background_predictions"] = [
                    {"label": self.labels[int(j)], "score": round(float(b[j]), 4)}
                    for j in self._top(b, 5) if b[j] >= 0.05
                ]
            chunk_results.append(entry)

        overall = self._overall(probs)
        return {
            "model": self.MODEL_ID,
            "sample_rate": sample_rate,
            "duration_seconds": round(len(audio) / sample_rate, 2),
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "min_score": self.min_score,
            "normalized": normalize,
            "windows_total": len(windows),
            "windows_classified": len(active),
            "has_sound_events": len(overall) > 0,
            "overall_predictions": overall,
            "background": self._background_summary(bg),
            "chunks": chunk_results,
            "note": (
                "Scores are sigmoid outputs of a pretrained AudioSet classifier, "
                "not calibrated probabilities. Timestamps are window-level, "
                "not exact event onsets. Speech classes are excluded from 'background'."
            ),
        }

    # ---------- classify only chosen time ranges (e.g. pauses) ----------
    def detect_in_ranges(self, audio, ranges, sample_rate=16_000,
                         min_range_seconds=1.0, normalize=True):
        audio = self._check(audio, sample_rate)

        win = int(self.window_seconds * self.SAMPLE_RATE)
        hop = int(self.hop_seconds * self.SAMPLE_RATE)
        min_len = int(min_range_seconds * self.SAMPLE_RATE)

        windows = []
        for r_start, r_end in ranges:
            a = max(0, int(r_start * self.SAMPLE_RATE))
            b = min(int(r_end * self.SAMPLE_RATE), len(audio))
            if b - a < min_len:
                continue
            last_start = max(a, b - win)
            starts = list(range(a, last_start + 1, hop))
            if starts[-1] != last_start:
                starts.append(last_start)
            for s in starts:
                e = min(s + win, b)
                windows.append((s, e, audio[s:e]))

        active, probs = self._classify(windows, normalize)
        bg = self._remove_speech(probs)

        return {
            "ranges": [[round(a, 2), round(b, 2)] for a, b in ranges],
            "windows_classified": len(active),
            "overall_predictions": self._overall(probs),
            "background": self._background_summary(bg),
            "chunks": [
                {"start": round(windows[i][0] / self.SAMPLE_RATE, 2),
                 "end": round(windows[i][1] / self.SAMPLE_RATE, 2)}
                for i in active
            ],
        }