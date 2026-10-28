class FusionService:
    """Combine speech and sound results into one timestamped evidence object.

    Only observed model outputs are used. Nothing is inferred, invented, or
    given a confidence value that the models did not produce.
    """

    LOW_LOGPROB = -1.0
    HIGH_NO_SPEECH = 0.6
    BACKGROUND_MIN_SCORE = 0.3     # one window this high = candidate for "strong"
    LIKELY_MIN_PERSISTENCE = 0.3   # present in >= 30% of windows = "likely"
    LIKELY_MIN_MEAN = 0.05
    MIN_PAUSE_WINDOWS = 2          # a pause sound must appear in >= 2 windows

    LIMITATIONS = [
        "Sound labels come from a pretrained AudioSet classifier; scores are "
        "model outputs, not calibrated probabilities.",
        "Sound timestamps are window-level, not exact event onsets.",
        "Overlap between a speech segment and a sound window means they share "
        "a time window, not that they were proven to occur together.",
        "A sound found only in pauses is not proven to be present during speech.",
        "Speaker identity, emotion and meaning are not analysed.",
    ]

    @staticmethod
    def _is_speech_label(label):
        low = label.lower()
        return any(w in low for w in ("speech", "conversation", "monologue", "narration"))

    @staticmethod
    def _overlap(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    def _level(self, item):
        mx = item.get("max_score") or 0.0
        # Strong: a high score that is also sustained (or very high)
        if mx >= self.BACKGROUND_MIN_SCORE and (
            item.get("max_run", 0) >= 2 or mx >= 0.5
        ):
            return "strong"
        # A single high-scoring window with no persistence: plausible, not confirmed
        if mx >= self.BACKGROUND_MIN_SCORE:
            return "likely"
        # Moderate score sustained over 3+ consecutive windows
        if item.get("max_run", 0) >= 3 and mx >= 0.15:
            return "likely"
        # Present in many windows overall (needs at least 3 windows)
        if (item.get("persistence") is not None
                and item["persistence"] >= self.LIKELY_MIN_PERSISTENCE
                and item.get("windows_present", 0) >= 3
                and (item.get("mean_score") or 0.0) >= self.LIKELY_MIN_MEAN):
            return "likely"
        return "weak"

    def _background_items(self, sound_result):
        """Use the speech-free 'background' list if present, else fall back."""
        items = sound_result.get("background")
        if items is not None:
            return items
        return [
            {"label": i["label"], "mean_score": i.get("mean_score", i.get("score")),
             "max_score": i.get("max_score"), "persistence": None,
             "windows_present": i.get("windows_present", i.get("chunks_present"))}
            for i in sound_result.get("overall_predictions", [])
            if not self._is_speech_label(i["label"])
        ]

    def fuse(self, speech_result, sound_result, duration_seconds=None, pause_result=None):
        if not isinstance(speech_result, dict):
            raise ValueError("speech_result must be a dictionary.")
        if not isinstance(sound_result, dict):
            raise ValueError("sound_result must be a dictionary.")

        warnings = []

        # ---- Speech evidence ----
        speech_segments = []
        for seg in speech_result.get("segments", []):
            flags = []
            if seg.get("avg_logprob") is not None and seg["avg_logprob"] < self.LOW_LOGPROB:
                flags.append("low_confidence_transcript")
            if seg.get("no_speech_prob") is not None and seg["no_speech_prob"] > self.HIGH_NO_SPEECH:
                flags.append("possible_non_speech_or_hallucination")
            speech_segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", "").strip(),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "flags": flags,
            })
        if any(s["flags"] for s in speech_segments):
            warnings.append("Some transcript segments are flagged as low confidence.")

        has_speech = bool(speech_result.get("has_speech", len(speech_segments) > 0))

        # ---- Sound windows ----
        sound_windows, silent_windows = [], 0
        for chunk in sound_result.get("chunks", []):
            if chunk.get("skipped_silence"):
                silent_windows += 1
                continue
            preds = chunk.get("predictions", [])
            bg_preds = chunk.get("background_predictions")
            if bg_preds is None:   # older format
                bg_preds = [p for p in preds if not self._is_speech_label(p["label"])]
            if not preds and not bg_preds:
                continue
            sound_windows.append({
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "predictions": preds,
                "background_predictions": bg_preds,
            })
        if silent_windows:
            warnings.append(f"{silent_windows} audio window(s) were silent and not classified.")

        # ---- Overall sound summary ----
        speech_related = [
            {"label": i["label"], "mean_score": i.get("mean_score"),
             "max_score": i.get("max_score"), "windows_present": i.get("windows_present")}
            for i in sound_result.get("overall_predictions", [])
            if self._is_speech_label(i["label"])
        ]

        background, weak_candidates = [], []
        for item in self._background_items(sound_result):
            entry = dict(item)
            entry["confidence_level"] = self._level(item)
            (weak_candidates if entry["confidence_level"] == "weak" else background).append(entry)

        if not sound_result.get("overall_predictions") and not background:
            warnings.append("No sound event passed the detection threshold.")
        elif not background:
            warnings.append(
                "No background sound was strong or persistent enough; "
                "weaker labels are listed as weak candidates only."
            )
        if not has_speech:
            warnings.append("No speech detected; transcript is empty.")
        if has_speech and not speech_related:
            warnings.append("Whisper found speech but the sound model gave no speech label.")

        # ---- Sounds found in pauses (no speech on top) ----
        sounds_in_pauses = []
        if pause_result is not None:
            if pause_result.get("windows_classified", 0) > 0:
                for item in pause_result.get("background", []):
                    if item.get("windows_present", 0) < self.MIN_PAUSE_WINDOWS:
                        continue
                    lvl = self._level(item)
                    if lvl != "weak":
                        e = dict(item)
                        e["confidence_level"] = lvl
                        sounds_in_pauses.append(e)
            else:
                warnings.append("No pause was long enough to classify sounds without speech.")

        whole_labels = {b["label"] for b in background}
        pause_only = [s for s in sounds_in_pauses if s["label"] not in whole_labels]
        if pause_only:
            warnings.append(
                "Sounds found only in pauses (missed in whole-file analysis): "
                + ", ".join(s["label"] for s in pause_only) + "."
            )

        # ---- Alignment: confirmed background sounds sharing a window with speech ----
        confirmed = {b["label"] for b in background}
        alignments = []
        for i, seg in enumerate(speech_segments):
            merged = {}
            for win in sound_windows:
                if self._overlap(seg["start"], seg["end"], win["start"], win["end"]) <= 0:
                    continue
                for p in win["background_predictions"]:
                    if p["label"] not in confirmed:
                        continue
                    m = merged.setdefault(p["label"], {
                        "label": p["label"], "max_score": 0.0,
                        "windows_present": 0, "windows": [],
                    })
                    m["max_score"] = max(m["max_score"], p["score"])
                    m["windows_present"] += 1
                    m["windows"].append([win["start"], win["end"]])
            alignments.append({
                "speech_index": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "overlapping_background_sounds":
                    sorted(merged.values(), key=lambda x: x["max_score"], reverse=True),
            })

        # ---- Timeline ----
        timeline = [
            {"type": "speech", "start": s["start"], "end": s["end"],
             "content": s["text"], "flags": s["flags"]}
            for s in speech_segments
        ] + [
            {"type": "sound_window", "start": w["start"], "end": w["end"],
             "content": [{"label": p["label"], "score": p["score"]} for p in w["predictions"]]}
            for w in sound_windows
        ]
        timeline.sort(key=lambda x: (x["start"], x["end"]))

        if duration_seconds is None:
            duration_seconds = max(
                float(sound_result.get("duration_seconds", 0) or 0),
                max((s["end"] for s in speech_segments), default=0.0),
            )

        return {
            "duration_seconds": round(float(duration_seconds), 2),
            "has_speech": has_speech,
            "language": speech_result.get("language"),
            "language_probability": speech_result.get("language_probability"),
            "language_forced": speech_result.get("language_forced", False),
            "transcript": speech_result.get("transcript", ""),
            "speech_segments": speech_segments,
            "background_sounds": background,
            "weak_sound_candidates": weak_candidates,
            "speech_related_sound_labels": speech_related,
            "sounds_in_pauses": sounds_in_pauses,
            "pause_only_sounds": pause_only,
            "pause_ranges": pause_result.get("ranges", []) if pause_result else [],
            "alignments": alignments,
            "timeline": timeline,
            "models": {
                "speech": speech_result.get("model"),
                "sound": sound_result.get("model"),
            },
            "sound_window_seconds": sound_result.get("window_seconds"),
            "background_min_score": self.BACKGROUND_MIN_SCORE,
            "warnings": warnings,
            "limitations": self.LIMITATIONS,
        }