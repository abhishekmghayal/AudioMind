import re


class QAService:
    """Deterministic, evidence-grounded question answering over a fused analysis.

    Answers use only fields in the analysis. Each answer says whether it is an
    observed model output, an insufficient-evidence reply, or unsupported.
    """

    GENERIC_SOUND_WORDS = {
        "sound", "sounds", "noise", "noises", "background", "audio",
        "speech", "voice", "thing", "anything", "something", "event", "events",
    }
    STOPWORDS = {
        "what", "when", "while", "during", "said", "say", "says", "speaker",
        "there", "were", "was", "that", "this", "with", "heard", "hear",
        "about", "the", "and", "did", "does", "any", "which", "who", "they",
    }

    def answer(self, question: str, analysis: dict) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty.")
        if not isinstance(analysis, dict):
            raise ValueError("Analysis must be a dictionary.")

        fused = analysis.get("fused", analysis)
        if not isinstance(fused, dict):
            raise ValueError("Fused analysis must be a dictionary.")

        q = re.sub(r"\s+", " ", question.strip().lower())
        warnings = fused.get("warnings", []) if isinstance(fused.get("warnings"), list) else []
        self._warnings = warnings

        # 1. Questions tied to a time or a spoken phrase
        t = re.search(r"\b(?:at|around|near|after|before)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b", q)
        if t:
            return self._at_time(float(t.group(1)), fused)
        if re.search(r"\b(when|while|during)\b", q) and re.search(
            r"\b(said|say|says|spoke|speaking|mention|mentioned|talk|talking)\b", q
        ):
            return self._sounds_around_speech(q, fused)

        # 2. Specific sound ("was there a dog?")
        specific = self._specific_sound_target(q)
        if specific:
            return self._specific_sound(specific, fused)

        # 3. General intents
        if re.search(r"\b(limitation|limitations|uncertain|uncertainty|confidence|reliable|"
                     r"warnings?|how accurate|trust)\b", q):
            return self._uncertainty(fused)
        if re.search(r"\b(language|languages)\b", q):
            return self._language(fused)
        if re.search(r"\b(summary|summarize|summarise|describe|overview|overall|"
                     r"what is happening|what's happening|scene)\b", q):
            return self._summary(fused)
        if re.search(r"\b(sound|sounds|noise|noises|background|environment|environmental|"
                     r"detected|events?)\b", q):
            return self._sounds(fused)
        if re.search(r"\b(is|was|were)\b.*\b(speech|speaking|speaker|talking|anyone|anybody|voice)\b", q) \
                and not re.search(r"\b(what|which)\b", q):
            return self._speech_presence(fused)
        if re.search(r"\b(transcript|said|say|says|spoken|speak|words|speech|talk|talked|"
                     r"conversation|mention)\b", q):
            return self._transcript(fused)
        if re.search(r"\b(timeline|segments?|timestamps?)\b", q):
            return self._timeline(fused)

        return self._result(
            "I cannot answer that reliably from the current audio analysis. "
            "I can answer questions about what was said, the language, background "
            "sounds, a specific sound, what happened while a phrase was spoken, "
            "or the limitations of the analysis.",
            "unsupported_question", "unsupported", [], grounded=False,
        )

    # ---------- helpers ----------
    def _result(self, answer, category, kind, evidence, grounded=True):
        return {
            "answer": answer,
            "category": category,
            "answer_type": kind,          # observed | insufficient | unsupported
            "evidence": evidence,
            "warnings": self._warnings,
            "grounded_in_analysis": grounded,
        }

    @staticmethod
    def _segments(fused):
        segs = fused.get("speech_segments", [])
        return [s for s in segs if isinstance(s, dict)] if isinstance(segs, list) else []

    def _transcript_text(self, fused):
        text = fused.get("transcript")
        if text is None:
            text = " ".join(str(s.get("text", "")).strip() for s in self._segments(fused)).strip()
        return str(text).strip()

    @staticmethod
    def _label_words(label):
        return [w.rstrip("s") for w in re.findall(r"[a-z]+", label.lower()) if len(w) > 2]

    # ---------- intents ----------
    def _transcript(self, fused):
        if not fused.get("has_speech", True):
            return self._result("No speech was detected in this recording.",
                                "speech_transcription", "observed", [])
        text = self._transcript_text(fused)
        if not text:
            return self._result("No transcript is available in the analysis.",
                                "speech_transcription", "insufficient", [])
        segs = self._segments(fused)
        flagged = [s for s in segs if s.get("flags")]
        note = (f" Note: {len(flagged)} segment(s) were flagged as low confidence, "
                "so some words may be wrong.") if flagged else ""
        forced = fused.get("language_forced")
        lang = fused.get("language")
        lang_note = f" (language {'set to' if forced else 'auto-detected as'} '{lang}')" if lang else ""
        return self._result(
            f'The recognised transcript{lang_note} is: "{text}".{note}',
            "speech_transcription", "observed",
            [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text")} for s in segs],
        )

    def _speech_presence(self, fused):
        detected = fused.get("has_speech")
        if detected is True:
            return self._result("Yes, the speech model found speech in the recording.",
                                "speech_detection", "observed", [])
        if detected is False:
            return self._result("No, the speech model did not find speech.",
                                "speech_detection", "observed", [])
        return self._result("Speech presence is not specified in the analysis.",
                            "speech_detection", "insufficient", [])

    def _language(self, fused):
        if not fused.get("has_speech", True) or not fused.get("language"):
            return self._result("No speech language was identified.", "language",
                                "insufficient", [])
        how = "set by the user" if fused.get("language_forced") else "auto-detected"
        prob = fused.get("language_probability")
        extra = f" with model confidence {prob}" if prob is not None else ""
        warn = ""
        if not fused.get("language_forced") and prob is not None and prob < 0.6:
            warn = " The confidence is low, so the detected language may be wrong."
        return self._result(f"The speech language was {how} as '{fused['language']}'{extra}.{warn}",
                            "language", "observed", [])

    def _sounds(self, fused):
        pause = fused.get("sounds_in_pauses", []) or []
        if pause and not (fused.get("background_sounds") or []):
            items = ", ".join(f'{p["label"]} (max score {p["max_score"]})' for p in pause)
            return self._result(
                "No background sound was found in the whole recording, but the pauses "
                f"between speech contained: {items}. They are not proven to be present "
                "during speech.",
                "sound_evidence", "observed", pause)
        bg = fused.get("background_sounds", []) or []
        weak = fused.get("weak_sound_candidates", []) or []
        thr = fused.get("background_min_score", 0.3)
        if bg:
            items = ", ".join(f'{b.get("label")} (max score {b.get("max_score")})' for b in bg)
            return self._result(
                f"Background sounds reported: {items}. These are model scores, not "
                "calibrated probabilities, and times are window-level.",
                "sound_evidence", "observed", bg)
        if weak:
            items = ", ".join(f'{w.get("label")} ({w.get("max_score")})' for w in weak[:3])
            return self._result(
                f"No background sound reached the {thr} score threshold. Weak, unconfirmed "
                f"candidates: {items}. These should not be treated as detections.",
                "sound_evidence", "insufficient", weak)
        return self._result(
            "No background sound was reported. This does not prove that none was present; "
            "quiet sounds under speech can be missed.",
            "sound_evidence", "observed", [])

    def _specific_sound_target(self, q):
        m = re.search(r"\b(?:is|was|were|are)\s+there\s+(?:a|an|any|some)?\s*([a-z][a-z\- ]{1,30})", q)
        if not m:
            m = re.search(r"\b(?:did|do|does)\s+(?:you|it|the model)?\s*(?:hear|detect|find)\s+(?:a|an|any)?\s*([a-z][a-z\- ]{1,30})", q)
        if not m:
            return None
        words = [w for w in re.findall(r"[a-z]+", m.group(1)) if w not in self.STOPWORDS]
        words = [w for w in words if w not in self.GENERIC_SOUND_WORDS]
        return words[0].rstrip("s") if words else None

    def _specific_sound(self, target, fused):
        def hit(items):
            return [i for i in items if isinstance(i, dict) and target in self._label_words(str(i.get("label", "")))]

        bg = hit(fused.get("background_sounds", []) or [])
        if bg:
            b = bg[0]
            return self._result(
                f"Yes, the sound model reported '{b['label']}' with max score {b['max_score']}, "
                "above the background threshold. This is a model output, not confirmation.",
                "specific_sound", "observed", bg)

        weak = hit(fused.get("weak_sound_candidates", []) or [])
        if weak:
            w = weak[0]
            return self._result(
                f"Not confirmed. '{w['label']}' appeared only weakly (max score {w['max_score']}), "
                "below the background threshold.",
                "specific_sound", "insufficient", weak)

        # Search the raw per-window evidence (new block)
        win_hits = []
        for e in fused.get("timeline", []) or []:
            if e.get("type") != "sound_window":
                continue
            for p in e.get("content", []):
                if target in self._label_words(str(p.get("label", ""))):
                    win_hits.append({"label": p["label"], "score": p["score"],
                                     "window": [e["start"], e["end"]]})
        if win_hits:
            best = max(win_hits, key=lambda x: x["score"])
            return self._result(
                f"Not confirmed. '{best['label']}' appeared in {len(win_hits)} window(s), "
                f"at most {best['score']} (window {best['window'][0]}s to {best['window'][1]}s), "
                "below the background threshold.",
                "specific_sound", "insufficient", win_hits)

        return self._result(
            f"The model did not report anything matching '{target}'. That does not prove it was "
            "absent, only that the sound model did not report it.",
            "specific_sound", "insufficient", [])

    def _find_segment(self, q, fused):
        keys = [w for w in re.findall(r"[a-z']+", q) if len(w) > 2 and w not in self.STOPWORDS]
        best, best_hits = None, 0
        for a in fused.get("alignments", []) or []:
            text = str(a.get("text", "")).lower()
            hits = sum(1 for k in keys if k in text)
            if hits > best_hits:
                best, best_hits = a, hits
        return best

    def _describe_alignment(self, a):
        span = f'{a["start"]}s to {a["end"]}s'
        sounds = a.get("overlapping_background_sounds", [])
        if not sounds:
            return self._result(
                f'While "{a["text"]}" was said ({span}), no background sound reached the '
                "detection threshold.", "speech_sound_alignment", "observed", [])
        items = ", ".join(f'{s["label"]} (max score {s["max_score"]})' for s in sounds)
        return self._result(
            f'While "{a["text"]}" was said ({span}), the sound model reported: {items}. '
            "This means they share a time window, not that they were proven to occur together.",
            "speech_sound_alignment", "observed", sounds)

    def _sounds_around_speech(self, q, fused):
        a = self._find_segment(q, fused)
        if a is None:
            return self._result(
                "Insufficient evidence: I could not match that phrase to the transcript.",
                "speech_sound_alignment", "insufficient", [])
        return self._describe_alignment(a)

    def _at_time(self, t, fused):
        for a in fused.get("alignments", []) or []:
            if a["start"] <= t <= a["end"]:
                return self._describe_alignment(a)
        return self._result(
            f"No speech segment covers {t} seconds, so I cannot report what was said there.",
            "speech_sound_alignment", "insufficient", [])

    def _timeline(self, fused):
        tl = fused.get("timeline", [])
        if not isinstance(tl, list) or not tl:
            return self._result("No timeline entries are available.", "timeline",
                                "insufficient", [])
        n_speech = sum(1 for e in tl if e.get("type") == "speech")
        n_sound = sum(1 for e in tl if e.get("type") == "sound_window")
        return self._result(
            f"The timeline has {n_speech} speech segment(s) and {n_sound} sound window(s). "
            "Sound timestamps are window-level, not exact onsets.",
            "timeline", "observed", [])

    def _summary(self, fused):
        parts = []
        if fused.get("duration_seconds") is not None:
            parts.append(f"The recording is {fused['duration_seconds']} seconds long.")
        if fused.get("has_speech"):
            parts.append(f"Speech was detected (language: {fused.get('language')}).")
            text = self._transcript_text(fused)
            if text:
                parts.append(f'Transcript: "{text}"')
        else:
            parts.append("No speech was detected.")
        bg = fused.get("background_sounds", []) or []
        if bg:
            parts.append("Background sounds: " + ", ".join(str(b.get("label")) for b in bg) + ".")
        else:
            parts.append("No background sound reached the confidence threshold.")
        parts.append("This lists detected evidence only; it does not infer what is happening in the scene.")
        return self._result(" ".join(parts), "summary", "observed", [])

    def _uncertainty(self, fused):
        warnings = fused.get("warnings", []) or []
        limits = fused.get("limitations", []) or []
        items = [str(i) for i in list(warnings) + list(limits)]
        if not items:
            return self._result(
                "No warnings were recorded, but that does not establish that every prediction is correct.",
                "uncertainty", "observed", [])
        return self._result("Relevant cautions: " + " ".join(items), "uncertainty", "observed", [])