from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import streamlit as st

from services.audio_pipeline import AudioPipeline
from services.qa_service import QAService


st.set_page_config(
    page_title="AudioIntelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Japanese": "ja",
    "Chinese": "zh",
}
MODEL_SIZES = ["base", "small"]
SUPPORTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


@st.cache_resource(show_spinner="Loading speech and sound models (first run may take a while)...")
def get_pipeline(model_size: str) -> AudioPipeline:
    """Reuse expensive ML models between Streamlit reruns."""
    return AudioPipeline(speech_model_size=model_size)


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def file_fingerprint(file_bytes: bytes, filename: str, model_size: str, language_code: str | None) -> str:
    """Identify the exact upload and settings; don't show results from an older analysis."""
    digest = hashlib.sha256(file_bytes).hexdigest()
    settings = json.dumps(
        {"filename": filename, "sha256": digest, "model": model_size, "language": language_code},
        sort_keys=True,
    )
    return hashlib.sha256(settings.encode("utf-8")).hexdigest()


def clear_analysis_state() -> None:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("qa_history", None)


st.title("🎧 AudioIntelligence")
st.caption(
    "Joint speech and environmental-sound analysis with evidence-grounded answers. "
    "Prototype using pretrained Faster-Whisper and AudioSet-based AST models."
)

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Speech model",
        MODEL_SIZES,
        index=1,  # 'small' by default
        help="'small' is slower but usually more accurate, especially with noise and accents.",
    )
    lang_name = st.selectbox("Speech language", list(LANGUAGES.keys()), index=0)
    language_code = LANGUAGES[lang_name]
    st.divider()
    st.markdown(
        "**Implemented**\n"
        "- Audio preprocessing\n"
        "- Speech transcription\n"
        "- Environmental sound classification\n"
        "- Evidence fusion and timeline\n"
        "- Rule-based question answering\n\n"
        "**Future work**\n"
        "- Speaker diarization\n"
        "- Emotion analysis\n"
        "- Learned multimodal fusion\n"
        "- LLM-based reasoning"
    )

uploaded = st.file_uploader("Upload an audio recording", type=SUPPORTED_TYPES)

if uploaded is None:
    if st.session_state.get("active_analysis_key") is not None:
        clear_analysis_state()
        st.session_state["active_analysis_key"] = None
    st.info("Upload a WAV, MP3, FLAC, OGG, or M4A file to begin.")
else:
    audio_bytes = uploaded.getvalue()
    current_key = file_fingerprint(audio_bytes, uploaded.name, model_size, language_code)

    if st.session_state.get("active_analysis_key") != current_key:
        clear_analysis_state()
        st.session_state["active_analysis_key"] = current_key

    st.audio(audio_bytes)
    left, right = st.columns(2)
    left.write(f"**File:** `{uploaded.name}`")
    right.write(f"**Size:** {len(audio_bytes) / 1024:.1f} KB")

    analyze_clicked = st.button("Analyze recording", type="primary")
    if analyze_clicked:
        try:
            pipeline = get_pipeline(model_size)
            progress = st.empty()
            started = time.perf_counter()
            with st.spinner("Preprocessing audio, transcribing speech, and classifying sounds..."):
                analysis = pipeline.analyze_bytes(
                    audio_bytes,
                    uploaded.name,
                    speech_language=language_code,
                )
            elapsed = time.perf_counter() - started
            st.session_state["analysis_result"] = analysis
            st.session_state["qa_history"] = []
            st.session_state["active_analysis_key"] = current_key
            progress.success(f"Analysis completed in {elapsed:.1f} seconds.")
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            st.error(f"Analysis failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected analysis error: {type(exc).__name__}: {exc}")
            st.caption("Check the terminal for the full traceback if the problem persists.")

result = st.session_state.get("analysis_result")

if isinstance(result, dict) and isinstance(result.get("fused"), dict):
    fused = result["fused"]
    speech_segments = safe_list(fused.get("speech_segments"))
    background_sounds = safe_list(fused.get("background_sounds"))
    weak_candidates = safe_list(fused.get("weak_sound_candidates"))

    st.divider()
    st.subheader("Analysis overview")
    c1, c2, c3, c4 = st.columns(4)
    duration = result.get("audio", {}).get("duration_seconds", "—")
    c1.metric("Duration", f"{duration} s" if duration != "—" else "—")
    c2.metric("Detected language", fused.get("language") or "—")
    c3.metric("Speech segments", len(speech_segments))
    c4.metric("Background sound labels", len(background_sounds))

    for warning in safe_list(fused.get("warnings")):
        st.warning(str(warning))

    tab_transcript, tab_sounds, tab_qa, tab_raw = st.tabs(
        ["Transcript", "Sounds & timeline", "Ask a question", "Raw analysis"]
    )

    with tab_transcript:
        st.subheader("Recognized speech")
        transcript = str(fused.get("transcript") or "").strip()
        if fused.get("has_speech") and transcript:
            st.write(transcript)
            rows = [
                {
                    "Start (s)": segment.get("start"),
                    "End (s)": segment.get("end"),
                    "Text": segment.get("text", ""),
                    "Flags": ", ".join(safe_list(segment.get("flags"))),
                }
                for segment in speech_segments
            ]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No speech was detected, or no transcript was produced.")

        language_probability = fused.get("language_probability")
        if language_probability is not None:
            st.caption(
                f"Language confidence reported by the speech model: {language_probability}. "
                "This is a model score, not a guarantee of correctness."
            )

    with tab_sounds:
        threshold = fused.get("background_min_score", 0.3)
        st.subheader(f"Background sounds (strong threshold: {threshold})")
        if background_sounds:
            st.dataframe(background_sounds, width="stretch", hide_index=True)
        else:
            st.info("No background sound reached the configured confirmation threshold.")

        if weak_candidates:
            st.subheader("Weak sound candidates — unconfirmed")
            st.dataframe(weak_candidates, width="stretch", hide_index=True)
            st.caption("Weak candidates are not confirmed detections.")

        pause_sounds = safe_list(fused.get("sounds_in_pauses"))
        if pause_sounds:
            st.subheader("Sounds detected in speech-free pauses")
            st.dataframe(pause_sounds, width="stretch", hide_index=True)
            st.caption("Pause-only evidence does not establish that a sound occurred during speech.")

        alignments = safe_list(fused.get("alignments"))
        st.subheader("Speech and sound alignment")
        if alignments:
            alignment_rows = []
            for alignment in alignments:
                overlapping = safe_list(alignment.get("overlapping_background_sounds"))
                alignment_rows.append(
                    {
                        "Start (s)": alignment.get("start"),
                        "End (s)": alignment.get("end"),
                        "Speech": alignment.get("text", ""),
                        "Overlapping model labels": ", ".join(
                            str(sound.get("label", "")) for sound in overlapping
                        ) or "None above threshold",
                    }
                )
            st.dataframe(alignment_rows, width="stretch", hide_index=True)
            st.caption(
                "Temporal overlap means the predictions share a window; it does not prove "
                "the sound and speech were causally or perceptually linked."
            )
        else:
            st.info("No speech/sound alignments were produced.")

        timeline = safe_list(fused.get("timeline"))
        with st.expander(f"View timeline ({len(timeline)} entries)"):
            if timeline:
                timeline_rows = []
                for entry in timeline:
                    content = entry.get("content")
                    if isinstance(content, list):
                        content = ", ".join(
                            f'{c.get("label")} ({c.get("score")})' for c in content
                        )
                    timeline_rows.append({
                        "Type": entry.get("type"),
                        "Start (s)": entry.get("start"),
                        "End (s)": entry.get("end"),
                        "Content": content,
                        "Flags": ", ".join(safe_list(entry.get("flags"))),
                    })
                st.dataframe(timeline_rows, width="stretch", hide_index=True)
            else:
                st.info("No timeline entries are available.")

    with tab_qa:
        st.subheader("Ask about this recording")
        st.caption("Answers are grounded in the analysis; unsupported details will be identified as such.")

        if "qa_history" not in st.session_state:
            st.session_state["qa_history"] = []

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder="What was said? Was there a siren? What happened at 10 seconds?",
            )
            ask_clicked = st.form_submit_button("Ask", type="primary")

        if ask_clicked:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    answer = QAService().answer(question, result)
                    st.session_state["qa_history"].append(
                        {
                            "question": question.strip(),
                            "answer": answer.get("answer", "No answer returned."),
                            "answer_type": answer.get("answer_type", "unknown"),
                            "grounded": answer.get("grounded_in_analysis", False),
                            "evidence": answer.get("evidence", []),
                        }
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Question answering failed: {type(exc).__name__}: {exc}")

        history = st.session_state.get("qa_history", [])
        if history:
            st.divider()
            for index, item in enumerate(history, start=1):
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant"):
                    st.write(item["answer"])
                    st.caption(
                        f"Answer type: {item['answer_type']} · "
                        f"Grounded in analysis: {'Yes' if item['grounded'] else 'No'}"
                    )
                    if item.get("evidence"):
                        with st.expander(f"Evidence for answer {index}"):
                            st.json(item["evidence"])
            if st.button("Clear question history"):
                st.session_state["qa_history"] = []
                st.rerun()

    with tab_raw:
        st.subheader("Processing timings")
        timings = result.get("timings", {})
        if timings:
            st.dataframe(
                [{"Stage": key, "Seconds": value} for key, value in timings.items()],
                width="stretch",
                hide_index=True,
            )
        st.subheader("Complete JSON result")
        st.download_button(
            "Download analysis JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False, default=str),
            file_name=f"{result.get('filename', 'audio_analysis')}.json",
            mime="application/json",
        )
        st.json(result)

    limitations = safe_list(fused.get("limitations"))
    with st.expander("Model limitations"):
        if limitations:
            for limitation in limitations:
                st.write(f"- {limitation}")
        else:
            st.write("No limitation notes were supplied by the fusion module.")
elif result is not None:
    st.error("The stored analysis result has an unexpected structure. Please analyze the recording again.")