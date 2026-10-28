# 🎧 AudioMind — Deep Learning-Based Audio Language Model

**Joint Speech and Non-Speech Audio Understanding with Evidence-Grounded Question Answering**

AudioMind is a research prototype that analyzes spoken content and environmental sounds from the same audio recording. It combines speech transcription, sound-event classification, temporal evidence fusion, and question answering in a single workflow.

The project explores how an audio system can answer questions using evidence from both speech and background sounds, rather than treating them as separate tasks.

> **Project status:** Prototype. The current implementation uses pretrained models and rule-based question answering. It does not train a foundation model from scratch.

## ✨ Key Features

- **Audio upload:** Accepts WAV, MP3, FLAC, OGG, and M4A recordings.
- **Speech transcription:** Generates a transcript with segment-level timestamps using Faster-Whisper.
- **Language selection:** Supports automatic language detection and configured options for English, Hindi, Marathi, Tamil, Japanese, and Chinese.
- **Environmental sound detection:** Uses an AudioSet-based Audio Spectrogram Transformer (AST) model to identify sound classes.
- **Temporal evidence fusion:** Combines speech segments and sound detections into a shared timeline.
- **Evidence-aware sound labels:** Groups sound candidates by strength (for example, strong, likely, or weak) to help communicate uncertainty.
- **Audio question answering:** Answers supported questions from the generated transcript and sound timeline; can indicate when available evidence is insufficient.
- **Interactive interface:** Streamlit interface with transcript, sound timeline, question-answering, and raw-analysis views.
- **Structured output:** The pipeline can produce JSON analysis results for inspection and testing.

## 🎯 Project Objectives

1. Analyze speech and non-speech sounds jointly.
2. Preserve temporal information so users can relate spoken phrases to surrounding sounds.
3. Provide answers grounded in detected audio evidence.
4. Communicate uncertainty instead of treating every model prediction as a confirmed fact.
5. Build a modular baseline that can later be evaluated and extended with diarization, emotion analysis, and stronger audio-language reasoning.

## 🧠 System Workflow

```text
Audio File Upload
       |
       v
Audio Validation and Preprocessing
       |
       +-----------------------------+
       |                             |
       v                             v
Speech Recognition             Sound-Event Detection
(Faster-Whisper)               (AudioSet-based AST)
       |                             |
       v                             v
Timestamped Transcript         Time-Window Sound Labels
       |                             |
       +--------------+--------------+
                      |
                      v
            Evidence Fusion / Timeline
                      |
                      v
          Evidence-Grounded Question Answering
                      |
                      v
        Transcript, Sound Timeline, and Answers
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Core application and processing pipeline |
| User interface | Streamlit | Interactive web application |
| Speech recognition | Faster-Whisper | Speech transcription and timestamps |
| Sound-event classification | AudioSet-based AST model | Environmental sound classification |
| Deep learning | PyTorch, Transformers | Model inference |
| Audio processing | FFmpeg, PyDub, SoundFile | Audio decoding, conversion, and handling |
| Data processing | NumPy, Pandas | Data manipulation and presentation |
| Testing | Python test modules | Pipeline, audio, fusion, sound, speech, and QA checks |

## 🏗️ Project Structure

```text
AudioMind/
├── app.py
├── requirements.txt
├── services/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── audio_pipeline.py
│   ├── fusion_service.py
│   ├── qa_service.py
│   ├── sound_service.py
│   └── speech_service.py
├── tests/
│   ├── test_audio.py
│   ├── test_fusion.py
│   ├── test_gaps.py
│   ├── test_pipeline.py
│   ├── test_qa.py
│   ├── test_sounds.py
│   └── test_speech.py
├── data/
│   └── sample_audio/
└── outputs/                 # Generated analysis results (created at runtime)
```

## 💻 Requirements

- macOS or Linux (other platforms may work with compatible dependencies)
- Python 3.12 recommended
- FFmpeg installed and available on your `PATH`
- Internet connection on the first run to download pretrained model files
- Sufficient RAM and disk space for the selected models

Check your Python and FFmpeg installations:

```bash
python3 --version
ffmpeg -version
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abhishekmghayal/AudioMind.git
cd AudioMind
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install FFmpeg

**macOS with Homebrew**

```bash
brew install ffmpeg
```

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
```

### 4. Install Python dependencies

Upgrade pip and install the repository requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If dependency installation fails, check the Python version and the package compatibility for your operating system before changing pinned dependency versions.

### 5. Start AudioMind

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

### 6. Analyze a recording

1. Upload a supported audio file.
2. Choose the speech model size and language option.
3. Select **Analyze recording**.
4. Review the transcript and timestamped speech segments.
5. Explore detected environmental sounds and their timeline.
6. Open **Ask a question** and ask about content supported by the recording.
7. Inspect the raw analysis when debugging or validating results.

## 💬 Example Questions

- What was said in the recording?
- Which environmental sounds were detected?
- Was a siren detected?
- What sounds were present around the time a particular phrase was spoken?
- What was detected at approximately 10 seconds?
- How strong is the evidence for the detected sound?
- Does the available audio evidence support a particular interpretation?

The system should distinguish direct observations from contextual interpretations. For example, an announcement combined with aircraft-like sounds may suggest an airport setting, but the audio alone does not prove the location.

## 🧪 Running Tests

Run the available test modules from the repository root with the virtual environment activated:

```bash
python -m tests.test_audio
python -m tests.test_fusion
python -m tests.test_sounds
python -m tests.test_speech
python -m tests.test_gaps
python -m tests.test_qa
```

The question-answering test may depend on a previously generated analysis file. Check the test module and generate its required fixture before running it.

To run the pipeline on a sample recording:

```bash
python -m tests.test_pipeline data/sample_audio/<sample-file>
```

Replace `<sample-file>` with the actual filename present in `data/sample_audio/`. Generated results may be written to `outputs/`, depending on the test or pipeline configuration.

## 📊 Current Implementation and Limitations

### Implemented in the current prototype

- Audio file validation and conversion
- Speech transcription with timestamps and language detection
- Window-based environmental sound classification
- Fusion of speech and sound evidence into a timeline
- Rule-based question answering over available evidence
- Streamlit interface for reviewing results
- Test modules for key processing components

### Known limitations

- **Pretrained models:** The project currently uses pretrained models; it does not train or fine-tune an audio foundation model.
- **Question-answering scope:** The current QA component is rule-based rather than a fully generative LLM reasoning system.
- **Language coverage:** The interface exposes several language choices, but the QA logic and evaluation are not yet multilingual end-to-end.
- **Sound timing:** Sound detections use time windows and do not necessarily represent exact event start and end times.
- **Prediction confidence:** Model scores and strength labels should not be interpreted as calibrated probabilities unless calibration is separately evaluated.
- **Overlapping audio:** Loud background noise or overlapping speakers can reduce transcription accuracy, and quiet sounds may be missed.
- **Speaker diarization:** Speaker diarization is not yet implemented in the current pipeline.
- **Formal evaluation:** Comprehensive evaluation using labelled data and metrics such as word error rate (WER), event-level precision/recall/F1, and QA accuracy remains future work.

## 🛣️ Future Enhancements

- Add speaker diarization and speaker-level transcript alignment.
- Add vocal emotion and paralinguistic analysis.
- Evaluate multilingual performance, including Hindi and Marathi.
- Improve sound detection using text-audio similarity models such as CLAP.
- Add LLM-based question answering with explicit evidence references.
- Explore learned fusion of speech and environmental-audio representations.
- Add confidence calibration and timestamp-level evidence display.
- Build a labelled evaluation dataset and report WER, event precision/recall/F1, and question-answering accuracy.
- Optimize inference speed, memory usage, and handling of longer recordings.

These are planned extensions, not claims about features already implemented.

## 🔬 Research Foundation

The project is related to research in:

- Automatic speech recognition
- Environmental sound classification
- Audio-language models
- Joint speech and non-speech understanding
- Audio question answering and reasoning
- Evidence grounding and temporal alignment

Useful starting points:

- Radford et al., **“Robust Speech Recognition via Large-Scale Weak Supervision”** — https://arxiv.org/abs/2212.04356
- Gong et al., **“Joint Audio and Speech Understanding”** — https://arxiv.org/abs/2309.14405
- Deshmukh et al., **“Pengi: An Audio Language Model for Audio Tasks”** — https://arxiv.org/abs/2305.11834
- **“Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models”** — https://arxiv.org/abs/2311.07919
- **“GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities”** — https://aclanthology.org/2024.emnlp-main.361/
- **PyTorch documentation** — https://pytorch.org/docs/
- **Hugging Face Transformers documentation** — https://huggingface.co/docs/transformers/
- **FFmpeg documentation** — https://ffmpeg.org/documentation.html
- **Streamlit documentation** — https://docs.streamlit.io/

## 🔐 Privacy and Data Handling

Audio recordings may contain personal or sensitive information. Only upload recordings you have permission to process. Review the implementation and the behavior of its dependencies before using the system with sensitive data. Model downloads require internet access on first use; this README does not claim that every part of the application is fully offline.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Keep changes modular and add or update tests.
4. Run relevant tests before submitting.
5. Open a pull request describing the change and its validation.

For bug reports, include the operating system, Python version, command used, error message, and a small reproducible example. Do not upload private recordings, API keys, or other sensitive data in an issue.

## 📄 License

No license information was verified for this repository when this README was prepared. Add a `LICENSE` file and update this section before redistributing the project or accepting contributions.

## 👤 Author

**Abhishek Ghayal**

- GitHub: [@abhishekmghayal](https://github.com/abhishekmghayal)
- Repository: [AudioMind](https://github.com/abhishekmghayal/AudioMind)

---

<p align="center">
  <b>AudioMind</b><br>
  Bringing speech, environmental sounds, and contextual questions into one evidence-aware audio analysis workflow.
</p>
'''
