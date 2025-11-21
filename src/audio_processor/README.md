# Audio Processing Pipeline

This module orchestrates the transformation of raw audio recordings into structured data suitable for machine learning. The pipeline is designed to handle real-world, naturalistic audio which often contains noise, interruptions, and overlapping speech.

## 🏗️ Pipeline Architecture

The pipeline executes three sequential stages for each audio file:

1. Speech Enhancement & Separation ([`enhancer.py`](enhancer.py))

2. Speaker Diarization & Transcription ([`diarizer.py`](diarizer.py))

3. Acoustic Feature Extraction ([`feature_extractor.py`](feature_extractor.py))

## 🤖 Model Choices & Rationale

### 1. Speech Enhancement

#### Model

`speechbrain/sepformer-wham16k-enhancement`

#### Library

[SpeechBrain](https://speechbrain.github.io/)

#### Rationale

Real-world recordings often suffer from background noise or the "cocktail party problem" (multiple speakers). We use **SepFormer**, a Transformer-based separation model trained on the WHAM! dataset. It is highly effective at isolating speech from noise and separating overlapping speakers, ensuring downstream tasks (like diarization) receive clean audio input.

### 2. Diarization & Transcription

#### Models

- **ASR:** `large-v3` (via WhisperX)

- **Diarization:** `pyannote/speaker-diarization-3.1`

#### Library

[WhisperX](https://github.com/m-bain/whisperX)

#### Rationale

Standard Whisper models struggle with precise timestamps. We use **WhisperX** because it performs forced alignment on the phoneme level, providing highly accurate word-level timestamps. This alignment is then combined with **Pyannote 3.1**, the current state-of-the-art for speaker clustering, to assign a specific speaker label (e.g., `SPEAKER_00`) to every word.

### 3. Feature Extraction

#### Feature Set

`eGeMAPSv02` (Geneva Minimalistic Acoustic Parameter Set)

#### Library

[OpenSMILE](https://audeering.github.io/opensmile-python/)

#### Rationale

Instead of learning features from raw waveforms (which requires massive datasets), we extract interpretable, hand-crafted acoustic features. **eGeMAPSv02** is the standard feature set for affective computing and paralinguistic analysis. It extracts **88 low-level descriptors** including:

- **Frequency:** Pitch (F0), Jitter, Formants.

- **Energy:** Loudness, Shimmer.

- **Spectral:** Alpha ratio, Hammarberg index (voice quality).