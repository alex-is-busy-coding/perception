# Building Multimodal Models of Person Perception

> This research project aims to advance a more naturalistic understanding of person perception by integrating visual, acoustic, and physiological signals from real-world, in-person interactions.

## 🚀 Project Overview

People spontaneously form impressions of one another, and these impressions shape important decisions in contexts like dating, hiring, and even criminal sentencing. Existing literature, however, has predominately focused on impressions in controlled environments (e.g., from static photos).

This project will analyze dynamic time-series of video, audio, and physiological recordings of participants during social interactions at different locations in New York City. We will use deep learning and recurrent neural networks to understand how people use this rich, multimodal information to form impressions.

## 📖 Architecture & Methodology

This project is divided into two main computational modules. Please refer to their dedicated documentation for detailed information on model choices and architectural decisions.

### [Audio Processing Pipeline](src/audio_processor/README.md)

Details the 3-stage pipeline: Speech Enhancement (**SepFormer**) → Diarization (**WhisperX/Pyannote**) → Feature Extraction (**OpenSMILE**).

### [Video Processing Pipeline](src/video_processor/README.md)

Analyzes visual cues using state-of-the-art computer vision models:

- **Facial Expressions:** Uses **Py-Feat** to extract Action Units (AUs) and emotions.

- **Identity:** Uses **InsightFace** to generate facial embedding vectors.

- **Body Pose:** Uses **PARE** (Part Attention Regressor) for 3D body mesh and pose estimation.

### [Unsupervised Representation Learning](src/model_training/README.md)

Explains the 1D Convolutional Autoencoder and the "Hybrid Masked Pooling" strategy used to learn fixed-length participant embeddings from variable-length audio.

## 🔬 About the IMPACT Lab

This project is conducted at the **IMPression in ACTion (IMPACT) Lab** at Columbia University.

The lab investigates how people form impressions of others—from "dubious shortcuts" (like static images) to "meaningful signals" (like real-world interactions)—and how these impressions influence consequential behavior in areas like politics, law, and science. The lab's methodology combines computational models, naturalistic datasets, and behavioral experiments.

* **Lab Website:** [impactlab-columbia.github.io](https://impactlab-columbia.github.io)

## 🧑‍💻 Project Team

* **Principal Investigator:** [Dr. Chujun Lin](https://psychology.columbia.edu/content/chujun-lin) [📧](mailto:cl4767@columbia.edu)
* **Developer:** [Alexander Vassilev](https://github.com/alex-is-busy-coding) [📧](mailto:av3341@columbia.edu)

## ⚖️ License

The code in this project is released under the [MIT License](LICENSE).

**Note on Data:** The data used for this project (including video, audio, and physiological recordings) is **not** covered by this license. It is protected, private data governed by IRB approval and participant consent, and is not available publicly.

## 💻 Installation & Setup

This project uses **[uv](https://docs.astral.sh/uv/)** for high-speed Python environment and package management.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/alex-is-busy-coding/perception
    cd perception
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, you can install it by following the instructions on the
    [official guide](https://docs.astral.sh/uv/getting-started/installation).

3.  **Create a virtual environment:**
    This command creates a local `.venv` folder to hold all your packages.

    ```bash
    uv venv
    ```

4.  **Install project dependencies:**
    This will install all required packages from your `pyproject.toml` or `requirements.txt` file into the `.venv`.

    ```bash
    uv pip install -e .
    ```

### ☁️ Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alex-is-busy-coding/perception/blob/main/notebooks/colab_runner.ipynb)

If you do not have a local GPU, you can run this pipeline directly in the browser using [Google Colab](https://colab.research.google.com/).

  1. Click the badge above to open the notebook.

  2. Set your `Runtime` to `T4 GPU` or some other GPU of your choice (`Runtime` > `Change runtime type`).

  3. Follow the instructions in the notebook to set up the environment and process your data.

### 🔑 Pyannote Model Access (Required)
This pipeline uses [**Pyannote**](https://huggingface.co/pyannote) for speaker diarization. Because these models are gated, you must:

  1. **[Create a Hugging Face account](https://huggingface.co/join)** if you don't have one.

  2. **Accept the user conditions** on the following model pages:
      * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
      * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

  3. **Create an Access Token** (Read role) in [your Hugging Face settings](https://huggingface.co/settings/tokens).

  4. **Set the token** as an environment variable (`HF_TOKEN`) before running the pipeline (see **Configuration** below).

You're all set\! You're now ready to run the project.

## 🔨 Usage
  * **To process raw audio data:** This runs the `scripts/process.py` pipeline, which enhances audio, diarizes speakers, and extracts acoustic features.

    By default, it runs in **dev mode** (processing a limited number of files using `config/dev.config.yaml`).

    ```bash
    # Run in dev mode
    make process

    # Run in production mode (processes all files)
    APP_ENV=prod make process
    ```

    Outputs are saved to `data/`:
      - `transcripts/`: CSVs with speaker diarization and text.

      - `diarization_plots/`: Visual timelines of speaker turns (PNG).

      - `features/`: Extracted acoustic features (OpenSMILE).

      - `enhanced/`: Denoised audio files.

  * **To train the model:** Trains a convolutional autoencoder on the extracted acoustic features. Checkpoints are saved to `checkpoints/`.

    ```bash
    make train
    ```

  * **To visualize embeddings:** Generates latent embeddings from the best model checkpoint and saves them for TensorBoard visualization.

    ```bash
    make visualize
    ```

  * **To visualize embeddings:** This will start a web server, usually at `http://localhost:6006`. Open the **Projector** tab to see the 3D embedding space.

    ```bash
    make tensorboard
    ```

  * **To clean up log files:**

    ```bash
    make clean
    ```