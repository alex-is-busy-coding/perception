# Building Multimodal Models of Person Perception

> This research project aims to advance a more naturalistic understanding of person perception by integrating visual, acoustic, and physiological signals from real-world, in-person interactions.

## 🚀 Project Overview

People spontaneously form impressions of one another, and these impressions shape important decisions in contexts like dating, hiring, and even criminal sentencing. Existing literature, however, has predominately focused on impressions in controlled environments (e.g., from static photos).

This project will analyze dynamic time-series of video, audio, and physiological recordings of participants during social interactions at different locations in New York City. We will use deep learning and recurrent neural networks to understand how people use this rich, multimodal information to form impressions.

## 🔬 About the IMPACT Lab

This project is conducted at the **IMPression in ACTion (IMPACT) Lab** at Columbia University.

The lab investigates how people form impressions of others—from "dubious shortcuts" (like static images) to "meaningful signals" (like real-world interactions)—and how these impressions influence consequential behavior in areas like politics, law, and science. The lab's methodology combines computational models, naturalistic datasets, and behavioral experiments.

* **Lab Website:** [impactlab-columbia.github.io](https://impactlab-columbia.github.io)

## 🧑‍💻 Project Team

* **Principal Investigator:** [Dr. Chujun Lin](https://psychology.columbia.edu/content/chujun-lin) [📧](mailto:cl4767@columbia.edu)
* **Developer:** Alexander Vassilev [📧](mailto:av3341@columbia.edu)

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

You're all set\! You're now ready to run the project.

## 🏃‍♀️ Usage
  * **To process raw audio data:** This runs the main.py script, which will transcribe audio, extract features, and save the final CSV to `data/features/`.

    ```bash
    make process
    ```

  * **To train the model:**

    ```bash
    make train
    ```

  * **To view results in TensorBoard:**
    This will start a web server, usually at `http://localhost:6006`.

    ```bash
    make tensorboard
    ```

  * **To clean up log files:**

    ```bash
    make clean
    ```