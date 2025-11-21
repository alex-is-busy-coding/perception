# Video Processing Pipeline

This module extracts high-dimensional visual features from raw video recordings. Unlike simple pixel analysis, this pipeline leverages state-of-the-art computer vision models to quantify three distinct layers of non-verbal behavior: Facial Affect, Static Identity, and Body Language.

## 🏗️ Pipeline Architecture

The pipeline processes each video file through three independent extractors:

- **Facial Expression Analysis** (`expression.py`) via **Py-Feat**.

- **Identity Embedding** (`identity.py`) via **InsightFace**.

- **3D Body Pose Estimation** (`pose.py`) via **PARE**.

## 🤖 Model Choices & Rationale

### 1. Facial Expressions (Py-Feat)

#### Library

[Py-Feat (Python Facial Expression Analysis Toolbox)](https://py-feat.org/)

#### Model Stack

- **Detection:** RetinaFace (Robust face detection).

- **Landmarks:** MobileFaceNet (Precise alignment).

- **Action Units:** XGBoost classifier.

- **Emotions:** ResMaskNet.

#### Rationale

While generic emotion classifiers (happy/sad) are common, they are often culturally biased and lack nuance. We use Py-Feat because it extracts Action Units (AUs) based on the Facial Action Coding System (FACS). AUs (e.g., "Cheek Raiser", "Brow Lowerer") are the objective anatomical building blocks of expression, allowing for granular behavioral analysis beyond broad emotional categories.

### 2. Facial Identity (InsightFace)

#### Library

[InsightFace](https://github.com/deepinsight/insightface)

#### Model Pack

buffalo_l (containing ArcFace recognition model).

#### Rationale

To distinguish between who a person is and what they are doing, we need a representation of static facial structure that is invariant to expression and lighting. InsightFace generates a 512-dimensional embedding vector for each face. In person perception research, this helps control for "facial stereotypes" (e.g., judging someone based on bone structure rather than behavior).

### 3. 3D Body Pose (PARE)

#### Model

[PARE (Part Attention Regressor)](https://github.com/mkocabas/PARE)

#### Output

SMPL Body Mesh parameters (Pose & Shape).

#### Rationale

Standard 2D pose estimators (like OpenPose) fail significantly in naturalistic settings where limbs are occluded (e.g., a person sitting behind a table). PARE is designed specifically to handle occlusion. It uses an attention mechanism to predict the full 3D body mesh (SMPL), inferring the position of hidden limbs based on visible body context.

## 📂 Output Structure

All processed data is saved to `data/video/` with the following structure:

- `expression/{filename}.csv`:

    - Frame-by-frame Action Unit intensities (0-1 scale).

    - Probability scores for 7 basic emotions.

- `identity/{filename}.csv`:

    - Frame-by-frame 512-d identity embeddings.

    - Bounding box coordinates and detection confidence scores.

- `pose/{filename}/`:

    - PARE outputs (typically .pkl or .npy files containing SMPL pose parameters and camera matrices).

    - 3D Mesh objects (.obj) if configured.

## ⚙️ Technical Setup Notes

#### PARE Installation

PARE is a research codebase and is not available on standard package managers. It must be cloned manually into a `vendor/` directory and added to the `PYTHONPATH`.

- **Requirement:** You must register and download the SMPL body models to use this module.

#### Platform Specifics

- **InsightFace & Py-Feat:** These modules utilize onnxruntime.

- **macOS (M1/M2):** The pipeline automatically detects macOS and switches to CoreMLExecutionProvider or CPU to prevent crashes, as standard CUDA libraries are incompatible with Apple Silicon.

- **Linux/Windows:** Defaults to CUDAExecutionProvider for hardware acceleration.