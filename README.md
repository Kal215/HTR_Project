# HTR Engine (Handwritten Text Recognition)

![HTR Engine Pipeline](https://img.shields.io/badge/Architecture-Hybrid%20CNN%20%7C%20BiLSTM-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgray?style=for-the-badge)

An Enterprise-grade, entirely on-premise pipeline designed to digitize raw cursive handwritten documents into highly accurate digital text strings.

## 🧠 Core Neural Architecture

This engine operates fully offline and executes via a sequential multi-stage Deep Learning pipeline:

1. **Vision Pre-processing:** Implements Inverse Otsu's Thresholding to isolate ink strokes from uneven paper illumination, yielding a pure binary spatial tensor.
2. **CNN Feature Extraction:** A 3-layer Convolutional Neural Network detects fundamental curves and translates visual morphology into 256 distinct numerical features.
3. **BiLSTM Temporal Memory:** Bidirectional Long Short-Term Memory networks scan the spatial sequences simultaneously from left-to-right and right-to-left to deduce ambiguous connected cursive letters via context.
4. **CTC Alignment:** The Connectionist Temporal Classification (CTC) algorithm collapses repeating probabilities (e.g., `a-a-a` to `a`) and drops blank tokens.
5. **Lexical DP Optimization:** A custom Dynamic Programming matrix (Levenshtein Distance) mathematically forces raw output strings to align with a valid linguistic corpus, ensuring maximum spelling integrity.

## 🚀 Tech Stack
- **Inference Engine:** TensorFlow 2.14
- **Backend API:** FastAPI 0.111 / Uvicorn
- **Computer Vision:** OpenCV 4.10
- **Frontend UI:** Native HTML5, Tailwind CSS, Javascript (Zero Dependencies)

## 🛠️ Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kal215/HTR_Project.git
   cd HTR_Project
   ```

2. **Activate Virtual Environment:**
   Ensure you have Python 3.11+ installed.
   ```bash
   python -m venv venv_tf
   source venv_tf/bin/activate  # On Windows use: venv_tf\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn tensorflow opencv-python pillow python-multipart pyspellchecker
   ```

4. **Launch the Inference Server:**
   ```bash
   python main.py
   ```
   Navigate to `http://localhost:8000` to access the Enterprise HTR Dashboard.

## 👤 Author
**Muhammad Riskal Fadhilla**
System Architect | AI/ML Engineer
