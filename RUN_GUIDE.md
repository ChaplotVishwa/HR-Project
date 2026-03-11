# Aarvi AI - Run Guide

Follow these steps to set up and run the Aarvi Multi-Project AI Server on Windows with GPU support.

## Prerequisites

- **Python 3.10** (Recommended)
- **Virtual Environment**
- **NVIDIA GPU** (For PaddleOCR and Ranking acceleration)

## Setup Instructions

1.  **Activate your Virtual Environment**:
    ```powershell
    a:\vishu\hr\venv\Scripts\activate
    ```

2.  **Install Stable GPU Versions**:
    Run these commands one by one to ensure no library conflicts:
    ```powershell
    # Install Torch with CUDA 12.1 support
    pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

    # Install PaddlePaddle GPU
    pip install paddlepaddle-gpu==2.6.2

    # Install PaddleOCR (Stable version for 2.6.2)
    pip install paddleocr==2.8.1

    # Install other required OCR dependencies
    pip install imageio imgaug beautifulsoup4 scikit-image
    ```

3.  **Install General Requirements**:
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Download Spacy Model**:
    ```powershell
    python -m spacy download en_core_web_sm
    ```

## Configuration

Set the following environment variables in your terminal before running:

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
$env:HF_TOKEN="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"
```

## Running the Server

Run the following command from the project root (`a:\vishu\hr`):

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`.

## Accessing the UI

Open your browser and navigate to:
**[http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)**

---

### Key Features:
- **Unified Recruiter**: Upload multiple resumes + Job Description for full pipeline processing.
- **CV Extract**: Structured data extraction with GPU-accelerated OCR fallback.
- **Resume Ranking**: Hybrid ranking using embeddings and cross-encoders (CUDA prioritized).
- **Sentiment Analysis**: Automated HR suggestions.
