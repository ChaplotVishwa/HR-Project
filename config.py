import os
from pathlib import Path

HF_TOKEN = "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"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

class Config:
    PORT: int = int(os.environ.get("PORT", 8002))
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    DEBUG: bool = os.environ.get("DEBUG", "False").lower() == "true"

    HF_TOKEN: str = os.environ.get("HF_TOKEN", HF_TOKEN)
    DEFAULT_MODEL: str = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

    FALLBACK_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    USE_GPU_OCR: bool = False  # PaddleOCR disabled - conflicts with PyTorch CUDA
    OCR_LANG: str = os.environ.get("OCR_LANG", "en")

    MAX_PDF_SIZE_MB: int = int(os.environ.get("MAX_PDF_SIZE_MB", 10))
    MAX_TEXT_LENGTH: int = int(os.environ.get("MAX_TEXT_LENGTH", 2500))
    EXTRACTION_MAX_RETRIES: int = int(os.environ.get("EXTRACTION_MAX_RETRIES", 3))

    TEMP_DIR: Path = Path(os.environ.get("TEMP_DIR", "C:/Windows/Temp"))
    UPLOAD_DIR: Path = Path(os.environ.get("UPLOAD_DIR", "./uploads"))

    @classmethod
    def validate(cls) -> bool:
        if not cls.HF_TOKEN:
            print("[WARNING] HF_TOKEN not set.")
            return False
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Path(str(cls.TEMP_DIR)).mkdir(parents=True, exist_ok=True)
        return True

    @classmethod
    def load_from_env_file(cls, env_file: str = ".env") -> None:
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value.strip('"').strip("'")
            print(f"[INFO] Loaded configuration from {env_file}")


class LLMConfig:
    """Configuration for LLM models using HuggingFace InferenceClient"""
    model_type: str = 'huggingface'
    hf_token: str = os.environ.get("HF_TOKEN", HF_TOKEN)
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    mistral_path: str = "USING_HUGGINGFACE_INFERENCE_API"
    n_ctx: int = 2048
    n_batch: int = 128
    n_gpu_layers: int = 20
    temperature: float = 0.3
    max_tokens: int = 200
    use_llm_for_top_n: int = 20


class sentimentConfig:
    HF_TOKEN = os.getenv("HF_TOKEN", HF_TOKEN)
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
