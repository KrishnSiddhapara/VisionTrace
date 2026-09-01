import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # Directories
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "processed"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"

    # Supported formats
    SUPPORTED_EXTENSIONS: list[str] = [".mp4", ".mov", ".avi", ".mkv"]

    # Constraints & Thresholds
    MAX_VIDEO_SIZE_MB: float = float(os.getenv("MAX_VIDEO_SIZE_MB", "200"))
    MAX_VIDEO_DURATION_SEC: float = float(os.getenv("MAX_VIDEO_DURATION_SEC", "600"))
    DEFAULT_SAMPLE_INTERVAL_SEC: float = float(os.getenv("DEFAULT_SAMPLE_INTERVAL_SEC", "5"))
    MAX_VLM_FRAMES: int = int(os.getenv("MAX_VLM_FRAMES", "100"))
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.40"))
    # OpenCV Movement & Visual Change Detection Settings
    MOTION_THRESHOLD: float = float(os.getenv("MOTION_THRESHOLD", "25.0"))
    CHANGE_THRESHOLD: float = float(os.getenv("CHANGE_THRESHOLD", "0.08"))
    MIN_MOTION_AREA: float = float(os.getenv("MIN_MOTION_AREA", "500.0"))
    MIN_FRAME_GAP_SEC: float = float(os.getenv("MIN_FRAME_GAP_SEC", "0.3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    USE_OPTICAL_FLOW: bool = os.getenv("USE_OPTICAL_FLOW", "True").lower() in ("true", "1", "yes")

    # Feature Flags
    ENABLE_TRACKING: bool = os.getenv("ENABLE_TRACKING", "True").lower() in ("true", "1", "yes")
    ENABLE_ANOMALY_DETECTION: bool = os.getenv("ENABLE_ANOMALY_DETECTION", "True").lower() in ("true", "1", "yes")
    ENABLE_SEMANTIC_SEARCH: bool = os.getenv("ENABLE_SEMANTIC_SEARCH", "True").lower() in ("true", "1", "yes")

    # VLM Credentials & Accuracy Refactor Settings
    VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "gemini").lower()
    VLM_API_KEY: str = os.getenv("VLM_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    VLM_MODEL: str = os.getenv("VLM_MODEL", "gemini-2.5-flash")
    VLM_BASE_URL: str = os.getenv("VLM_BASE_URL", "")
    VLM_MOCK_MODE: bool = os.getenv("VLM_MOCK_MODE", "False").lower() in ("true", "1", "yes")
    VLM_MAX_RETRIES: int = int(os.getenv("VLM_MAX_RETRIES", "3"))
    ANALYSIS_VERSION: str = "2.0-accuracy"
    VLM_PROMPT_VERSION: str = "2.0"
    DEVELOPER_MODE: bool = os.getenv("DEVELOPER_MODE", "True").lower() in ("true", "1", "yes")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for d in [self.UPLOADS_DIR, self.PROCESSED_DIR, self.OUTPUTS_DIR, self.PROMPTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
