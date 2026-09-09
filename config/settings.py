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
    # YOLO & Multi-Object Tracking Settings
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8m.pt")
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    PERSON_CONFIDENCE: float = float(os.getenv("PERSON_CONFIDENCE", "0.55"))
    PERSON_NMS_IOU: float = float(os.getenv("PERSON_NMS_IOU", "0.40"))
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.50"))
    YOLO_IMGSZ: int = int(os.getenv("YOLO_IMGSZ", "960"))
    YOLO_MAX_DET: int = int(os.getenv("YOLO_MAX_DET", "300"))
    
    # Tracking Lifecycle & Association Parameters
    TRACK_MIN_CONFIRMED_HITS: int = int(os.getenv("TRACK_MIN_CONFIRMED_HITS", "2"))
    TRACK_TENTATIVE_HITS: int = int(os.getenv("TRACK_TENTATIVE_HITS", "2"))
    TRACK_MAX_LOST_FRAMES: int = int(os.getenv("TRACK_MAX_LOST_FRAMES", "15"))
    TRACK_MAX_LOST_SECONDS: float = float(os.getenv("TRACK_MAX_LOST_SECONDS", "3.5"))
    TRACK_MERGE_MAX_DISTANCE_PX: float = float(os.getenv("TRACK_MERGE_MAX_DISTANCE_PX", "150.0"))
    TRACK_MERGE_MAX_GAP_SEC: float = float(os.getenv("TRACK_MERGE_MAX_GAP_SEC", "5.0"))
    TRACKING_SAMPLE_FPS: float = float(os.getenv("TRACKING_SAMPLE_FPS", "5.0"))
    TRACKING_VERSION: str = "v3.2_multi_signal"
    DEBUG_PERSON_TRACKING: bool = os.getenv("DEBUG_PERSON_TRACKING", "False").lower() in ("true", "1", "yes")
    
    # VLM Budget Limits by Profile Mode (Optimized for speed & precision)
    VLM_MAX_FRAMES_FAST: int = int(os.getenv("VLM_MAX_FRAMES_FAST", "6"))
    VLM_MAX_FRAMES_BALANCED: int = int(os.getenv("VLM_MAX_FRAMES_BALANCED", "10"))
    VLM_MAX_FRAMES_DEEP: int = int(os.getenv("VLM_MAX_FRAMES_DEEP", "18"))

    # Inference Resolution Optimization (downscale large 4K frames for fast OpenCV/YOLO inference)
    INFERENCE_MAX_WIDTH: int = int(os.getenv("INFERENCE_MAX_WIDTH", "1280"))
    INFERENCE_MAX_HEIGHT: int = int(os.getenv("INFERENCE_MAX_HEIGHT", "720"))

    # Performance Profiling Flag
    DEBUG_PERFORMANCE: bool = os.getenv("VISIONTRACE_DEBUG_PERFORMANCE", "False").lower() in ("true", "1", "yes")

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
    ANALYSIS_VERSION: str = "3.0-accuracy-grounded"
    VLM_PROMPT_VERSION: str = "3.0"
    DEVELOPER_MODE: bool = os.getenv("DEVELOPER_MODE", "True").lower() in ("true", "1", "yes")
    DISABLE_VIDEO_CACHE: bool = os.getenv("DISABLE_VIDEO_CACHE", "False").lower() in ("true", "1", "yes")
    VLM_MAX_WORKERS: int = int(os.getenv("VLM_MAX_WORKERS", "8"))

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for d in [self.UPLOADS_DIR, self.PROCESSED_DIR, self.OUTPUTS_DIR, self.PROMPTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
