import hashlib
from pathlib import Path
from typing import Union
from config.settings import settings

def get_file_hash(file_path: Union[str, Path]) -> str:
    """Generate SHA256 hash of a file for caching and identification."""
    path = Path(file_path)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Return size of file in Megabytes."""
    path = Path(file_path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def validate_file_extension(filename_or_path: Union[str, Path]) -> bool:
    """Check if file extension is supported."""
    ext = Path(filename_or_path).suffix.lower()
    return ext in settings.SUPPORTED_EXTENSIONS
