import math

def seconds_to_timestamp(seconds: float, include_hours: bool = False) -> str:
    """
    Convert seconds to timestamp string.
    Example: 83.42 -> '01:23.42' or '00:01:23.42'
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp string ('MM:SS.ss' or 'HH:MM:SS.ss') to seconds.
    Example: '01:23.42' -> 83.42
    """
    parts = timestamp.strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = float(parts[0]), float(parts[1])
            return m * 60 + s
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return 0.0
    except ValueError:
        return 0.0


def frame_to_timestamp(frame_index: int, fps: float, include_hours: bool = False) -> str:
    """Convert a 0-indexed or 1-indexed frame number to timestamp string given fps."""
    if fps <= 0:
        return "00:00.00"
    seconds = frame_index / fps
    return seconds_to_timestamp(seconds, include_hours=include_hours)


def timestamp_to_frame(timestamp: str, fps: float) -> int:
    """Convert timestamp string to closest frame index given fps."""
    if fps <= 0:
        return 0
    seconds = timestamp_to_seconds(timestamp)
    return int(round(seconds * fps))
