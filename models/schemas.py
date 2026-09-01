from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ValidationResult(BaseModel):
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


class VideoMetadata(BaseModel):
    filename: str
    filepath: str
    file_size_mb: float
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str
    video_hash: str
    resolution_str: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.resolution_str and self.width and self.height:
            self.resolution_str = f"{self.width}x{self.height}"


class Scene(BaseModel):
    scene_id: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    keyframe_paths: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    people_count: int = 0
    objects: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)


class SampledFrame(BaseModel):
    frame_id: str
    timestamp: float
    frame_index: int
    path: str
    scene_id: int
    sampling_reason: str = "uniform"  # 'scene_boundary', 'motion', 'uniform', etc.


class YOLODetection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[str] = None


class PersonObservation(BaseModel):
    temporary_id: Optional[str] = None
    description: str
    activity: Optional[str] = None
    location: Optional[str] = None
    confidence: Optional[float] = 1.0


class ObjectObservation(BaseModel):
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    confidence: Optional[float] = 1.0


class FrameObservation(BaseModel):
    frame_id: str
    timestamp: float
    scene_id: int
    environment: str = "Unknown"
    people: List[PersonObservation] = Field(default_factory=list)
    objects: List[ObjectObservation] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    visible_text: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class TrackedObject(BaseModel):
    track_id: str
    object_type: str
    first_seen: float
    last_seen: float
    positions: List[Dict[str, Any]] = Field(default_factory=list)  # [{'timestamp': 1.2, 'bbox': [...]}]
    activities: List[str] = Field(default_factory=list)
    interactions: List[str] = Field(default_factory=list)
    lifecycle_events: List[str] = Field(default_factory=list)  # e.g., ["appeared", "picked_up", "moved"]


class VideoEvent(BaseModel):
    event_id: str
    start_time: float
    end_time: float
    event_type: str  # MOVEMENT, OBJECT, PERSON, SCENE
    subject: Optional[str] = None
    object: Optional[str] = None
    description: str
    confidence: float = 0.9
    evidence_frames: List[str] = Field(default_factory=list)


class TemporalInsight(BaseModel):
    title: str
    summary: str
    key_moments: List[Dict[str, Any]] = Field(default_factory=list)
    state_changes: List[Dict[str, Any]] = Field(default_factory=list)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)


class QAResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    evidence_timestamps: List[float] = Field(default_factory=list)
    evidence_frames: List[str] = Field(default_factory=list)
    observed_facts: List[str] = Field(default_factory=list)
    inferred_facts: List[str] = Field(default_factory=list)
    unknown_aspects: List[str] = Field(default_factory=list)


class VideoMemory(BaseModel):
    video_hash: str
    metadata: VideoMetadata
    scenes: List[Scene] = Field(default_factory=list)
    sampled_frames: List[SampledFrame] = Field(default_factory=list)
    frame_observations: List[FrameObservation] = Field(default_factory=list)
    yolo_detections: Dict[str, List[YOLODetection]] = Field(default_factory=dict)  # frame_id -> detections
    tracks: List[TrackedObject] = Field(default_factory=list)
    events: List[VideoEvent] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, str] = Field(default_factory=dict)  # 'quick', 'standard', 'detailed', 'technical'
    insights: List[TemporalInsight] = Field(default_factory=list)
