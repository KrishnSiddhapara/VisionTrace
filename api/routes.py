import asyncio
import json
import queue
import threading
from pathlib import Path
import shutil
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from models.schemas import VideoMemory, QAResponse, TrackedObject, FrameObservation, VideoMetadata
from video.processor import video_processor
from intelligence.engine import run_pipeline_with_progress
from intelligence.video_memory import video_memory_manager
from vision.object_registry import physical_object_registry
from vision.person_registry import canonical_person_registry
from qa.video_qa import video_qa_engine
from utils.file_utils import get_file_hash
from utils.logger import logger

router = APIRouter()

def _sanitize_memory_dict(memory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure relative URL paths for media frames in VideoMemory dict and attach reconciled canonical entities."""
    v_hash = memory_dict.get("metadata", {}).get("video_hash") or memory_dict.get("video_hash", "")
    # Sanitize video_hash if it has trailing analysis id
    if "_" in v_hash and len(v_hash.split("_")[0]) == 32:
        v_hash = v_hash.split("_")[0]

    sampled_frames = memory_dict.get("sampled_frames", [])
    for sf in sampled_frames:
        if sf.get("path"):
            fname = Path(sf["path"]).name
            sf["url"] = f"/media/processed/{v_hash}/frames/{fname}"
            sf["path"] = str(settings.PROCESSED_DIR / v_hash / "frames" / fname)

    # Attach canonical people & physical objects if not already present
    try:
        raw_tracks = [TrackedObject(**t) for t in memory_dict.get("tracks", [])]
        raw_obs = [FrameObservation(**o) for o in memory_dict.get("frame_observations", [])]
        meta_dict = memory_dict.get("metadata")
        meta_obj = VideoMetadata(**meta_dict) if meta_dict else None

        # Reconcile canonical people
        person_entities = canonical_person_registry.reconcile_person_tracks(raw_tracks, raw_obs, meta_obj)
        memory_dict["canonical_people"] = [pe.model_dump() for pe in person_entities]

        # Reconcile physical objects (non-person)
        phys_objects = physical_object_registry.reconcile_tracks(raw_tracks, raw_obs)
        non_person_phys = [po.model_dump() for po in phys_objects if po.canonical_name != "person"]
        memory_dict["physical_objects"] = non_person_phys
    except Exception as e:
        logger.warning(f"Error attaching canonical entities to memory dict: {e}")

    return memory_dict


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file, extract metadata, and check if prior VideoMemory exists."""
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: MP4, MOV, AVI, MKV."
        )

    save_path = settings.UPLOADS_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            chunk_size = 8 * 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video file: {str(e)}")

    # Extract video metadata
    try:
        validation, metadata, scenes, _ = video_processor.process_video(save_path)
        if not validation.is_valid:
            raise HTTPException(status_code=400, detail=validation.error_message or "Invalid video file format.")
        
        # Check if existing analysis memory exists
        existing_mem = video_memory_manager.load_memory(metadata.video_hash)
        existing_mem_dict = None
        if existing_mem:
            existing_mem_dict = _sanitize_memory_dict(existing_mem.model_dump())

        return {
            "status": "success",
            "video_id": metadata.video_hash,
            "filename": metadata.filename,
            "filepath": str(save_path),
            "metadata": metadata.model_dump(),
            "existing_memory": existing_mem_dict
        }
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video metadata: {str(e)}")


@router.get("/videos/{video_hash}/analyze/stream")
async def stream_video_analysis(
    video_hash: str,
    sampling_mode: str = Query("Balanced", description="Sampling Profile: Balanced, Fast, Deep Analysis"),
    yolo_confidence: float = Query(0.35, ge=0.05, le=0.95, description="YOLO confidence threshold")
):
    """
    Execute full 8-stage video analysis pipeline with real-time SSE progress streaming.
    """
    video_path = None
    
    # 1. Fast match: check exact hash match in uploads directory using get_file_hash
    for item in settings.UPLOADS_DIR.glob("*"):
        if item.is_file() and item.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            if item.stem == video_hash:
                video_path = item
                break
            try:
                f_hash = get_file_hash(item)
                if f_hash == video_hash or f_hash.startswith(video_hash) or video_hash.startswith(f_hash):
                    video_path = item
                    break
            except Exception:
                continue

    # 2. Fallback: check saved memory metadata or latest uploaded file
    if not video_path:
        mem = video_memory_manager.load_memory(video_hash)
        if mem and mem.metadata and mem.metadata.filepath and Path(mem.metadata.filepath).exists():
            video_path = Path(mem.metadata.filepath)

    if not video_path:
        all_uploads = [p for p in settings.UPLOADS_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
        if all_uploads:
            all_uploads.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            video_path = all_uploads[0]

    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found for analysis.")

    event_queue = queue.Queue()

    def progress_callback(step: int, total_steps: int, percentage: int, stage_name: str, status_text: str):
        payload = {
            "step": step,
            "total_steps": total_steps,
            "progress": percentage,
            "stage": stage_name,
            "status": status_text,
            "complete": False
        }
        event_queue.put(payload)

    def worker():
        try:
            memory = run_pipeline_with_progress(
                video_path=video_path,
                sampling_mode=sampling_mode,
                yolo_confidence=yolo_confidence,
                progress_callback=progress_callback
            )
            sanitized_dict = _sanitize_memory_dict(memory.model_dump())
            event_queue.put({
                "step": 8,
                "total_steps": 8,
                "progress": 100,
                "stage": "Complete",
                "status": "Pipeline finished successfully!",
                "complete": True,
                "video_hash": memory.video_hash,
                "memory": sanitized_dict
            })
        except Exception as e:
            logger.error(f"Analysis worker error: {e}", exc_info=True)
            event_queue.put({
                "error": str(e),
                "complete": True
            })

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_generator():
        while True:
            await asyncio.sleep(0.2)
            try:
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    yield {
                        "event": "message",
                        "data": json.dumps(evt, default=str)
                    }
                    if evt.get("complete"):
                        return
            except queue.Empty:
                pass
            if not thread.is_alive() and event_queue.empty():
                break

    return EventSourceResponse(event_generator())


@router.get("/videos/")
async def list_videos():
    """List all analyzed video memories stored in PROCESSED_DIR."""
    memories = []
    for mem_file in settings.PROCESSED_DIR.glob("memory_*.json"):
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            tracks = [TrackedObject(**t) for t in data.get("tracks", [])]
            frame_obs = [FrameObservation(**o) for o in data.get("frame_observations", [])]
            meta_obj = VideoMetadata(**metadata) if metadata else None
            
            # Reconcile unique canonical people
            canonical_people = canonical_person_registry.reconcile_person_tracks(tracks, frame_obs, meta_obj)
            final_people = data.get("final_summary", {}).get("people", []) if data.get("final_summary") else []
            people_cnt = len(final_people) if final_people else len(canonical_people)

            # Reconcile unique physical objects
            phys_objects = physical_object_registry.reconcile_tracks(tracks, frame_obs)
            final_objects = data.get("final_summary", {}).get("objects", []) if data.get("final_summary") else []
            non_person_phys = [po for po in phys_objects if po.canonical_name != "person"]
            objects_cnt = len(final_objects) if final_objects else len(non_person_phys)

            memories.append({
                "video_hash": data.get("video_hash"),
                "filename": metadata.get("filename", "Unknown"),
                "duration_sec": metadata.get("duration_sec", 0.0),
                "fps": metadata.get("fps", 0),
                "resolution_str": metadata.get("resolution_str", ""),
                "people_count": people_cnt,
                "objects_count": objects_cnt,
                "events_count": len(data.get("events", [])),
                "summary_overview": data.get("summary", {}).get("quick", ""),
                "file_size_mb": metadata.get("file_size_mb", 0.0)
            })
        except Exception as e:
            logger.warning(f"Error reading memory file {mem_file}: {e}")

    # Sort descending by filename
    memories.sort(key=lambda m: m.get("filename", ""), reverse=True)
    return {"videos": memories}


@router.delete("/videos/")
async def clear_all_video_history():
    """Clear all uploaded videos, processed analysis, frame caches, and reports."""
    deleted_counts = {"uploads": 0, "processed_memories": 0, "processed_dirs": 0, "cache_files": 0, "outputs": 0}
    try:
        # Clear uploads (excluding .gitkeep)
        for item in settings.UPLOADS_DIR.glob("*"):
            if item.is_file() and item.name != ".gitkeep":
                item.unlink(missing_ok=True)
                deleted_counts["uploads"] += 1

        # Clear processed files and subdirectories (excluding .gitkeep)
        for item in settings.PROCESSED_DIR.glob("*"):
            if item.name == ".gitkeep":
                continue
            if item.is_file():
                if item.name.startswith("memory_"):
                    deleted_counts["processed_memories"] += 1
                else:
                    deleted_counts["cache_files"] += 1
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                deleted_counts["processed_dirs"] += 1

        # Clear outputs (excluding .gitkeep)
        for item in settings.OUTPUTS_DIR.glob("*"):
            if item.is_file() and item.name != ".gitkeep":
                item.unlink(missing_ok=True)
                deleted_counts["outputs"] += 1

        return {
            "status": "success",
            "message": "Cleared all video upload and analysis history successfully.",
            "deleted": deleted_counts
        }
    except Exception as e:
        logger.error(f"Error clearing video history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear video history: {str(e)}")


@router.delete("/videos/{video_hash}")
async def delete_single_video_history(video_hash: str):
    """Delete a single video upload and its associated memory/frames."""
    deleted_files = []
    try:
        # 1. Delete memory file
        mem_path = settings.PROCESSED_DIR / f"memory_{video_hash}.json"
        if mem_path.exists():
            mem_path.unlink()
            deleted_files.append(mem_path.name)

        # 2. Delete processed frames directory if exists
        hash_dir = settings.PROCESSED_DIR / video_hash
        if hash_dir.exists() and hash_dir.is_dir():
            shutil.rmtree(hash_dir, ignore_errors=True)
            deleted_files.append(hash_dir.name)

        # 3. Delete frame caches matching video_hash
        for item in settings.PROCESSED_DIR.glob(f"*{video_hash}*"):
            if item.is_file():
                item.unlink(missing_ok=True)
                deleted_files.append(item.name)

        # 4. Delete uploaded file matching video_hash
        for item in settings.UPLOADS_DIR.glob("*"):
            if item.is_file() and (item.stem == video_hash or get_file_hash(item) == video_hash):
                item.unlink(missing_ok=True)
                deleted_files.append(item.name)

        return {
            "status": "success",
            "message": f"Deleted history for video hash '{video_hash}'.",
            "deleted_files": deleted_files
        }
    except Exception as e:
        logger.error(f"Error deleting video {video_hash}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete video history: {str(e)}")


@router.get("/videos/{video_hash}")
async def get_video_memory(video_hash: str):
    """Retrieve full VideoMemory schema by video_hash."""
    mem_path = settings.PROCESSED_DIR / f"memory_{video_hash}.json"
    if not mem_path.exists():
        candidates = list(settings.PROCESSED_DIR.glob(f"memory_{video_hash}*.json"))
        if candidates:
            mem_path = candidates[0]

    if not mem_path.exists():
        raise HTTPException(status_code=404, detail=f"Video memory for hash '{video_hash}' not found.")

    try:
        with open(mem_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _sanitize_memory_dict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load video memory: {str(e)}")


@router.get("/videos/{video_hash}/people")
async def get_video_people(video_hash: str):
    """Return tracked people entities for a video using canonical person registry."""
    memory_data = await get_video_memory(video_hash)
    raw_tracks = [TrackedObject(**t) for t in memory_data.get("tracks", [])]
    frame_obs = [FrameObservation(**o) for o in memory_data.get("frame_observations", [])]
    meta_dict = memory_data.get("metadata")
    meta_obj = VideoMetadata(**meta_dict) if meta_dict else None

    # Canonical person entities
    canonical_people = canonical_person_registry.reconcile_person_tracks(raw_tracks, frame_obs, meta_obj)
    final_people = memory_data.get("final_summary", {}).get("people", []) if memory_data.get("final_summary") else []
    person_tracks = [t.model_dump() for t in raw_tracks if (getattr(t, "canonical_name", "") == "person" or t.object_type.lower() == "person")]

    return {
        "count": len(final_people) if final_people else len(canonical_people),
        "canonical_people": [pe.model_dump() for pe in canonical_people],
        "people_tracks": person_tracks,
        "final_people_records": final_people
    }


@router.get("/videos/{video_hash}/objects")
async def get_video_objects(video_hash: str):
    """Return physical object tracks reconciled via PhysicalObjectRegistry."""
    memory_data = await get_video_memory(video_hash)
    raw_tracks = [TrackedObject(**t) for t in memory_data.get("tracks", [])]
    frame_obs = [FrameObservation(**o) for o in memory_data.get("frame_observations", [])]
    
    # Reconcile physical objects using existing AI registry
    phys_objects = physical_object_registry.reconcile_tracks(raw_tracks, frame_obs)
    non_person_phys = [po.model_dump() for po in phys_objects if po.canonical_name != "person"]
    
    final_objects = memory_data.get("final_summary", {}).get("objects", []) if memory_data.get("final_summary") else []
    
    return {
        "count": len(final_objects) if final_objects else len(non_person_phys),
        "physical_objects": non_person_phys,
        "raw_object_tracks": [t.model_dump() for t in raw_tracks if t.object_type.lower() != "person"],
        "final_object_records": final_objects
    }


@router.get("/videos/{video_hash}/events")
async def get_video_events(video_hash: str):
    """Return verified events and chronological timeline for a video."""
    memory_data = await get_video_memory(video_hash)
    return {
        "events": memory_data.get("events", []),
        "timeline": memory_data.get("timeline", [])
    }


@router.get("/videos/{video_hash}/movement")
async def get_video_movement(video_hash: str):
    """Return key movement frames and OpenCV motion score metrics."""
    memory_data = await get_video_memory(video_hash)
    return {
        "sampled_frames": memory_data.get("sampled_frames", []),
        "frame_observations": memory_data.get("frame_observations", [])
    }


@router.get("/videos/{video_hash}/summary")
async def get_video_summary(video_hash: str):
    """Return AI FinalSummary and structured text summaries."""
    memory_data = await get_video_memory(video_hash)
    return {
        "final_summary": memory_data.get("final_summary"),
        "text_summary": memory_data.get("summary", {}),
        "developer_metrics": memory_data.get("developer_metrics")
    }


@router.post("/videos/{video_hash}/ask")
async def ask_video_question(video_hash: str, payload: Dict[str, Any]):
    """Ask a question about an analyzed video and return grounded QAResponse."""
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")

    memory_dict = await get_video_memory(video_hash)
    try:
        memory = VideoMemory(**memory_dict)
    except Exception as e:
        logger.error(f"Failed to load VideoMemory model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse video memory schema: {str(e)}")

    try:
        response: QAResponse = video_qa_engine.answer_question(memory, question)
        return response.model_dump()
    except Exception as e:
        logger.error(f"QA engine failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate Q&A answer: {str(e)}")


@router.get("/stats/dashboard")
async def get_dashboard_stats():
    """Return aggregate statistics across all analyzed videos using canonical registries."""
    memory_files = list(settings.PROCESSED_DIR.glob("memory_*.json"))
    videos_cnt = len(memory_files)
    people_cnt = 0
    objects_cnt = 0
    events_cnt = 0

    for mem_file in memory_files:
        try:
            with open(mem_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            tracks = [TrackedObject(**t) for t in data.get("tracks", [])]
            frame_obs = [FrameObservation(**o) for o in data.get("frame_observations", [])]
            meta_dict = data.get("metadata", {})
            meta_obj = VideoMetadata(**meta_dict) if meta_dict else None
            
            # Use canonical person registry or final summary
            final_people = data.get("final_summary", {}).get("people", []) if data.get("final_summary") else []
            if final_people:
                people_cnt += len(final_people)
            else:
                people_cnt += len(canonical_person_registry.reconcile_person_tracks(tracks, frame_obs, meta_obj))
            
            # Use physical object registry or final summary
            final_objects = data.get("final_summary", {}).get("objects", []) if data.get("final_summary") else []
            if final_objects:
                objects_cnt += len(final_objects)
            else:
                phys_objects = physical_object_registry.reconcile_tracks(tracks, frame_obs)
                objects_cnt += len([po for po in phys_objects if po.canonical_name != "person"])

            events_cnt += len(data.get("events", []))
        except Exception as e:
            logger.warning(f"Error processing stats for {mem_file}: {e}")
            continue

    return {
        "videos_analyzed": videos_cnt,
        "people_detected": people_cnt,
        "objects_detected": objects_cnt,
        "events_detected": events_cnt
    }
