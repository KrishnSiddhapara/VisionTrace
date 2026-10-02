import uuid
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List

from config.settings import settings
from video.processor import video_processor
from video.sampler import frame_sampler
from vision.frame_analyzer import frame_analyzer
from vision.object_detector import object_detector
from vision.tracker import SpatialIoUTracker
from vision.person_registry import canonical_person_registry
from vision.object_registry import physical_object_registry
from intelligence.event_detector import event_detector
from intelligence.temporal_reasoner import temporal_reasoner
from intelligence.video_memory import video_memory_manager
from models.schemas import VideoMemory, DeveloperMetrics
from utils.profiler import profiler
from utils.logger import logger


def run_pipeline_with_progress(
    video_path: Path,
    sampling_mode: str = "Balanced",
    yolo_confidence: float = 0.45,
    progress_callback: Optional[Callable[[int, int, int, str, str], None]] = None
) -> VideoMemory:
    """
    Execute 8-stage processing pipeline with optional progress callback.
    progress_callback signature: (step: int, total_steps: int, percentage: int, stage_name: str, status_text: str)
    """
    def _notify(step: int, pct: int, stage: str, msg: str):
        if progress_callback:
            try:
                progress_callback(step, 8, pct, stage, msg)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    profiler.start_pipeline()
    analysis_id = str(uuid.uuid4())
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("========================================")
    logger.info("ENGINE: NEW VIDEO ANALYSIS")
    logger.info("========================================")
    logger.info(f"Analysis ID: {analysis_id}")
    logger.info(f"Upload timestamp: {timestamp_str}")
    logger.info(f"Filename: {video_path.name}")
    logger.info(f"Sampling Mode: {sampling_mode}")
    logger.info(f"YOLO Confidence Threshold: {yolo_confidence:.2f}")
    logger.info("========================================")

    _notify(1, 10, "Validation & Metadata Extraction", "Validating video file & extracting metadata...")

    # Step 1: Validation & Metadata Extraction
    profiler.start_stage("Validation & Metadata")
    validation, metadata, scenes, _ = video_processor.process_video(video_path)
    profiler.stop_stage("Validation & Metadata", item_count=metadata.frame_count, unit="video frames")

    # Step 2: Intelligent Multi-Criteria Frame Sampler
    _notify(2, 25, "Adaptive Frame Sampling", f"Scanning & selecting keyframes ({sampling_mode} Mode)...")
    profiler.start_stage("Adaptive Frame Sampling")
    output_frames_dir = settings.PROCESSED_DIR / metadata.video_hash / "frames"
    sampled_frames = frame_sampler.sample_scene_frames(video_path, scenes, output_frames_dir, sampling_mode=sampling_mode)
    profiler.stop_stage("Adaptive Frame Sampling", item_count=len(sampled_frames), unit="selected keyframes")

    # Step 3: VLM Frame Window Analysis
    _notify(3, 40, "VLM Analysis", f"Analyzing {len(sampled_frames)} frames concurrently with VLM...")
    profiler.start_stage("VLM Analysis")
    frame_obs_map = {}
    max_workers = min(settings.VLM_MAX_WORKERS, max(1, len(sampled_frames)))

    def _analyze_window_worker(arg):
        i, sf = arg
        pf = sampled_frames[i - 1] if i > 0 else None
        nf = sampled_frames[i + 1] if i + 1 < len(sampled_frames) else None
        return sf.frame_id, frame_analyzer.analyze_frame_window(
            sampled_frame=sf,
            prev_frame=pf,
            next_frame=nf,
            video_hash=metadata.video_hash,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_analyze_window_worker, (i, sf)) for i, sf in enumerate(sampled_frames)]
        for future in concurrent.futures.as_completed(futures):
            f_id, obs = future.result()
            frame_obs_map[f_id] = obs

    frame_obs_list = sorted(list(frame_obs_map.values()), key=lambda o: o.timestamp)
    profiler.stop_stage("VLM Analysis", item_count=len(sampled_frames), unit="VLM frame queries")

    # Step 4: YOLO Object Detection
    _notify(4, 55, "YOLO Object Detection", f"Running batched YOLO detection (conf={yolo_confidence:.2f})...")
    profiler.start_stage("YOLO Detection")
    batch_paths = [sf.path for sf in sampled_frames]
    batch_dets = object_detector.detect_objects_batch(batch_paths, video_hash=metadata.video_hash, confidence_threshold=yolo_confidence)
    
    yolo_dets = {}
    for sf in sampled_frames:
        yolo_dets[sf.frame_id] = batch_dets.get(str(Path(sf.path).resolve()), [])
    profiler.stop_stage("YOLO Detection", item_count=sum(len(d) for d in yolo_dets.values()), unit="detections")

    # Step 5: Entity Tracking
    _notify(5, 65, "Entity Tracking", "Tracking entity trajectories with Spatial IoU Matcher...")
    profiler.start_stage("Entity Tracking")
    fresh_tracker = SpatialIoUTracker()
    tracks = fresh_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
    profiler.stop_stage("Entity Tracking", item_count=len(tracks), unit="tracked entities")

    # Step 6: Candidate Events
    _notify(6, 75, "Candidate Event Detection", "Identifying candidate events & verifying event windows...")
    profiler.start_stage("Candidate Event Detection")
    candidate_events = event_detector.detect_events(scenes, frame_obs_list, tracks)
    event_windows = [(e.start_time, e.end_time) for e in candidate_events if e.event_type != "SCENE"]

    if event_windows:
        sampled_frames = frame_sampler.sample_event_dense_frames(video_path, event_windows, output_frames_dir, sampled_frames)
        uncached_frames = [sf for sf in sampled_frames if sf.frame_id not in frame_obs_map]
        if uncached_frames:
            max_p2_workers = min(settings.VLM_MAX_WORKERS, max(1, len(uncached_frames)))
            def _analyze_p2_worker(sf):
                return sf.frame_id, frame_analyzer.analyze_frame_window(sf, video_hash=metadata.video_hash)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_p2_workers) as executor:
                futures = [executor.submit(_analyze_p2_worker, sf) for sf in uncached_frames]
                for future in concurrent.futures.as_completed(futures):
                    f_id, obs = future.result()
                    frame_obs_map[f_id] = obs

            p2_dets = object_detector.detect_objects_batch([sf.path for sf in uncached_frames], video_hash=metadata.video_hash, confidence_threshold=yolo_confidence)
            for sf in uncached_frames:
                yolo_dets[sf.frame_id] = p2_dets.get(str(Path(sf.path).resolve()), [])

        frame_obs_list = sorted(list(frame_obs_map.values()), key=lambda o: o.timestamp)
    profiler.stop_stage("Candidate Event Detection", item_count=len(candidate_events), unit="candidate events")

    # Step 7: Event Verification
    _notify(7, 85, "Event Verification", "Running multi-source event verification & confidence scoring...")
    profiler.start_stage("Event Verification")
    tracks = fresh_tracker.track_entities(sampled_frames, yolo_dets, frame_obs_list)
    verified_events = event_detector.detect_events(scenes, frame_obs_list, tracks)
    profiler.stop_stage("Event Verification", item_count=len(verified_events), unit="verified events")

    # Step 8: Temporal Reasoning & Final Summary Generation
    _notify(8, 95, "Temporal Reasoning", "Synthesizing grounded timeline & physical object summary...")
    profiler.start_stage("Temporal Reasoning")

    person_entities = canonical_person_registry.reconcile_person_tracks(tracks, frame_obs_list, metadata)
    phys_objects = physical_object_registry.reconcile_tracks(tracks, frame_obs_list)
    non_person_phys = [po for po in phys_objects if po.canonical_name != "person"]

    # Execute Isolated Grounded Person VLM Analysis per canonical person entity
    if person_entities and not settings.VLM_MOCK_MODE:
        try:
            from vision.person_analyzer import person_analyzer
            def _analyze_pe_worker(pe):
                obs_list = person_analyzer.analyze_canonical_person(pe, sampled_frames, metadata.video_hash)
                pe.person_observations = obs_list
                held = []
                for obs in obs_list:
                    for item in obs.objects_held:
                        o_name = item.get("object")
                        if o_name and o_name not in held:
                            held.append(o_name)
                pe.objects_held = held
                if obs_list and obs_list[0].actions:
                    act_item = obs_list[0].actions[0]
                    if act_item.object_name and act_item.action.lower() in ("holding", "carrying"):
                        pe.person_description = f"Holding {act_item.object_name}."
                    elif act_item.action and act_item.action.lower() not in ("observing", "standing", "present"):
                        pe.person_description = f"{act_item.action.capitalize()} in visible area."

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(settings.VLM_MAX_WORKERS, max(1, len(person_entities)))) as executor:
                list(executor.map(_analyze_pe_worker, person_entities))
        except Exception as pe_err:
            logger.warning(f"Person grounded VLM analysis error: {pe_err}")

    timeline = temporal_reasoner.synthesize_timeline(scenes, verified_events, tracks, frame_obs_list)
    summaries = temporal_reasoner.generate_summaries(metadata, scenes, timeline, tracks)
    final_summary = temporal_reasoner.generate_final_summary(metadata, scenes, timeline, tracks, frame_obs_list, sampled_frames)
    profiler.stop_stage("Temporal Reasoning", item_count=len(timeline), unit="timeline items")

    total_proc_time = profiler.stop_pipeline()

    raw_person_dets_cnt = sum(
        len([d for d in det_list if d.class_name.lower() in ("person", "people", "human", "man", "woman", "child", "kid", "player")])
        for det_list in yolo_dets.values()
    )
    confirmed_person_tracks_cnt = len([t for t in tracks if t.canonical_name == "person"])
    canonical_person_cnt = len(person_entities)

    # Detailed Per-Frame Developer & Validation Debug Logs
    if settings.DEVELOPER_MODE or getattr(settings, "DEBUG_PERSON_TRACKING", False):
        logger.info("--- PER-FRAME PERCEPTION DEBUG LOG ---")
        for sf in sampled_frames:
            f_dets = yolo_dets.get(sf.frame_id, [])
            raw_p = len([d for d in f_dets if d.class_name.lower() in ("person", "people", "human", "man", "woman", "child", "kid", "player")])
            raw_o = len([d for d in f_dets if d.class_name.lower() not in ("person", "people", "human", "man", "woman", "child", "kid", "player")])
            
            # Active tracks at this frame timestamp
            active_p_trks = len([t for t in tracks if t.canonical_name == "person" and any(abs(p.get("timestamp", 0) - sf.timestamp) < 0.1 for p in t.positions)])
            
            # VLM observation count if present
            v_obs = frame_obs_map.get(sf.frame_id)
            vlm_p = len(v_obs.people) if v_obs and v_obs.people else 0
            
            logger.info(
                f"[{sf.frame_id} @ {sf.timestamp:.2f}s] Raw persons: {raw_p} | Active person tracks: {active_p_trks} | "
                f"Canonical persons: {canonical_person_cnt} | Raw objects: {raw_o} | VLM reported people: {vlm_p} "
                f"(Authority: Canonical Registry)"
            )
        logger.info("--------------------------------------")

    logger.info(
        f"[PERSON COUNT CONSISTENCY CHECK] Raw Person Detections: {raw_person_dets_cnt} | "
        f"Confirmed Person Tracks: {confirmed_person_tracks_cnt} | "
        f"Canonical Person Entities: {canonical_person_cnt}"
    )

    if confirmed_person_tracks_cnt > 0 and canonical_person_cnt > int(confirmed_person_tracks_cnt * 1.5):
        logger.warning(
            f"WARNING: Potential person identity fragmentation detected. "
            f"Raw detections: {raw_person_dets_cnt}, Confirmed tracks: {confirmed_person_tracks_cnt}, Canonical people: {canonical_person_cnt}"
        )

    # Debug Frame Visualization if DEBUG_PERSON_TRACKING is enabled
    if getattr(settings, "DEBUG_PERSON_TRACKING", False):
        try:
            from utils.debug_visualizer import render_debug_tracking_frames
            debug_out_dir = settings.OUTPUTS_DIR / metadata.video_hash / "debug_tracks"
            render_debug_tracking_frames(sampled_frames, yolo_dets, tracks, person_entities, debug_out_dir)
            logger.info(f"[DEBUG VISUALIZER] Generated annotated person tracking debug frames in {debug_out_dir}")
        except Exception as dbg_err:
            logger.warning(f"[DEBUG VISUALIZER] Failed to render debug frames: {dbg_err}")

    analyzed_cnt = len([o for o in frame_obs_list if o.is_analyzed])
    skipped_cnt = len([o for o in frame_obs_list if not o.is_analyzed])
    avg_c = float(sum(e.confidence for e in verified_events) / max(1, len(verified_events)))

    metrics = DeveloperMetrics(
        total_video_frames=metadata.frame_count,
        candidate_movement_frames=len(sampled_frames) * 3,
        selected_change_frames=len(sampled_frames),
        static_frames_discarded=max(0, metadata.frame_count - len(sampled_frames)),
        frames_sampled=len(sampled_frames),
        frames_analyzed=analyzed_cnt,
        frames_skipped=skipped_cnt,
        vlm_calls=analyzed_cnt,
        vlm_retries=0,
        vlm_failures=skipped_cnt,
        yolo_confidence_threshold=round(yolo_confidence, 2),
        yolo_detections_count=sum(len(d) for d in yolo_dets.values()),
        tracked_entities_count=len(tracks),
        candidate_events_count=len(candidate_events),
        verified_events_count=len(verified_events),
        rejected_events_count=max(0, len(candidate_events) - len(verified_events)),
        average_confidence=round(avg_c, 2),
        raw_yolo_detections=sum(len(d) for d in yolo_dets.values()),
        confirmed_person_detections=confirmed_person_tracks_cnt,
        unique_people_count=canonical_person_cnt,
        unique_objects_count=len(non_person_phys),
        active_tracks_count=len([t for t in tracks if t.track_state == "CONFIRMED"]),
        lost_tracks_count=len([t for t in tracks if t.track_state == "LOST"]),
        rejected_tracks_count=len([t for t in tracks if t.track_state == "TENTATIVE"]),
    )

    memory = VideoMemory(
        video_hash=metadata.video_hash,
        metadata=metadata,
        scenes=scenes,
        sampled_frames=sampled_frames,
        frame_observations=frame_obs_list,
        yolo_detections=yolo_dets,
        tracks=tracks,
        person_entities=[pe.model_dump() for pe in person_entities],
        events=verified_events,
        timeline=timeline,
        summary=summaries,
        final_summary=final_summary,
        insights=[],
        developer_metrics=metrics,
    )

    video_memory_manager.save_memory(memory)
    _notify(8, 100, "Complete", f"Pipeline finished successfully in {total_proc_time:.1f}s!")
    return memory
