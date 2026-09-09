import React, { useState } from 'react';
import {
  Upload,
  Video,
  Play,
  FileVideo,
  Settings,
  Sparkles,
  CheckCircle,
  Users,
  Box,
  Zap,
  Activity,
  ArrowRight,
  Clock,
  ShieldCheck,
  RotateCcw,
  MessageSquare,
  Eye,
  Layers,
  FileText
} from 'lucide-react';

import { uploadVideo, createAnalysisStream } from '../api';
import ProgressStepper from '../components/ProgressStepper';
import VideoPlayer from '../components/VideoPlayer';

export default function AnalyzePage({ activeMemory, setActiveMemory, seekTime, onNavigateTab }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Settings
  const [samplingMode, setSamplingMode] = useState('Balanced');
  const [yoloConfidence, setYoloConfidence] = useState(0.45);

  // Progress state
  const [progressState, setProgressState] = useState({
    step: 0,
    totalSteps: 8,
    progress: 0,
    stage: '',
    status: '',
    error: null,
  });

  const handleFileDrop = async (file) => {
    if (!file) return;
    
    // Clear active memory from previous video to prevent stale data display
    setActiveMemory(null);
    setSelectedFile(file);
    setUploading(true);

    try {
      const data = await uploadVideo(file);
      setUploadedData(data);
      
      // If previous memory exists for this video hash, we can inform the user
      if (data.existing_memory) {
        console.log('[FRONTEND] Found existing memory for uploaded video hash:', data.video_id);
      }
    } catch (err) {
      alert(`Upload failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const startAnalysis = (overrideHash = null) => {
    const vId = overrideHash || uploadedData?.video_id;
    if (!vId) return;

    setAnalyzing(true);
    setProgressState({
      step: 1,
      totalSteps: 8,
      progress: 5,
      stage: 'Validation & Metadata',
      status: 'Validating video & extracting metadata...',
      error: null,
    });

    createAnalysisStream(
      vId,
      samplingMode,
      yoloConfidence,
      (data) => {
        if (data.error) {
          setProgressState((prev) => ({ ...prev, error: data.error }));
          setAnalyzing(false);
          return;
        }

        setProgressState({
          step: data.step || 1,
          totalSteps: data.total_steps || 8,
          progress: data.progress || 0,
          stage: data.stage || '',
          status: data.status || '',
          error: null,
        });

        if (data.complete && data.memory) {
          setActiveMemory(data.memory);
          setAnalyzing(false);
        }
      },
      (err) => {
        setProgressState((prev) => ({ ...prev, error: 'Connection lost during streaming analysis. Backend might still be processing.' }));
        setAnalyzing(false);
      }
    );
  };

  const formatTimestamp = (secs) => {
    if (secs === undefined || secs === null || isNaN(secs)) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Accurate entity records derived from canonical registries or final_summary
  const canonicalPeople = activeMemory?.canonical_people || [];
  const finalPeople = activeMemory?.final_summary?.people || [];
  const displayPeople = finalPeople.length > 0 ? finalPeople : canonicalPeople;
  const peopleCount = displayPeople.length || activeMemory?.tracks?.filter((t) => t.object_type?.toLowerCase() === 'person').length || 0;

  const physicalObjects = activeMemory?.physical_objects || [];
  const finalObjects = activeMemory?.final_summary?.objects || [];
  const displayObjects = finalObjects.length > 0 ? finalObjects : physicalObjects;
  const objectsCount = displayObjects.length || activeMemory?.tracks?.filter((t) => t.object_type?.toLowerCase() !== 'person').length || 0;

  const eventsCount = activeMemory?.events?.length || 0;
  const keyMomentsCount = activeMemory?.sampled_frames?.length || 0;

  const finalDescription = activeMemory?.final_summary?.final_description || activeMemory?.summary?.standard || activeMemory?.summary?.quick || '';
  const narrativeBlocks = finalDescription
    ? (finalDescription.includes('\n\n')
        ? finalDescription.split('\n\n').filter(Boolean)
        : finalDescription.split('. ').filter(Boolean).map(s => s.endsWith('.') ? s : s + '.'))
    : [];

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-indigo-400" />
            Video Ingestion & AI Intelligence Engine
          </h1>
          <p className="text-xs text-slate-400">
            Upload video streams to run YOLOv8 object detection, Spatial IoU entity tracking, and VLM temporal reasoning.
          </p>
        </div>

        {activeMemory && !analyzing && (
          <button
            onClick={() => {
              setActiveMemory(null);
              setSelectedFile(null);
              setUploadedData(null);
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-bold text-white bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 shadow-lg shadow-indigo-600/30 transition-all duration-200 active:scale-95 shrink-0"
          >
            <Upload className="w-4 h-4" />
            <span>+ Upload New Video</span>
          </button>
        )}
      </div>

      {/* Analysis Running State */}
      {analyzing ? (
        <ProgressStepper
          currentStep={progressState.step}
          totalSteps={progressState.totalSteps}
          progressPercent={progressState.progress}
          currentStage={progressState.stage}
          currentLog={progressState.status}
          error={progressState.error}
        />
      ) : (
        <>
          {/* Upload Zone & Ingestion Settings (Shown when no active analysis or when re-uploading) */}
          {!activeMemory && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Upload Drop Zone */}
              <div className="lg:col-span-2 glass-panel p-8 rounded-3xl border border-dashed border-indigo-500/30 text-center flex flex-col items-center justify-center min-h-[320px] relative group hover:border-indigo-500/60 transition-colors">
                <input
                  type="file"
                  accept="video/mp4,video/mov,video/avi,video/mkv"
                  onChange={(e) => handleFileDrop(e.target.files?.[0])}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
                />

                <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <FileVideo className="w-8 h-8" />
                </div>

                <h3 className="text-lg font-bold text-white tracking-tight">
                  {selectedFile ? selectedFile.name : 'Drop your video here'}
                </h3>
                <p className="text-xs text-slate-400 mt-1 mb-4">
                  {selectedFile ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB` : 'or click to browse from device (MP4, MOV, AVI, MKV)'}
                </p>

                <div className="flex items-center space-x-2 text-[11px] font-mono text-slate-500 bg-slate-900/60 px-3 py-1.5 rounded-xl border border-slate-800">
                  <span>MP4 • AVI • MOV • MKV (Up to 200MB)</span>
                </div>

                {uploading && (
                  <p className="text-xs font-mono text-indigo-400 animate-pulse mt-4">
                    Uploading video file and extracting metadata...
                  </p>
                )}
              </div>

              {/* Ingestion Settings Card */}
              <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-5 flex flex-col justify-between">
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Settings className="w-4 h-4 text-indigo-400" />
                    Pipeline Settings
                  </h3>

                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-slate-300">Sampling Profile:</label>
                    <div className="grid grid-cols-3 gap-2">
                      {['Balanced', 'Fast', 'Deep Analysis'].map((mode) => (
                        <button
                          key={mode}
                          onClick={() => setSamplingMode(mode)}
                          className={`py-2 px-2 rounded-xl text-xs font-semibold transition-all ${
                            samplingMode === mode
                              ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                              : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {mode}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-slate-300">YOLO Confidence:</span>
                      <span className="font-mono text-indigo-400">{yoloConfidence.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={0.05}
                      max={0.95}
                      step={0.05}
                      value={yoloConfidence}
                      onChange={(e) => setYoloConfidence(parseFloat(e.target.value))}
                      className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />
                  </div>

                  {uploadedData && (
                    <div className="p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-xs space-y-2 font-mono">
                      <div className="flex justify-between text-indigo-300">
                        <span>Duration:</span>
                        <span className="font-bold text-white">{uploadedData.metadata?.duration_sec?.toFixed(1)}s</span>
                      </div>
                      <div className="flex justify-between text-indigo-300">
                        <span>Resolution:</span>
                        <span className="font-bold text-white">{uploadedData.metadata?.resolution_str}</span>
                      </div>
                      <div className="flex justify-between text-indigo-300">
                        <span>FPS / Frames:</span>
                        <span className="font-bold text-white">{uploadedData.metadata?.fps} FPS ({uploadedData.metadata?.frame_count} frames)</span>
                      </div>
                    </div>
                  )}

                  {uploadedData?.existing_memory && (
                    <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs space-y-2">
                      <span className="text-emerald-400 font-bold block flex items-center gap-1.5">
                        <CheckCircle className="w-3.5 h-3.5" />
                        Existing Analysis Memory Found!
                      </span>
                      <p className="text-[11px] text-slate-400">
                        A verified pipeline result is already cached for this video hash.
                      </p>
                      <button
                        onClick={() => setActiveMemory(uploadedData.existing_memory)}
                        className="w-full py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs shadow-md transition-colors"
                      >
                        Load Cached Analysis
                      </button>
                    </div>
                  )}
                </div>

                <button
                  disabled={!uploadedData || uploading}
                  onClick={() => startAnalysis()}
                  className="w-full py-3.5 rounded-2xl text-xs font-bold text-white bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 shadow-xl shadow-indigo-600/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 active:scale-95 flex items-center justify-center gap-2 mt-4"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>{uploadedData?.existing_memory ? 'Re-run Fresh Pipeline' : 'Start Video Analysis'}</span>
                </button>
              </div>
            </div>
          )}

          {/* Active Memory Analysis Result View */}
          {activeMemory && (
            <div className="space-y-8">
              {/* Video Metadata Header */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center space-x-3">
                  <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{activeMemory.metadata?.filename}</h3>
                    <p className="text-xs text-slate-400 font-mono">
                      {activeMemory.metadata?.resolution_str} • {activeMemory.metadata?.duration_sec?.toFixed(1)}s • {activeMemory.metadata?.fps} FPS ({activeMemory.metadata?.frame_count} frames)
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => {
                      setActiveMemory(null);
                      setUploadedData(null);
                      setSelectedFile(null);
                    }}
                    className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-300 bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-colors flex items-center gap-1.5"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Analyze Another Video
                  </button>
                </div>
              </div>

              {/* Large Video Player with Dynamic Bounding Boxes */}
              <VideoPlayer
                videoUrl={`/media/uploads/${activeMemory.metadata?.filename}`}
                seekTime={seekTime}
                sampledFrames={activeMemory.sampled_frames || []}
                yoloDetections={activeMemory.yolo_detections || {}}
                tracks={activeMemory.tracks || []}
                metadata={activeMemory.metadata}
              />

              {/* Analysis Summary Metric Cards */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Analysis Overview</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="glass-panel p-4 rounded-2xl border border-slate-800 text-center space-y-1">
                    <p className="text-[11px] font-semibold text-slate-400 uppercase">People Count</p>
                    <p className="text-2xl font-black text-emerald-400 font-mono">
                      {peopleCount}
                    </p>
                    <p className="text-[10px] text-slate-500">Unique canonical people</p>
                  </div>

                  <div className="glass-panel p-4 rounded-2xl border border-slate-800 text-center space-y-1">
                    <p className="text-[11px] font-semibold text-slate-400 uppercase">Objects Tracked</p>
                    <p className="text-2xl font-black text-indigo-400 font-mono">
                      {objectsCount}
                    </p>
                    <p className="text-[10px] text-slate-500">Unique physical objects</p>
                  </div>

                  <div className="glass-panel p-4 rounded-2xl border border-slate-800 text-center space-y-1">
                    <p className="text-[11px] font-semibold text-slate-400 uppercase">Events Verified</p>
                    <p className="text-2xl font-black text-amber-400 font-mono">
                      {eventsCount}
                    </p>
                    <p className="text-[10px] text-slate-500">Multi-source verified</p>
                  </div>

                  <div className="glass-panel p-4 rounded-2xl border border-slate-800 text-center space-y-1">
                    <p className="text-[11px] font-semibold text-slate-400 uppercase">Movement Moments</p>
                    <p className="text-2xl font-black text-cyan-400 font-mono">
                      {keyMomentsCount}
                    </p>
                    <p className="text-[10px] text-slate-500">Selected keyframes</p>
                  </div>
                </div>
              </div>

              {/* SECTION 1 — OBJECTS */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <h3 className="text-base font-extrabold text-white flex items-center gap-2">
                    <Box className="w-5 h-5 text-indigo-400" />
                    📦 SECTION 1 — OBJECTS ({displayObjects.length})
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">Reconciled physical object trajectories</span>
                </div>

                {displayObjects.length === 0 ? (
                  <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-center text-xs text-slate-400">
                    No distinct physical objects detected with sufficient visual evidence.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {displayObjects.map((obj, idx) => {
                      const name = obj.name || obj.object_id || obj.canonical_name || `Object #${idx + 1}`;
                      const conf = obj.confidence !== undefined ? Math.round(obj.confidence * 100) : Math.round((obj.avg_confidence || 0.9) * 100);
                      const firstSeen = obj.first_seen_str || (typeof obj.first_seen === 'number' ? formatTimestamp(obj.first_seen) : obj.first_seen);
                      const lastSeen = obj.last_seen_str || (typeof obj.last_seen === 'number' ? formatTimestamp(obj.last_seen) : obj.last_seen);
                      const movement = obj.movement || (obj.lifecycle_events?.length ? obj.lifecycle_events.join(', ') : 'Observed in scene');
                      const trackIds = obj.track_ids?.length ? obj.track_ids.join(', #') : (obj.track_id ? `#${obj.track_id}` : '');

                      return (
                        <div key={idx} className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
                          <div className="flex items-start justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-bold font-mono">
                                📦
                              </div>
                              <div>
                                <h4 className="text-sm font-bold text-white capitalize">{name}</h4>
                                {trackIds && <span className="text-[10px] font-mono text-slate-400">Tracks: #{trackIds}</span>}
                              </div>
                            </div>
                            <span className="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-mono text-xs font-semibold">
                              {conf}%
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
                            <div>
                              <span className="text-slate-500 text-[10px] uppercase block">First Seen</span>
                              <span className="text-white font-bold">{firstSeen}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[10px] uppercase block">Last Seen</span>
                              <span className="text-white font-bold">{lastSeen}</span>
                            </div>
                          </div>

                          {obj.description && (
                            <p className="text-xs text-slate-300 line-clamp-2">{obj.description}</p>
                          )}

                          <div className="text-xs text-slate-400 space-y-1 pt-1 border-t border-slate-800/80 font-mono">
                            <div>State: <strong className="text-indigo-300">{movement}</strong></div>
                            {obj.interactions?.length > 0 && (
                              <div>Interactions: <span className="text-slate-300">{obj.interactions.join(', ')}</span></div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* SECTION 2 — PEOPLE */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <h3 className="text-base font-extrabold text-white flex items-center gap-2">
                    <Users className="w-5 h-5 text-emerald-400" />
                    👤 SECTION 2 — PEOPLE ({displayPeople.length})
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">Confirmed unique person entities</span>
                </div>

                {displayPeople.length === 0 ? (
                  <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 text-center text-xs text-slate-400">
                    No person entities detected with sufficient visual evidence.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {displayPeople.map((person, idx) => {
                      const idStr = person.temporary_id || person.person_id || `Person #${idx + 1}`;
                      const conf = person.confidence !== undefined ? Math.round(person.confidence * 100) : Math.round((person.avg_confidence || 0.9) * 100);
                      const firstSeen = person.first_seen_str || (typeof person.first_seen === 'number' ? formatTimestamp(person.first_seen) : person.first_seen);
                      const lastSeen = person.last_seen_str || (typeof person.last_seen === 'number' ? formatTimestamp(person.last_seen) : person.last_seen);
                      const acts = person.activities || [];
                      const mvts = person.movements || (person.motion_state ? [person.motion_state] : []);
                      const shortNote = person.description || (
                        acts.length > 0 ? acts[0] : (
                          mvts.length > 0 ? mvts[0] : (
                            person.motion_state === 'MOVING' ? 'Moving in visible area' : 'Stationary in scene'
                          )
                        )
                      );

                      return (
                        <div key={idx} className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
                          <div className="flex items-start justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold font-mono">
                                👤
                              </div>
                              <div>
                                <h4 className="text-sm font-bold text-white">{idStr}</h4>
                                <span className="text-[10px] font-mono text-emerald-400">CONFIRMED</span>
                              </div>
                            </div>
                            <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 font-mono text-xs font-semibold">
                              {conf}%
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
                            <div>
                              <span className="text-slate-500 text-[10px] uppercase block">First Seen</span>
                              <span className="text-white font-bold">{firstSeen}</span>
                            </div>
                            <div>
                              <span className="text-slate-500 text-[10px] uppercase block">Last Seen</span>
                              <span className="text-white font-bold">{lastSeen}</span>
                            </div>
                          </div>

                          <div className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/15 space-y-1">
                            <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider block">Shortnote Description:</span>
                            <p className="text-xs text-slate-200 font-mono leading-relaxed">
                              {shortNote}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* SECTION 3 — FINAL DESCRIPTION & NARRATIVE */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <h3 className="text-base font-extrabold text-white flex items-center gap-2">
                    <FileText className="w-5 h-5 text-cyan-400" />
                    📝 SECTION 3 — AI VIDEO FINAL DESCRIPTION
                  </h3>
                  <span className="text-xs text-slate-400 font-mono">Chronological grounded narrative</span>
                </div>

                <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-3">
                  {narrativeBlocks.length > 0 ? (
                    narrativeBlocks.map((block, bIdx) => (
                      <div
                        key={bIdx}
                        className="p-4 rounded-xl bg-slate-900/60 border-l-4 border-indigo-500 text-slate-200 text-sm leading-relaxed"
                      >
                        {block}
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-400 italic">No summary narrative available.</p>
                  )}
                </div>
              </div>

              {/* Evidence Movement Keyframe Previews */}
              {activeMemory.sampled_frames?.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Clock className="w-4 h-4 text-indigo-400" />
                    OpenCV Selected Movement Keyframes ({activeMemory.sampled_frames.length})
                  </h3>

                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                    {activeMemory.sampled_frames.slice(0, 12).map((sf, idx) => {
                      const imgUrl = sf.url || `/media/processed/${activeMemory.metadata?.video_hash}/frames/${sf.path?.split(/[/\\]/).pop()}`;
                      return (
                        <div key={idx} className="glass-panel p-2 rounded-xl border border-slate-800 space-y-1.5 group">
                          <div className="aspect-video rounded-lg overflow-hidden bg-black relative">
                            <img
                              src={imgUrl}
                              alt={sf.frame_id}
                              className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                              onError={(e) => {
                                e.target.style.display = 'none';
                              }}
                            />
                            <div className="absolute bottom-1 right-1 bg-black/70 px-1.5 py-0.5 rounded text-[10px] font-mono text-white">
                              {formatTimestamp(sf.timestamp)}
                            </div>
                          </div>
                          <p className="text-[10px] text-slate-400 truncate font-mono">{sf.selection_reason || sf.frame_id}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
