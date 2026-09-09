import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCcw, Eye, Crosshair, Tag, Maximize2 } from 'lucide-react';

export default function VideoPlayer({
  videoUrl,
  seekTime,
  sampledFrames = [],
  yoloDetections = {},
  tracks = [],
  metadata = null
}) {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoDims, setVideoDims] = useState({ width: 1280, height: 720 });

  // BBox visualization toggles
  const [showBBoxes, setShowBBoxes] = useState(true);
  const [showTrackIDs, setShowTrackIDs] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);

  // Initialize video dimensions from metadata if available
  useEffect(() => {
    if (metadata?.width && metadata?.height) {
      setVideoDims({ width: metadata.width, height: metadata.height });
    }
  }, [metadata]);

  // Handle external seek requests
  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const formatTime = (seconds) => {
    if (isNaN(seconds) || seconds === null) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Find nearest sampled frame detection for current video time (within 1.5s)
  const currentFrame = sampledFrames.find(
    (sf) => Math.abs(sf.timestamp - currentTime) < 1.5
  );

  const currentDetections = currentFrame ? (yoloDetections[currentFrame.frame_id] || []) : [];

  const frameWidth = videoDims.width || 1280;
  const frameHeight = videoDims.height || 720;

  return (
    <div className="glass-panel rounded-2xl p-4 border border-slate-800 space-y-4 shadow-2xl">
      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center group shadow-inner"
      >
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain"
          onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
          onLoadedMetadata={() => {
            const v = videoRef.current;
            if (v) {
              setDuration(v.duration || 0);
              if (v.videoWidth && v.videoHeight) {
                setVideoDims({ width: v.videoWidth, height: v.videoHeight });
              }
            }
          }}
          onEnded={() => setIsPlaying(false)}
        />

        {/* Bounding Box Visual Overlay */}
        {showBBoxes && currentDetections.length > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            {currentDetections.map((det, idx) => {
              const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];
              const isPerson = det.class_name?.toLowerCase() === 'person';
              
              // Calculate percentages using actual video/frame dimensions
              const leftPct = (x1 / frameWidth) * 100;
              const topPct = (y1 / frameHeight) * 100;
              const widthPct = ((x2 - x1) / frameWidth) * 100;
              const heightPct = ((y2 - y1) / frameHeight) * 100;

              return (
                <div
                  key={idx}
                  className={`absolute border-2 rounded ${
                    isPerson ? 'border-emerald-400 bg-emerald-500/15' : 'border-indigo-400 bg-indigo-500/15'
                  }`}
                  style={{
                    left: `${Math.max(0, Math.min(100, leftPct))}%`,
                    top: `${Math.max(0, Math.min(100, topPct))}%`,
                    width: `${Math.max(1, Math.min(100 - leftPct, widthPct))}%`,
                    height: `${Math.max(1, Math.min(100 - topPct, heightPct))}%`,
                  }}
                >
                  <div
                    className={`absolute -top-5 left-0 text-[10px] font-mono px-1.5 py-0.5 rounded text-white font-bold flex items-center gap-1 shadow-md whitespace-nowrap ${
                      isPerson ? 'bg-emerald-600' : 'bg-indigo-600'
                    }`}
                  >
                    <span>{det.class_name}</span>
                    {showTrackIDs && det.track_id && <span className="opacity-90">#{det.track_id}</span>}
                    {showConfidence && det.confidence !== undefined && (
                      <span className="opacity-90">({Math.round(det.confidence * 100)}%)</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Video Control Overlay on Hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-4">
          <div className="flex justify-between items-center">
            {currentFrame && (
              <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-black/60 text-indigo-300 border border-indigo-500/30 backdrop-blur-md">
                Sampled Keyframe: {currentFrame.frame_id} ({currentFrame.selection_reason || 'movement'})
              </span>
            )}
            <button
              onClick={() => {
                if (containerRef.current?.requestFullscreen) {
                  containerRef.current.requestFullscreen();
                } else if (videoRef.current?.requestFullscreen) {
                  videoRef.current.requestFullscreen();
                }
              }}
              className="p-2 rounded-lg bg-slate-900/80 text-white hover:bg-slate-800 backdrop-blur-md ml-auto"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={togglePlay}
              className="p-3 rounded-full bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/40 transition-transform active:scale-95"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 fill-current" />}
            </button>

            <input
              type="range"
              min={0}
              max={duration || 100}
              step={0.1}
              value={currentTime}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setCurrentTime(val);
                if (videoRef.current) videoRef.current.currentTime = val;
              }}
              className="flex-1 accent-indigo-500 h-1.5 bg-slate-700/80 rounded-lg cursor-pointer"
            />

            <span className="text-xs font-mono text-white font-medium whitespace-nowrap">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>
        </div>
      </div>

      {/* Control Toolbar & Vision Toggles */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
          <span>Vision BBox Controls:</span>
          {currentDetections.length > 0 && (
            <span className="text-emerald-400 font-bold">({currentDetections.length} detections at current time)</span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowBBoxes(!showBBoxes)}
            className={`flex items-center px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              showBBoxes
                ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5 mr-1.5" />
            Bounding Boxes
          </button>

          <button
            onClick={() => setShowTrackIDs(!showTrackIDs)}
            className={`flex items-center px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              showTrackIDs
                ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Crosshair className="w-3.5 h-3.5 mr-1.5" />
            Track IDs
          </button>

          <button
            onClick={() => setShowConfidence(!showConfidence)}
            className={`flex items-center px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              showConfidence
                ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Tag className="w-3.5 h-3.5 mr-1.5" />
            Confidence
          </button>
        </div>
      </div>
    </div>
  );
}
