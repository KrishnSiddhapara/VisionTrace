import React, { useEffect, useState } from 'react';
import { Video, Users, Box, Zap, Sparkles, Play, Shield, Layers, Brain, ArrowRight, Trash2 } from 'lucide-react';
import StatCard from '../components/StatCard';
import { fetchDashboardStats, fetchVideosList, deleteSingleVideo, clearAllVideos } from '../api';

export default function DashboardPage({ onAnalyzeClick, onSelectVideo }) {
  const [stats, setStats] = useState({
    videos_analyzed: 0,
    people_detected: 0,
    objects_detected: 0,
    events_detected: 0,
  });
  const [recentVideos, setRecentVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  const reloadData = async () => {
    try {
      const [statsData, videosData] = await Promise.all([
        fetchDashboardStats(),
        fetchVideosList()
      ]);
      if (statsData) setStats(statsData);
      if (videosData?.videos) setRecentVideos(videosData.videos);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reloadData();
  }, []);

  const handleDeleteVideo = async (e, hash) => {
    e.stopPropagation();
    if (window.confirm('Delete this analyzed video memory?')) {
      try {
        await deleteSingleVideo(hash);
        await reloadData();
      } catch (err) {
        alert(`Failed to delete video: ${err.message}`);
      }
    }
  };

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-3xl bg-gradient-to-r from-indigo-950/60 via-slate-900 to-slate-950 border border-indigo-500/20 shadow-2xl relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="space-y-2 relative z-10">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-semibold text-indigo-400">
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI Video Intelligence Platform</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Video Intelligence Dashboard
          </h1>
          <p className="text-slate-400 text-sm max-w-2xl leading-relaxed">
            Understand, analyze, track entities, and query your video streams with multimodal AI, Spatial IoU entity tracking, and grounded temporal reasoning.
          </p>
        </div>

        <button
          onClick={onAnalyzeClick}
          className="flex items-center justify-center px-6 py-3 rounded-2xl text-sm font-bold text-white bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 shadow-xl shadow-indigo-600/30 transition-all duration-200 active:scale-95 shrink-0 z-10 gap-2"
        >
          <Sparkles className="w-4 h-4" />
          <span>+ Analyze New Video</span>
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Videos Analyzed"
          value={loading ? "..." : stats.videos_analyzed}
          subtitle="Processed pipelines"
          icon={Video}
          badgeText="Memory DB"
          badgeColor="indigo"
        />
        <StatCard
          title="People Tracked"
          value={loading ? "..." : stats.people_detected}
          subtitle="Unique person tracks"
          icon={Users}
          badgeText="Spatial IoU"
          badgeColor="emerald"
        />
        <StatCard
          title="Objects Tracked"
          value={loading ? "..." : stats.objects_detected}
          subtitle="Unique object tracks"
          icon={Box}
          badgeText="YOLOv8 + VLM"
          badgeColor="cyan"
        />
        <StatCard
          title="Events Verified"
          value={loading ? "..." : stats.events_detected}
          subtitle="Temporal events"
          icon={Zap}
          badgeText="Confirmed"
          badgeColor="amber"
        />
      </div>

      {/* Main Section */}
      {recentVideos.length > 0 ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white tracking-tight">Analyzed Videos Directory</h2>
            <span className="text-xs text-slate-400 font-mono">{recentVideos.length} stored memories</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {recentVideos.map((vid) => (
              <div
                key={vid.video_hash}
                onClick={() => onSelectVideo(vid.video_hash)}
                className="glass-panel p-5 rounded-2xl glass-panel-hover cursor-pointer border border-slate-800 flex flex-col justify-between space-y-4 group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3 min-w-0">
                    <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                      <Video className="w-5 h-5" />
                    </div>
                    <div className="truncate">
                      <h3 className="text-sm font-bold text-white truncate group-hover:text-indigo-300 transition-colors">
                        {vid.filename}
                      </h3>
                      <p className="text-xs text-slate-400 font-mono mt-0.5">
                        {vid.duration_sec?.toFixed(1)}s • {vid.resolution_str || '1080p'}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteVideo(e, vid.video_hash)}
                    title="Delete video memory"
                    className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors shrink-0"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <p className="text-xs text-slate-300 line-clamp-2 italic bg-slate-900/60 p-2.5 rounded-xl border border-slate-800/80">
                  "{vid.summary_overview || 'Video analysis memory available.'}"
                </p>

                <div className="flex items-center justify-between pt-3 border-t border-slate-800 text-xs text-slate-400 font-mono">
                  <span>{vid.people_count} People • {vid.objects_count} Objects</span>
                  <span className="text-indigo-400 font-semibold group-hover:translate-x-1 transition-transform flex items-center gap-1">
                    Explore <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Welcome / Professional Landing State */
        <div className="glass-panel p-8 rounded-3xl border border-slate-800 text-center space-y-8">
          <div className="max-w-xl mx-auto space-y-3">
            <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8" />
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">VisionTrace AI Video Intelligence</h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              See beyond the frames. Upload your videos to execute computer vision object detection, person tracking, movement analysis, temporal event verification, and conversational AI Q&A.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <Shield className="w-5 h-5 text-indigo-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">Object Detection</h4>
              <p className="text-xs text-slate-400">Batched YOLOv8 detection with configurable confidence thresholding.</p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <Users className="w-5 h-5 text-emerald-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">Person Tracking</h4>
              <p className="text-xs text-slate-400">Spatial IoU trajectory matcher tracking unique identities across time.</p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <Zap className="w-5 h-5 text-amber-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">Event Detection</h4>
              <p className="text-xs text-slate-400">Multi-source verification pipeline filtering key movement & actions.</p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <Brain className="w-5 h-5 text-cyan-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">Ask AI Q&A</h4>
              <p className="text-xs text-slate-400">Grounded visual Q&A engine with clear fact & evidence verification.</p>
            </div>
          </div>

          <div>
            <button
              onClick={onAnalyzeClick}
              className="inline-flex items-center px-8 py-3.5 rounded-2xl text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-500 shadow-xl shadow-indigo-600/30 transition-all duration-200 active:scale-95 gap-2"
            >
              <Sparkles className="w-4 h-4" />
              Upload & Analyze Video Now
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
