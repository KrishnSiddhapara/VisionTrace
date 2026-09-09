import React, { useState } from 'react';
import { Calendar, Zap, Filter, Clock, Play, CheckCircle2, AlertCircle } from 'lucide-react';

export default function EventsTimelinePage({ activeMemory, onSeekToTimestamp }) {
  const [filterType, setFilterType] = useState('ALL');

  if (!activeMemory) {
    return (
      <div className="p-8 text-center glass-panel rounded-3xl max-w-xl mx-auto my-12 border border-slate-800 space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto">
          <Calendar className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-white tracking-tight">No Video Memory Loaded</h3>
        <p className="text-xs text-slate-400">
          Upload and analyze a video first to explore verified timeline events, scene changes, and timestamp seek links.
        </p>
      </div>
    );
  }

  const events = activeMemory.events || [];
  const duration = activeMemory.metadata?.duration_sec || 60;

  const filteredEvents = events.filter((e) => {
    if (filterType === 'ALL') return true;
    return e.event_type?.toUpperCase() === filterType;
  });

  const formatTimestamp = (secs) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight">Events & Temporal Timeline</h1>
          <p className="text-xs text-slate-400">
            Multi-source verified temporal event sequence and interactive video timeline axis.
          </p>
        </div>

        {/* Filter Toolbar */}
        <div className="flex items-center space-x-2 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800 overflow-x-auto">
          {['ALL', 'MOVEMENT', 'INTERACTION', 'OBJECT', 'PERSON', 'SCENE'].map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold uppercase tracking-wider transition-all whitespace-nowrap ${
                filterType === type
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Visual Timeline Axis */}
      <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4">
        <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Clock className="w-4 h-4 text-indigo-400" />
          Interactive Video Timeline Axis
        </h3>

        <div className="relative pt-6 pb-4">
          <div className="h-2 bg-slate-900 rounded-full border border-slate-800 relative">
            {events.map((evt, idx) => {
              const posPercent = Math.min(100, Math.max(0, (evt.start_time / duration) * 100));
              return (
                <div
                  key={evt.event_id || idx}
                  onClick={() => onSeekToTimestamp(evt.start_time)}
                  className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-indigo-500 border-2 border-slate-950 shadow-lg shadow-indigo-500/50 cursor-pointer hover:scale-150 transition-transform group"
                  style={{ left: `${posPercent}%` }}
                  title={`${formatTimestamp(evt.start_time)} - ${evt.description}`}
                >
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-6 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-[10px] font-mono px-2 py-1 rounded shadow-xl whitespace-nowrap border border-slate-800 z-10 pointer-events-none">
                    {formatTimestamp(evt.start_time)} • {evt.event_type}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-between text-[11px] font-mono text-slate-500 mt-3">
            <span>00:00</span>
            <span>{formatTimestamp(duration / 2)}</span>
            <span>{formatTimestamp(duration)}</span>
          </div>
        </div>
      </div>

      {/* Event Cards List */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
          Detected Verified Events ({filteredEvents.length})
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredEvents.map((evt, idx) => {
            const level = evt.evidence_level || 'CONFIRMED';
            const badgeColor =
              level === 'CONFIRMED'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : level === 'PROBABLE'
                ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/20';

            return (
              <div
                key={evt.event_id || idx}
                onClick={() => onSeekToTimestamp(evt.start_time)}
                className="glass-panel p-5 rounded-2xl glass-panel-hover cursor-pointer border border-slate-800 space-y-3 group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 font-mono text-xs text-indigo-400 font-bold">
                    <Clock className="w-3.5 h-3.5" />
                    <span>[{formatTimestamp(evt.start_time)} - {formatTimestamp(evt.end_time)}]</span>
                  </div>

                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${badgeColor}`}>
                    {level}
                  </span>
                </div>

                <p className="text-sm font-bold text-white group-hover:text-indigo-300 transition-colors">
                  {evt.description}
                </p>

                <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80 font-mono">
                  <span>Type: <strong className="text-slate-200">{evt.event_type}</strong></span>
                  <span className="text-indigo-400 group-hover:translate-x-1 transition-transform flex items-center gap-1 font-semibold">
                    Jump Video <Play className="w-3 h-3 fill-current" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
