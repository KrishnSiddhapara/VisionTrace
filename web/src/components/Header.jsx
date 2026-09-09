import React from 'react';
import { Video, Cpu, Sparkles, AlertCircle } from 'lucide-react';

export default function Header({ activeTab, activeMemory, onNewAnalysisClick }) {
  const pageTitles = {
    dashboard: 'Video Intelligence Dashboard',
    analyze: 'Video Analysis & Ingestion',
    analytics: 'People & Objects Analytics',
    events: 'Events & Timeline',
    'ask-ai': 'Ask AI Video Q&A Engine',
    settings: 'Platform Settings',
  };

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between h-16 px-6 border-b border-slate-800 bg-slate-950/80 backdrop-blur-xl">
      <div className="flex items-center space-x-4">
        <h1 className="text-lg font-bold text-white tracking-tight">
          {pageTitles[activeTab] || 'VisionTrace AI'}
        </h1>

        {activeMemory?.metadata && (
          <div className="hidden md:flex items-center px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 font-mono">
            <Video className="w-3.5 h-3.5 mr-1.5 text-indigo-400" />
            <span className="truncate max-w-[200px]">{activeMemory.metadata.filename}</span>
            <span className="mx-2 text-indigo-500/50">•</span>
            <span>{activeMemory.metadata.duration_sec?.toFixed(1)}s</span>
          </div>
        )}
      </div>

      <div className="flex items-center space-x-3">
        {activeTab !== 'analyze' && (
          <button
            onClick={onNewAnalysisClick}
            className="flex items-center px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-600/20 transition-all duration-150 active:scale-95"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
            + Analyze New Video
          </button>
        )}

        <div className="hidden sm:flex items-center px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-400 font-mono">
          <Cpu className="w-3.5 h-3.5 mr-1.5 text-slate-400" />
          <span>YOLOv8 + VLM</span>
        </div>
      </div>
    </header>
  );
}
