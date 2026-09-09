import React, { useState } from 'react';
import { Settings as SettingsIcon, Cpu, Sliders, ShieldCheck, Activity, Database, Check } from 'lucide-react';

export default function SettingsPage() {
  const [model, setModel] = useState('gemini-2.5-flash');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="p-6 space-y-8 max-w-4xl mx-auto">
      <div className="space-y-1 pb-4 border-b border-slate-800">
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <SettingsIcon className="w-6 h-6 text-indigo-400" />
          Platform Settings
        </h1>
        <p className="text-xs text-slate-400">Configure AI model parameters and detection pipeline thresholds.</p>
      </div>

      <div className="space-y-6">
        {/* Model Selector Card */}
        <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            AI Model Engine
          </h3>

          <div className="space-y-3">
            <label className="text-xs font-semibold text-slate-300">Multimodal VLM Model Provider:</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:border-indigo-500 focus:outline-none"
            >
              <option value="gemini-2.5-flash">Gemini 2.5 Flash (Google DeepMind - Fast & High Accuracy)</option>
              <option value="gemini-1.5-pro">Gemini 1.5 Pro (Deep Visual Reasoning)</option>
              <option value="gpt-4o">GPT-4o (OpenAI Vision Engine)</option>
            </select>
          </div>
        </div>

        {/* Object Detection Settings */}
        <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
            Computer Vision Detection Defaults
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
              <span className="text-slate-400 font-semibold">Object Detector Model</span>
              <p className="text-sm font-bold text-white font-mono">YOLOv8 Medium (yolov8m.pt)</p>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1">
              <span className="text-slate-400 font-semibold">Entity Tracker Strategy</span>
              <p className="text-sm font-bold text-white font-mono">Spatial Bounding Box IoU Matcher</p>
            </div>
          </div>
        </div>

        {/* Backend API Health Status */}
        <div className="glass-panel p-6 rounded-3xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            System Backend Status
          </h3>

          <div className="flex items-center justify-between p-4 rounded-2xl bg-slate-900/60 border border-slate-800 text-xs">
            <div className="flex items-center space-x-3">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </span>
              <div>
                <p className="font-bold text-white">FastAPI Engine Backend API</p>
                <p className="text-slate-400 font-mono text-[11px]">http://localhost:8000/api</p>
              </div>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 font-mono font-semibold border border-emerald-500/20">
              HEALTHY
            </span>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className="flex items-center px-6 py-3 rounded-2xl text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-500 shadow-xl shadow-indigo-600/30 transition-all active:scale-95 gap-2"
          >
            {saved ? <Check className="w-4 h-4 text-emerald-400" /> : <SettingsIcon className="w-4 h-4" />}
            <span>{saved ? 'Settings Saved!' : 'Save Configuration'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
