import React from 'react';
import { CheckCircle2, Loader2, Circle, AlertTriangle } from 'lucide-react';

export default function ProgressStepper({ currentStep, totalSteps = 8, progressPercent = 0, currentStage = '', currentLog = '', error = null }) {
  const steps = [
    { id: 1, name: 'Validation & Metadata', desc: 'Video format validation & FPS metadata extraction' },
    { id: 2, name: 'Adaptive Frame Sampling', desc: 'Scanning representative change & keyframes' },
    { id: 3, name: 'VLM Analysis', desc: 'Multimodal vision language analysis' },
    { id: 4, name: 'YOLO Object Detection', desc: 'Batched spatial object bounding box detection' },
    { id: 5, name: 'Entity Tracking', desc: 'Spatial IoU trajectory entity matcher' },
    { id: 6, name: 'Candidate Event Detection', desc: 'Identifying candidate temporal events' },
    { id: 7, name: 'Event Verification', desc: 'Multi-source confidence scoring' },
    { id: 8, name: 'Temporal Reasoning & Summary', desc: 'Synthesizing grounded timeline & summary' },
  ];

  return (
    <div className="glass-panel p-6 rounded-2xl border border-indigo-500/20 max-w-2xl mx-auto my-6 shadow-2xl">
      <div className="flex items-center justify-between pb-4 mb-5 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
            </span>
            Real-Time Video Intelligence Analysis
          </h2>
          <p className="text-xs text-slate-400 mt-1">Executing multi-pass computer vision & VLM pipeline</p>
        </div>

        <div className="text-right">
          <span className="text-2xl font-black text-indigo-400 font-mono">{progressPercent}%</span>
          <p className="text-[11px] text-slate-400 font-medium uppercase tracking-wider">Overall Progress</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden mb-6 border border-slate-800">
        <div
          className="bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-400 h-full rounded-full transition-all duration-300 shadow-lg shadow-indigo-500/50"
          style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }}
        />
      </div>

      {error ? (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold">Analysis Failed</h4>
            <p className="text-xs mt-1 text-red-200">{error}</p>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {steps.map((step) => {
            let statusIcon = null;
            let statusClasses = 'text-slate-500 border-slate-800 bg-slate-900/40';
            let labelClasses = 'text-slate-400';

            if (step.id < currentStep) {
              statusIcon = <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
              statusClasses = 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
              labelClasses = 'text-slate-200 font-medium';
            } else if (step.id === currentStep) {
              statusIcon = <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />;
              statusClasses = 'text-indigo-400 border-indigo-500/40 bg-indigo-500/10 shadow-lg shadow-indigo-500/20';
              labelClasses = 'text-white font-bold';
            } else {
              statusIcon = <Circle className="w-3.5 h-3.5 text-slate-600" />;
            }

            return (
              <div key={step.id} className="flex items-start space-x-3.5">
                <div className={`flex items-center justify-center w-7 h-7 rounded-full border text-xs shrink-0 mt-0.5 ${statusClasses}`}>
                  {statusIcon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className={`text-sm tracking-tight ${labelClasses}`}>
                      {step.name}
                    </span>
                    {step.id === currentStep && (
                      <span className="text-[11px] font-mono text-indigo-400 font-medium animate-pulse">
                        Processing...
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 truncate">{step.desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {currentLog && !error && (
        <div className="mt-5 p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono text-indigo-300 flex items-center gap-2">
          <span className="text-slate-500">&gt;</span>
          <span className="truncate">{currentLog}</span>
        </div>
      )}
    </div>
  );
}
