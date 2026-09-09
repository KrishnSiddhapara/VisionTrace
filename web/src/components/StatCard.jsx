import React from 'react';

export default function StatCard({ title, value, subtitle, icon: Icon, badgeText, badgeColor = 'indigo' }) {
  const badgeClasses = {
    indigo: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    cyan: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  };

  return (
    <div className="glass-panel p-5 rounded-2xl glass-panel-hover flex flex-col justify-between relative overflow-hidden group">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</p>
          <h3 className="text-3xl font-extrabold text-white mt-2 tracking-tight">{value}</h3>
        </div>

        {Icon && (
          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-indigo-400 group-hover:border-indigo-500/40 group-hover:text-indigo-300 transition-colors">
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between pt-3 border-t border-slate-800/60 text-xs">
        <span className="text-slate-400 font-medium truncate">{subtitle}</span>
        {badgeText && (
          <span className={`px-2 py-0.5 rounded font-mono font-semibold text-[11px] border ${badgeClasses[badgeColor] || badgeClasses.indigo}`}>
            {badgeText}
          </span>
        )}
      </div>
    </div>
  );
}
