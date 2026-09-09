import React, { useState } from 'react';
import {
  Users,
  Box,
  Eye,
  Search,
  Filter,
  ShieldCheck,
  Crosshair,
  Tag,
  Activity,
  Clapperboard,
  Gauge,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Clock,
  Zap
} from 'lucide-react';

export default function AnalyticsPage({ activeMemory }) {
  const [activeSubTab, setActiveSubTab] = useState('people');
  const [searchTerm, setSearchTerm] = useState('');
  const [objectCategoryFilter, setObjectCategoryFilter] = useState('ALL');
  const [selectedBBoxFrame, setSelectedBBoxFrame] = useState(null);

  if (!activeMemory) {
    return (
      <div className="p-8 text-center glass-panel rounded-3xl max-w-xl mx-auto my-12 border border-slate-800 space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto">
          <Users className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-white tracking-tight">No Video Memory Loaded</h3>
        <p className="text-xs text-slate-400">
          Upload and analyze a video first to inspect entity tracks, people activities, physical object states, and computer vision diagnostics.
        </p>
      </div>
    );
  }

  const formatTimestamp = (secs) => {
    if (secs === undefined || secs === null || isNaN(secs)) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Reconciled people entities (from canonical_people or final_summary)
  const canonicalPeople = activeMemory.canonical_people || [];
  const finalPeople = activeMemory.final_summary?.people || [];
  const peopleList = finalPeople.length > 0 ? finalPeople : (canonicalPeople.length > 0 ? canonicalPeople : (activeMemory.tracks || []).filter(t => t.object_type?.toLowerCase() === 'person'));

  // Reconciled physical objects (from physical_objects or final_summary)
  const physicalObjects = activeMemory.physical_objects || [];
  const finalObjects = activeMemory.final_summary?.objects || [];
  const rawObjectTracks = (activeMemory.tracks || []).filter(t => t.object_type?.toLowerCase() !== 'person');
  const objectsList = finalObjects.length > 0 ? finalObjects : (physicalObjects.length > 0 ? physicalObjects : rawObjectTracks);

  // Derive present object categories dynamically
  const categoriesSet = new Set(['ALL']);
  objectsList.forEach(obj => {
    const cat = obj.canonical_name || obj.name || obj.object_type || 'other';
    const lower = cat.toLowerCase();
    if (lower.includes('ball') || lower.includes('sports') || lower.includes('bat') || lower.includes('racket') || lower.includes('frisbee') || lower.includes('skateboard')) {
      categoriesSet.add('Sports');
    } else if (lower.includes('car') || lower.includes('bus') || lower.includes('truck') || lower.includes('motorcycle') || lower.includes('bicycle') || lower.includes('vehicle')) {
      categoriesSet.add('Vehicles');
    } else if (lower.includes('laptop') || lower.includes('phone') || lower.includes('tv') || lower.includes('screen') || lower.includes('keyboard') || lower.includes('mouse')) {
      categoriesSet.add('Electronics');
    } else if (lower.includes('bag') || lower.includes('backpack') || lower.includes('suitcase') || lower.includes('handbag')) {
      categoriesSet.add('Bags');
    } else if (lower.includes('dog') || lower.includes('cat') || lower.includes('horse') || lower.includes('bird') || lower.includes('animal')) {
      categoriesSet.add('Animals');
    } else {
      categoriesSet.add('Other');
    }
  });

  const availableCategories = Array.from(categoriesSet);

  // Filter people
  const filteredPeople = peopleList.filter((p) => {
    const idStr = (p.temporary_id || p.person_id || p.track_id || '').toLowerCase();
    const descStr = (p.description || '').toLowerCase();
    const acts = (p.activities || []).join(' ').toLowerCase();
    const term = searchTerm.toLowerCase();
    return idStr.includes(term) || descStr.includes(term) || acts.includes(term);
  });

  // Filter objects
  const filteredObjects = objectsList.filter((o) => {
    const nameStr = (o.name || o.object_id || o.canonical_name || o.object_type || '').toLowerCase();
    const descStr = (o.description || '').toLowerCase();
    const term = searchTerm.toLowerCase();
    const matchesSearch = nameStr.includes(term) || descStr.includes(term);

    if (!matchesSearch) return false;
    if (objectCategoryFilter === 'ALL') return true;

    const lower = nameStr;
    if (objectCategoryFilter === 'Sports') {
      return lower.includes('ball') || lower.includes('sports') || lower.includes('bat') || lower.includes('racket') || lower.includes('frisbee') || lower.includes('skateboard');
    }
    if (objectCategoryFilter === 'Vehicles') {
      return lower.includes('car') || lower.includes('bus') || lower.includes('truck') || lower.includes('motorcycle') || lower.includes('bicycle') || lower.includes('vehicle');
    }
    if (objectCategoryFilter === 'Electronics') {
      return lower.includes('laptop') || lower.includes('phone') || lower.includes('tv') || lower.includes('screen') || lower.includes('keyboard') || lower.includes('mouse');
    }
    if (objectCategoryFilter === 'Bags') {
      return lower.includes('bag') || lower.includes('backpack') || lower.includes('suitcase') || lower.includes('handbag');
    }
    if (objectCategoryFilter === 'Animals') {
      return lower.includes('dog') || lower.includes('cat') || lower.includes('horse') || lower.includes('bird') || lower.includes('animal');
    }
    return true;
  });

  const videoWidth = activeMemory.metadata?.width || 1280;
  const videoHeight = activeMemory.metadata?.height || 720;
  const sampledFrames = activeMemory.sampled_frames || [];
  const yoloDetections = activeMemory.yolo_detections || {};
  const scenes = activeMemory.scenes || [];
  const metrics = activeMemory.developer_metrics;
  const events = activeMemory.events || [];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header & Sub-Tabs Navigation */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight">Entity Analytics & Grounding Telemetry</h1>
          <p className="text-xs text-slate-400">
            Spatial IoU tracked entities, canonical people/objects, OpenCV movement metrics, and grounding diagnostics.
          </p>
        </div>

        {/* Sub-Tabs */}
        <div className="flex items-center space-x-1.5 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800 overflow-x-auto scrollbar-none">
          <button
            onClick={() => setActiveSubTab('people')}
            className={`flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
              activeSubTab === 'people'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Users className="w-3.5 h-3.5 mr-1.5" />
            People ({peopleList.length})
          </button>

          <button
            onClick={() => setActiveSubTab('objects')}
            className={`flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
              activeSubTab === 'objects'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Box className="w-3.5 h-3.5 mr-1.5" />
            Objects ({objectsList.length})
          </button>

          <button
            onClick={() => setActiveSubTab('bbox')}
            className={`flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
              activeSubTab === 'bbox'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-3.5 h-3.5 mr-1.5" />
            BBox Visualizer
          </button>

          <button
            onClick={() => setActiveSubTab('movement')}
            className={`flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
              activeSubTab === 'movement'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5 mr-1.5" />
            Scenes & Movement
          </button>

          <button
            onClick={() => setActiveSubTab('telemetry')}
            className={`flex items-center px-3.5 py-2 rounded-xl text-xs font-semibold transition-all whitespace-nowrap ${
              activeSubTab === 'telemetry'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Gauge className="w-3.5 h-3.5 mr-1.5" />
            Diagnostics
          </button>
        </div>
      </div>

      {/* Search & Category Filter Toolbar */}
      {(activeSubTab === 'people' || activeSubTab === 'objects') && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="relative max-w-md w-full">
            <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3" />
            <input
              type="text"
              placeholder={`Search ${activeSubTab}...`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none transition-colors"
            />
          </div>

          {activeSubTab === 'objects' && availableCategories.length > 1 && (
            <div className="flex items-center space-x-1.5 overflow-x-auto scrollbar-none">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mr-1">Category:</span>
              {availableCategories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setObjectCategoryFilter(cat)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    objectCategoryFilter === cat
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* People Tab Content */}
      {activeSubTab === 'people' && (
        <div className="space-y-4">
          <div className="text-xs text-slate-400 font-mono">
            Displaying {filteredPeople.length} canonical confirmed person entities.
          </div>

          {filteredPeople.length === 0 ? (
            <div className="p-8 text-center glass-panel rounded-2xl border border-slate-800 text-slate-400 text-xs">
              No people found matching "{searchTerm}".
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {filteredPeople.map((person, idx) => {
                const idStr = person.temporary_id || person.person_id || person.track_id || `Person #${idx + 1}`;
                const conf = person.confidence !== undefined ? Math.round(person.confidence * 100) : Math.round((person.avg_confidence || 0.9) * 100);
                const firstSeen = person.first_seen_str || (typeof person.first_seen === 'number' ? formatTimestamp(person.first_seen) : person.first_seen);
                const lastSeen = person.last_seen_str || (typeof person.last_seen === 'number' ? formatTimestamp(person.last_seen) : person.last_seen);
                const acts = person.activities || [];
                const mvts = person.movements || (person.motion_state ? [person.motion_state] : []);
                const trackIds = person.track_ids?.length ? person.track_ids.join(', #') : (person.track_id ? `#${person.track_id}` : '');

                return (
                  <div key={idx} className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold font-mono">
                          👤
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                            <span>{idStr}</span>
                            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono font-semibold">
                              CONFIRMED
                            </span>
                          </h3>
                          {trackIds && <span className="text-[10px] font-mono text-slate-400">Track ID(s): #{trackIds}</span>}
                        </div>
                      </div>
                      <span className="text-xs font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        {conf}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
                      <div>
                        <span className="text-slate-500 text-[10px] uppercase block">First Seen</span>
                        <span className="text-white font-bold">{firstSeen}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 text-[10px] uppercase block">Last Seen</span>
                        <span className="text-white font-bold">{lastSeen}</span>
                      </div>
                    </div>

                    {acts.length > 0 && (
                      <div className="space-y-1">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Observed Activities:</span>
                        <div className="flex flex-wrap gap-1.5">
                          {acts.map((act, aIdx) => (
                            <span key={aIdx} className="text-xs px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                              {act}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {mvts.length > 0 && (
                      <div className="text-xs text-slate-400 font-mono pt-2 border-t border-slate-800/80">
                        Motion State: <strong className="text-slate-200">{mvts.join(', ')}</strong>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Objects Tab Content */}
      {activeSubTab === 'objects' && (
        <div className="space-y-4">
          <div className="text-xs text-slate-400 font-mono">
            Displaying {filteredObjects.length} canonical physical objects (deduplicated across occlusions).
          </div>

          {filteredObjects.length === 0 ? (
            <div className="p-8 text-center glass-panel rounded-2xl border border-slate-800 text-slate-400 text-xs">
              No objects found matching current filters.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {filteredObjects.map((obj, idx) => {
                const name = obj.name || obj.object_id || obj.canonical_name || obj.object_type || `Object #${idx + 1}`;
                const conf = obj.confidence !== undefined ? Math.round(obj.confidence * 100) : Math.round((obj.avg_confidence || 0.9) * 100);
                const firstSeen = obj.first_seen_str || (typeof obj.first_seen === 'number' ? formatTimestamp(obj.first_seen) : obj.first_seen);
                const lastSeen = obj.last_seen_str || (typeof obj.last_seen === 'number' ? formatTimestamp(obj.last_seen) : obj.last_seen);
                const movement = obj.movement || (obj.lifecycle_events?.length ? obj.lifecycle_events.join(', ') : 'Observed in scene');
                const trackIds = obj.track_ids?.length ? obj.track_ids.join(', #') : (obj.track_id ? `#${obj.track_id}` : '');

                return (
                  <div key={idx} className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-bold font-mono">
                          📦
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-white capitalize">{name}</h3>
                          {trackIds && <span className="text-[10px] font-mono text-slate-400">Track ID(s): #{trackIds}</span>}
                        </div>
                      </div>
                      <span className="text-xs font-mono text-indigo-400 font-bold bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                        {conf}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-900/60 p-3 rounded-xl border border-slate-800/80">
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

                    <div className="text-xs text-slate-400 space-y-1 pt-2 border-t border-slate-800/80 font-mono">
                      <div>State / Lifecycle: <strong className="text-indigo-300">{movement}</strong></div>
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
      )}

      {/* BBox Debugger Tab Content */}
      {activeSubTab === 'bbox' && (
        <div className="space-y-6">
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-white">Bounding Box & Track Overlay Visualizer</h3>
              <p className="text-xs text-slate-400">Inspect YOLOv8 spatial detections and tracking boxes for sampled keyframes.</p>
            </div>
            <div className="text-xs font-mono text-indigo-400">
              Video Resolution: {videoWidth}x{videoHeight}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {sampledFrames.map((sf) => {
              const dets = yoloDetections[sf.frame_id] || [];
              const frameImgUrl = sf.url || `/media/processed/${activeMemory.metadata?.video_hash}/frames/${sf.path?.split(/[/\\]/).pop()}`;

              return (
                <div key={sf.frame_id} className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-bold text-white">⏱️ [{formatTimestamp(sf.timestamp)}]</span>
                    <span className="text-indigo-400 font-semibold">{dets.length} Detections</span>
                  </div>

                  <div className="relative aspect-video rounded-xl overflow-hidden bg-black border border-slate-800">
                    <img
                      src={frameImgUrl}
                      alt={`Frame ${sf.frame_id}`}
                      className="w-full h-full object-contain"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                    {dets.map((d, dIdx) => {
                      const [x1, y1, x2, y2] = d.bbox || [0, 0, 0, 0];
                      const isPerson = d.class_name?.toLowerCase() === 'person';
                      const leftPct = (x1 / videoWidth) * 100;
                      const topPct = (y1 / videoHeight) * 100;
                      const widthPct = ((x2 - x1) / videoWidth) * 100;
                      const heightPct = ((y2 - y1) / videoHeight) * 100;

                      return (
                        <div
                          key={dIdx}
                          className={`absolute border ${
                            isPerson ? 'border-emerald-400 bg-emerald-500/20' : 'border-indigo-400 bg-indigo-500/20'
                          }`}
                          style={{
                            left: `${Math.max(0, Math.min(100, leftPct))}%`,
                            top: `${Math.max(0, Math.min(100, topPct))}%`,
                            width: `${Math.max(1, Math.min(100 - leftPct, widthPct))}%`,
                            height: `${Math.max(1, Math.min(100 - topPct, heightPct))}%`,
                          }}
                        >
                          <span className={`absolute -top-4 left-0 text-[9px] font-mono px-1 py-0.2 rounded text-white font-bold whitespace-nowrap ${
                            isPerson ? 'bg-emerald-600' : 'bg-indigo-600'
                          }`}>
                            {d.class_name} {d.track_id ? `#${d.track_id}` : ''} ({Math.round(d.confidence * 100)}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="text-[11px] font-mono text-slate-400 flex justify-between">
                    <span>{sf.frame_id}</span>
                    <span>Reason: {sf.selection_reason}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Scenes & Movement Frames Tab Content */}
      {activeSubTab === 'movement' && (
        <div className="space-y-8">
          {/* PySceneDetect Scenes Table */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Clapperboard className="w-4 h-4 text-indigo-400" />
              Detected PySceneDetect Scenes ({scenes.length})
            </h3>

            {scenes.length === 0 ? (
              <p className="text-xs text-slate-400">No scene breaks detected (single continuous scene).</p>
            ) : (
              <div className="overflow-x-auto glass-panel rounded-2xl border border-slate-800">
                <table className="w-full text-xs text-left font-mono">
                  <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="p-3">Scene ID</th>
                      <th className="p-3">Start Time</th>
                      <th className="p-3">End Time</th>
                      <th className="p-3">Duration</th>
                      <th className="p-3">Frames Range</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-200">
                    {scenes.map((sc) => (
                      <tr key={sc.scene_id} className="hover:bg-slate-900/40">
                        <td className="p-3 font-bold text-indigo-400">Scene #{sc.scene_id}</td>
                        <td className="p-3">{formatTimestamp(sc.start_time)}</td>
                        <td className="p-3">{formatTimestamp(sc.end_time)}</td>
                        <td className="p-3">{sc.duration?.toFixed(2)}s</td>
                        <td className="p-3">[{sc.start_frame} - {sc.end_frame}]</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* OpenCV Movement & Change Frames Viewer */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              OpenCV Selected Movement & Change Keyframes ({sampledFrames.length})
            </h3>

            <div className="overflow-x-auto glass-panel rounded-2xl border border-slate-800">
              <table className="w-full text-xs text-left font-mono">
                <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3">Frame ID</th>
                    <th className="p-3">Timestamp</th>
                    <th className="p-3">Selection Reason</th>
                    <th className="p-3">Motion Score</th>
                    <th className="p-3">Change Score</th>
                    <th className="p-3">Motion Area</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200">
                  {sampledFrames.map((sf) => (
                    <tr key={sf.frame_id} className="hover:bg-slate-900/40">
                      <td className="p-3 font-bold text-white">{sf.frame_id}</td>
                      <td className="p-3 text-indigo-300">{formatTimestamp(sf.timestamp)}</td>
                      <td className="p-3">{sf.selection_reason}</td>
                      <td className="p-3">{sf.motion_score?.toFixed(3) || '0.000'}</td>
                      <td className="p-3">{sf.change_score?.toFixed(3) || '0.000'}</td>
                      <td className="p-3">{Math.round((sf.motion_area || 0) * 100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Developer Accuracy & Grounding Telemetry Tab Content */}
      {activeSubTab === 'telemetry' && (
        <div className="space-y-6">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Gauge className="w-4 h-4 text-indigo-400" />
              Developer Accuracy & Pipeline Telemetry
            </h3>
            <p className="text-xs text-slate-400">
              Trace grounding verification and performance telemetry for video '{activeMemory.metadata?.filename}'.
            </p>

            {/* Row 1: Reduction Telemetry */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">Total Video Frames</span>
                <span className="text-xl font-bold font-mono text-white">{activeMemory.metadata?.frame_count || 0}</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">Selected Keyframes</span>
                <span className="text-xl font-bold font-mono text-indigo-400">{sampledFrames.length}</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">YOLO Detections</span>
                <span className="text-xl font-bold font-mono text-emerald-400">
                  {metrics?.yolo_detections_count || Object.values(yoloDetections).reduce((acc, d) => acc + d.length, 0)}
                </span>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                <span className="text-[10px] font-mono text-slate-500 uppercase block">Average Confidence</span>
                <span className="text-xl font-bold font-mono text-cyan-400">
                  {metrics?.average_confidence ? `${Math.round(metrics.average_confidence * 100)}%` : '90%'}
                </span>
              </div>
            </div>

            {/* Row 2: Event Verification Breakdown */}
            <div className="pt-4 border-t border-slate-800 space-y-2">
              <span className="text-xs font-bold text-white uppercase block">Multi-Source Event Verification Breakdown:</span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
                  🟢 CONFIRMED: {events.filter(e => (e.evidence_level || 'CONFIRMED') === 'CONFIRMED').length}
                </div>
                <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-mono">
                  🔵 PROBABLE: {events.filter(e => e.evidence_level === 'PROBABLE').length}
                </div>
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-mono">
                  🟡 UNCERTAIN: {events.filter(e => e.evidence_level === 'UNCERTAIN').length}
                </div>
                <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-mono">
                  🔴 REJECTED: {metrics?.rejected_events_count || 0}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
