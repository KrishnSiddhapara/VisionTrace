import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  Send,
  Sparkles,
  Bot,
  User,
  CheckCircle2,
  AlertCircle,
  Clock,
  Loader2,
  HelpCircle,
  RotateCcw
} from 'lucide-react';
import { askVideoQuestion } from '../api';

export default function AskAIPage({ activeMemory, onSeekToTimestamp }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'ai',
      text: 'Hello! I am your VisionTrace AI video intelligence assistant. Ask me anything about the people, physical objects, movements, or timeline events in your analyzed video.',
    },
  ]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [loading, setLoading] = useState(false);

  // Invalidate QA conversation state when active video changes
  const activeHash = activeMemory?.metadata?.video_hash || activeMemory?.video_hash;
  useEffect(() => {
    setMessages([
      {
        id: 1,
        sender: 'ai',
        text: `Hello! I am ready to answer questions grounded in the video '${activeMemory?.metadata?.filename || 'analyzed video'}'. Ask me about people, objects, activities, or chronological events.`,
      },
    ]);
  }, [activeHash]);

  const suggestedQuestions = [
    'How many people are present?',
    'What objects are visible in the video?',
    'What are the people doing?',
    'What happened first in the video?',
    'Did anyone move or change position?',
    'What happened near the end of the video?',
  ];

  if (!activeMemory) {
    return (
      <div className="p-8 text-center glass-panel rounded-3xl max-w-xl mx-auto my-12 border border-slate-800 space-y-5">
        <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto">
          <MessageSquare className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-white tracking-tight">No Video Memory Loaded</h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          Analyze a video first to start asking grounded AI questions about objects, people trajectories, activities, and visual events.
        </p>
      </div>
    );
  }

  const handleSend = async (questionText) => {
    const q = questionText || inputQuestion;
    if (!q.trim() || loading || !activeHash) return;

    const userMsg = { id: Date.now(), sender: 'user', text: q };
    setMessages((prev) => [...prev, userMsg]);
    setInputQuestion('');
    setLoading(true);

    try {
      const response = await askVideoQuestion(activeHash, q);
      const aiMsg = {
        id: Date.now() + 1,
        sender: 'ai',
        text: response.answer,
        confidence: response.confidence,
        observedFacts: response.observed_facts || [],
        evidenceTimestamps: response.evidence_timestamps || [],
        unknownAspects: response.unknown_aspects || [],
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'ai',
          text: `Unable to generate an answer: ${err.response?.data?.detail || err.message || 'QA engine error'}.`,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleResetChat = () => {
    setMessages([
      {
        id: Date.now(),
        sender: 'ai',
        text: `Conversation reset. Ask me anything about '${activeMemory.metadata?.filename || 'this video'}'.`,
      },
    ]);
  };

  return (
    <div className="p-6 max-w-5xl mx-auto flex flex-col h-[calc(100vh-6rem)]">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800 shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-indigo-400" />
            Ask AI Video Intelligence Engine
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Grounded multimodal Q&A for '{activeMemory.metadata?.filename}'. Responses are strictly anchored in frame evidence.
          </p>
        </div>

        <button
          onClick={handleResetChat}
          className="p-2 rounded-xl text-slate-400 hover:text-white bg-slate-900 border border-slate-800 hover:bg-slate-800 transition-colors text-xs flex items-center gap-1.5"
          title="Reset conversation"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset Chat</span>
        </button>
      </div>

      {/* Suggested Questions Pills */}
      <div className="py-3 flex items-center space-x-2 overflow-x-auto shrink-0 scrollbar-none">
        <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider shrink-0 mr-1">Suggested:</span>
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(q)}
            className="px-3 py-1.5 rounded-xl text-xs bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:border-indigo-500/40 transition-colors whitespace-nowrap"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Messages Thread */}
      <div className="flex-1 overflow-y-auto space-y-4 py-4 pr-2">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start space-x-3 max-w-3xl ${
              msg.sender === 'user' ? 'ml-auto flex-row-reverse space-x-reverse' : ''
            }`}
          >
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 text-white shadow-lg ${
                msg.sender === 'user'
                  ? 'bg-indigo-600'
                  : 'bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400'
              }`}
            >
              {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            <div className="space-y-3 flex-1 min-w-0">
              {/* Answer Box */}
              <div
                className={`p-4 rounded-2xl text-sm leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none'
                    : msg.error
                    ? 'bg-red-500/10 border border-red-500/20 text-red-200 rounded-tl-none'
                    : 'glass-panel text-slate-100 border border-slate-800 rounded-tl-none'
                }`}
              >
                <p>{msg.text}</p>
              </div>

              {/* Grounded Evidence Card (AI Only) */}
              {msg.sender === 'ai' && (msg.observedFacts?.length > 0 || msg.evidenceTimestamps?.length > 0 || msg.unknownAspects?.length > 0) && (
                <div className="glass-panel p-4 rounded-2xl border border-indigo-500/20 text-xs space-y-3 bg-slate-950/70">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-indigo-300 uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      Grounded Evidence
                    </span>
                    {msg.confidence !== undefined && (
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-semibold">
                        {Math.round(msg.confidence * 100)}% Confidence
                      </span>
                    )}
                  </div>

                  {/* Observed Facts */}
                  {msg.observedFacts?.length > 0 && (
                    <div className="space-y-1">
                      <span className="text-slate-400 font-semibold text-[10px] uppercase">Direct Visual Observations:</span>
                      <ul className="list-disc list-inside space-y-1 text-slate-300">
                        {msg.observedFacts.map((fact, fIdx) => (
                          <li key={fIdx}>{fact}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Timestamps links */}
                  {msg.evidenceTimestamps?.length > 0 && (
                    <div className="space-y-1 pt-1">
                      <span className="text-slate-400 font-semibold text-[10px] uppercase">Evidence Timestamps:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.evidenceTimestamps.map((ts, tIdx) => (
                          <button
                            key={tIdx}
                            onClick={() => onSeekToTimestamp && onSeekToTimestamp(ts)}
                            className="px-2 py-1 rounded bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono text-[11px] flex items-center gap-1 transition-colors"
                          >
                            <Clock className="w-3 h-3" />
                            {typeof ts === 'number' ? `${ts.toFixed(1)}s` : ts}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Unknown Aspects */}
                  {msg.unknownAspects?.length > 0 && (
                    <div className="pt-2 border-t border-slate-800 text-slate-400 space-y-1">
                      <span className="text-slate-500 font-semibold text-[10px] uppercase">Unverified / Unknown Aspects:</span>
                      <ul className="list-disc list-inside text-slate-400 italic">
                        {msg.unknownAspects.map((unc, uIdx) => (
                          <li key={uIdx}>{unc}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center space-x-3 text-xs text-indigo-400 font-mono p-3 bg-slate-900/60 rounded-xl border border-slate-800 w-fit">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>AI is analyzing video memory evidence & retrieving observations...</span>
          </div>
        )}
      </div>

      {/* Input Form at Bottom */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="pt-3 border-t border-slate-800 shrink-0 flex items-center space-x-3"
      >
        <input
          type="text"
          placeholder="Ask a question about this video..."
          value={inputQuestion}
          onChange={(e) => setInputQuestion(e.target.value)}
          className="flex-1 bg-slate-900 border border-slate-800 rounded-2xl px-5 py-3.5 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none transition-colors"
        />
        <button
          type="submit"
          disabled={!inputQuestion.trim() || loading}
          className="px-5 py-3.5 rounded-2xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm shadow-xl shadow-indigo-600/30 disabled:opacity-40 transition-all active:scale-95 flex items-center justify-center gap-1.5 shrink-0"
        >
          <span>Send</span>
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
