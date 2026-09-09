import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardPage from './pages/DashboardPage';
import AnalyzePage from './pages/AnalyzePage';
import AnalyticsPage from './pages/AnalyticsPage';
import EventsTimelinePage from './pages/EventsTimelinePage';
import AskAIPage from './pages/AskAIPage';
import SettingsPage from './pages/SettingsPage';
import { fetchVideosList, fetchVideoMemory } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeMemory, setActiveMemory] = useState(null);
  const [seekTime, setSeekTime] = useState(null);

  // Load latest memory on startup if available
  useEffect(() => {
    async function loadLatest() {
      try {
        const list = await fetchVideosList();
        if (list?.videos?.length > 0) {
          const latestHash = list.videos[0].video_hash;
          const memoryData = await fetchVideoMemory(latestHash);
          if (memoryData) setActiveMemory(memoryData);
        }
      } catch (err) {
        console.log('No previous memory loaded on startup');
      }
    }
    loadLatest();
  }, []);

  const handleSelectVideoHash = async (hash) => {
    try {
      const memoryData = await fetchVideoMemory(hash);
      if (memoryData) {
        setActiveMemory(memoryData);
        setActiveTab('analyze');
      }
    } catch (err) {
      alert(`Failed to load video memory: ${err.message}`);
    }
  };

  const handleSeekToTimestamp = (timestamp) => {
    setSeekTime(timestamp);
    setActiveTab('analyze');
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950 text-slate-100 selection:bg-indigo-500 selection:text-white">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
      />

      {/* Main Content View */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <Header
          activeTab={activeTab}
          activeMemory={activeMemory}
          onNewAnalysisClick={() => setActiveTab('analyze')}
        />

        <main className="flex-1 overflow-y-auto bg-slate-950">
          {activeTab === 'dashboard' && (
            <DashboardPage
              onAnalyzeClick={() => setActiveTab('analyze')}
              onSelectVideo={handleSelectVideoHash}
            />
          )}

          {activeTab === 'analyze' && (
            <AnalyzePage
              activeMemory={activeMemory}
              setActiveMemory={setActiveMemory}
              seekTime={seekTime}
            />
          )}

          {activeTab === 'analytics' && <AnalyticsPage activeMemory={activeMemory} />}

          {activeTab === 'events' && (
            <EventsTimelinePage
              activeMemory={activeMemory}
              onSeekToTimestamp={handleSeekToTimestamp}
            />
          )}

          {activeTab === 'ask-ai' && (
            <AskAIPage
              activeMemory={activeMemory}
              onSeekToTimestamp={handleSeekToTimestamp}
            />
          )}

          {activeTab === 'settings' && <SettingsPage />}
        </main>
      </div>
    </div>
  );
}
