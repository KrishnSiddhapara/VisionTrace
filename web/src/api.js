import axios from 'axios';

const API_BASE = '/api';

export const uploadVideo = async (file) => {
  console.log(`[FRONTEND] Upload started: filename=${file.name}, size=${(file.size / (1024 * 1024)).toFixed(2)}MB`);
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await axios.post(`${API_BASE}/videos/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    console.log(`[FRONTEND] Upload completed: video_id=${response.data.video_id}, filename=${response.data.filename}`);
    return response.data;
  } catch (err) {
    console.error(`[FRONTEND] Upload failed:`, err);
    throw err;
  }
};

export const fetchDashboardStats = async () => {
  const response = await axios.get(`${API_BASE}/stats/dashboard`);
  return response.data;
};

export const fetchVideosList = async () => {
  const response = await axios.get(`${API_BASE}/videos/`);
  return response.data;
};

export const fetchVideoMemory = async (videoHash) => {
  const response = await axios.get(`${API_BASE}/videos/${videoHash}`);
  const mem = response.data;
  if (mem) {
    const peopleCount = mem.canonical_people?.length || mem.final_summary?.people?.length || mem.tracks?.filter(t => t.object_type?.toLowerCase() === 'person').length || 0;
    const objectsCount = mem.physical_objects?.length || mem.final_summary?.objects?.length || mem.tracks?.filter(t => t.object_type?.toLowerCase() !== 'person').length || 0;
    const eventsCount = mem.events?.length || 0;
    console.log(`[FRONTEND] Memory loaded: video_id=${mem.video_hash}, People: ${peopleCount}, Objects: ${objectsCount}, Events: ${eventsCount}`);
  }
  return mem;
};

export const askVideoQuestion = async (videoHash, question) => {
  console.log(`[FRONTEND] Asking AI question: video_id=${videoHash}, question="${question}"`);
  const response = await axios.post(`${API_BASE}/videos/${videoHash}/ask`, { question });
  console.log(`[FRONTEND] AI response received: confidence=${response.data?.confidence}`);
  return response.data;
};

export const createAnalysisStream = (videoHash, samplingMode = 'Balanced', yoloConfidence = 0.45, onMessage, onError) => {
  console.log(`[FRONTEND] Analysis stream started: video_hash=${videoHash}, mode=${samplingMode}, conf=${yoloConfidence}`);
  const url = `${API_BASE}/videos/${videoHash}/analyze/stream?sampling_mode=${encodeURIComponent(samplingMode)}&yolo_confidence=${yoloConfidence}`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log(`[FRONTEND] Analysis status: step=${data.step}/${data.total_steps}, stage="${data.stage}", progress=${data.progress}% - ${data.status}`);
      
      if (data.complete) {
        if (data.memory) {
          const pCount = data.memory.canonical_people?.length || data.memory.final_summary?.people?.length || 0;
          const oCount = data.memory.physical_objects?.length || data.memory.final_summary?.objects?.length || 0;
          const eCount = data.memory.events?.length || 0;
          console.log(`[FRONTEND] Analysis result received: People=${pCount}, Objects=${oCount}, Events=${eCount}`);
        }
        eventSource.close();
      }

      if (onMessage) onMessage(data);
    } catch (e) {
      console.error('[FRONTEND] Error parsing SSE data:', e);
    }
  };

  eventSource.onerror = (err) => {
    console.error('[FRONTEND] SSE connection error:', err);
    if (onError) onError(err);
    eventSource.close();
  };

  return () => {
    console.log('[FRONTEND] Analysis stream closed manually');
    eventSource.close();
  };
};

export const clearAllVideos = async () => {
  const response = await axios.delete(`${API_BASE}/videos/`);
  return response.data;
};

export const deleteSingleVideo = async (videoHash) => {
  const response = await axios.delete(`${API_BASE}/videos/${videoHash}`);
  return response.data;
};

