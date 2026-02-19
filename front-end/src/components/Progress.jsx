import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import '../styles/Progress.css';

const Progress = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  
  // AI Module states - Updated for S2S Translation Pipeline
  const [modules, setModules] = useState({
    audioExtraction: { status: 'pending', progress: 0, message: '' },
    s2sTranslation: { status: 'pending', progress: 0, message: '' },
    lipSync: { status: 'pending', progress: 0, message: '' }
  });
  
  const [overallProgress, setOverallProgress] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [error, setError] = useState(null);
  const [videoData, setVideoData] = useState(null);

  useEffect(() => {
    console.log('Progress page loaded with videoId:', videoId); // Debug
    
    if (!videoId) {
      console.log('No videoId found, redirecting to upload');
      navigate('/upload');
      return;
    }

    const pollProgress = async () => {
      try {
        const accessToken = localStorage.getItem('access_token');
        
        if (!accessToken) {
          navigate('/login');
          return;
        }

        // Fetch video status
        const response = await fetch(`http://127.0.0.1:8000/api/videos/${videoId}/`, {
          headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.status === 401) {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          navigate('/login');
          return;
        }

        if (response.ok) {
          const data = await response.json();
          console.log('Video data:', data); // Debug log
          
          // Store video data for download button
          setVideoData(data);
          
          // Map backend progress to frontend modules (NEW S2S PIPELINE)
          const progress = data.progress || 0;
          const videoStatus = data.status;
          
          let newModules = {
            audioExtraction: { status: 'pending', progress: 0, message: 'Waiting to start...' },
            s2sTranslation: { status: 'pending', progress: 0, message: 'Waiting to start...' },
            lipSync: { status: 'pending', progress: 0, message: 'Waiting to start...' }
          };
          
          // Module 1: Audio Extraction (0-20%)
          if (progress >= 10) {
            newModules.audioExtraction = {
              status: progress >= 20 ? 'completed' : 'in-progress',
              progress: Math.min(progress * 5, 100),
              message: progress >= 20 ? '✓ Audio separated from video successfully' : 'Separating audio and video tracks...'
            };
          }
          
          // Module 2: Speech-to-Speech Translation (20-80%)
          // This replaces: transcription + translation + voice cloning
          if (progress >= 20) {
            newModules.s2sTranslation = {
              status: progress >= 80 ? 'completed' : 'in-progress',
              progress: Math.min((progress - 20) * 1.67, 100),
              message: progress >= 80 
                ? '✓ Speech translated with emotions & tone preserved' 
                : progress >= 60
                ? 'Translating speech with Seamless M4T (preserving voice, emotions, naturalness)...'
                : progress >= 40
                ? 'Processing audio chunks for translation...'
                : 'Initializing speech-to-speech translation engine...'
            };
          }
          
          // Module 3: Lip Synchronization (80-100%)
          if (progress >= 80) {
            newModules.lipSync = {
              status: progress >= 100 ? 'completed' : 'in-progress',
              progress: Math.min((progress - 80) * 5, 100),
              message: progress >= 100 
                ? '✓ Lip movements synchronized perfectly' 
                : progress >= 93 
                ? 'Composing final video with synchronized lips...' 
                : 'Synchronizing lip movements with translated audio (Wav2Lip)...'
            };
          }
          
          setModules(newModules);
          setOverallProgress(progress);
          
          if (videoStatus === 'completed') {
            setIsCompleted(true);
          } else if (videoStatus === 'failed') {
            setError(data.error_message || 'Processing failed');
          }
        } else {
          console.error('Failed to fetch video:', response.status);
          setError(`Failed to fetch video (${response.status})`);
        }
      } catch (error) {
        console.error('Error polling progress:', error);
        setError('Failed to fetch progress: ' + error.message);
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollProgress, 2000);
    pollProgress(); // Initial call

    return () => clearInterval(interval);
  }, [videoId, navigate]);

  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  const handleDownload = async () => {
    if (!videoData || !videoData.processed_video) {
      alert('Processed video not available yet');
      return;
    }

    try {
      const videoUrl = `http://127.0.0.1:8000${videoData.processed_video}`;
      const link = document.createElement('a');
      link.href = videoUrl;
      link.download = `processed_video_${videoId}.mp4`;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download video');
    }
  };

  const handleDownloadTranslatedAudio = async () => {
    if (!videoData) {
      alert('Video data not available');
      return;
    }

    // Check if S2S translation is completed
    if (modules.s2sTranslation.status !== 'completed') {
      alert('Translation is still in progress. Please wait...');
      return;
    }

    try {
      const accessToken = localStorage.getItem('access_token');
      
      // Use the translated_audio_url from backend response
      const audioUrl = videoData.translated_audio_url 
        ? `http://127.0.0.1:8000${videoData.translated_audio_url}`
        : `http://127.0.0.1:8000/media/processing/${videoData.id}/translated_audio.wav`;
      
      console.log('📥 Downloading audio from:', audioUrl);
      console.log('📊 Video data:', videoData);
      
      const response = await fetch(audioUrl, {
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `translated_audio_${videoData.id}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } else {
        // More informative error
        console.error('Audio download response:', response.status, response.statusText);
        alert(`Translated audio file not found. The file may still be processing or there was an error. (Status: ${response.status})`);
      }
    } catch (error) {
      console.error('Audio download failed:', error);
      alert('Failed to download translated audio: ' + error.message);
    }
  };

  const handleDownloadClonedAudio = async () => {
    if (!videoData || !videoData.cloned_audio_url) {
      alert('Cloned audio not available yet');
      return;
    }

    try {
      const audioUrl = `http://127.0.0.1:8000${videoData.cloned_audio_url}`;
      const link = document.createElement('a');
      link.href = audioUrl;
      link.download = `cloned_audio_${videoId}.wav`;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Audio download failed:', error);
      alert('Failed to download audio');
    }
  };

  const handleDownloadVideo = async () => {
    if (!videoData || !videoData.processed_video_url) {
      alert('Processed video not available yet');
      return;
    }

    try {
      console.log('📥 Downloading final video from:', videoData.processed_video_url);
      
      const videoUrl = videoData.processed_video_url.startsWith('http') 
        ? videoData.processed_video_url
        : `http://127.0.0.1:8000${videoData.processed_video_url}`;
      
      const link = document.createElement('a');
      link.href = videoUrl;
      link.download = `translated_video_${videoId}.mp4`;
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Video download failed:', error);
      alert('Failed to download video: ' + error.message);
    }
  };

  const getModuleStatusIcon = (status) => {
    switch (status) {
      case 'completed': return '✅';
      case 'in-progress': return '⏳';
      case 'failed': return '❌';
      default: return '⏸️';
    }
  };

  const getOverallColor = () => {
    if (error) return '#ff4757';
    if (isCompleted) return '#2ed573';
    return '#3742fa';
  };

  return (
    <div className="progress-container">
      <div className="progress-card">
        <div className="progress-header">
          <h1>🤖 AI Video Processing Pipeline</h1>
          <p>Real-time AI modules tracking</p>
        </div>

        <div className="progress-content">
          {/* Overall Progress Circle */}
          <div className="progress-circle-container">
            <div className="progress-circle">
              <svg className="progress-ring" width="200" height="200">
                <circle
                  className="progress-ring-circle-bg"
                  cx="100"
                  cy="100"
                  r="80"
                  fill="transparent"
                  stroke="#e3e3e3"
                  strokeWidth="8"
                />
                <circle
                  className="progress-ring-circle"
                  cx="100"
                  cy="100"
                  r="80"
                  fill="transparent"
                  stroke={getOverallColor()}
                  strokeWidth="8"
                  strokeLinecap="round"
                  strokeDasharray="502.654"
                  strokeDashoffset={502.654 - (502.654 * overallProgress) / 100}
                />
              </svg>
              <div className="progress-text">
                <span className="progress-percentage">{Math.round(overallProgress)}%</span>
                <span className="progress-status">
                  {isCompleted ? 'Completed' : error ? 'Failed' : 'Processing'}
                </span>
              </div>
            </div>
          </div>

          {/* AI Modules - NEW S2S PIPELINE */}
          <div className="ai-modules">
            <h3>🔬 AI Processing Modules (New S2S Pipeline)</h3>
            
            {/* Module 1: Audio/Video Separation */}
            <div className={`module module-${modules.audioExtraction.status}`}>
              <div className="module-header">
                <span className="module-icon">{getModuleStatusIcon(modules.audioExtraction.status)}</span>
                <h4>🎵 Module 1: Audio & Video Separation</h4>
              </div>
              <div className="module-progress-bar">
                <div 
                  className="module-progress-fill" 
                  style={{width: `${modules.audioExtraction.progress}%`}}
                />
              </div>
              <p className="module-message">{modules.audioExtraction.message}</p>
              <p className="module-description">Extracting audio track from video file</p>
            </div>

            {/* Module 2: Speech-to-Speech Translation (NEW!) */}
            <div className={`module module-${modules.s2sTranslation.status}`}>
              <div className="module-header">
                <span className="module-icon">{getModuleStatusIcon(modules.s2sTranslation.status)}</span>
                <h4>🌐 Module 2: Speech-to-Speech Translation (Seamless M4T)</h4>
                {modules.s2sTranslation.status === 'completed' && (
                  <button 
                    className="download-audio-btn"
                    onClick={handleDownloadTranslatedAudio}
                    title="Download translated audio with voice cloning"
                  >
                    🎵 Download Audio (Voice Cloned)
                  </button>
                )}
              </div>
              <div className="module-progress-bar">
                <div 
                  className="module-progress-fill" 
                  style={{width: `${modules.s2sTranslation.progress}%`}}
                />
              </div>
              <p className="module-message">{modules.s2sTranslation.message}</p>
              <p className="module-description">
                ✨ Direct speech translation preserving:
                • Voice tone & emotions 🎭
                • Speaking style & naturalness 🗣️
                • Speaker characteristics 👤
                • Prosody & rhythm 🎵
              </p>
            </div>

            {/* Module 3: Lip Synchronization */}
            <div className={`module module-${modules.lipSync.status}`}>
              <div className="module-header">
                <span className="module-icon">{getModuleStatusIcon(modules.lipSync.status)}</span>
                <h4>👄 Module 3: Lip Synchronization (Wav2Lip)</h4>
                {modules.lipSync.status === 'completed' && (
                  <button 
                    className="download-video-btn"
                    onClick={handleDownloadVideo}
                    title="Download final translated video with lip sync"
                  >
                    🎬 Download Video
                  </button>
                )}
              </div>
              <div className="module-progress-bar">
                <div 
                  className="module-progress-fill" 
                  style={{width: `${modules.lipSync.progress}%`}}
                />
              </div>
              <p className="module-message">{modules.lipSync.message}</p>
              <p className="module-description">
                🎬 Creating final video with:
                • Lip movements synchronized 👄
                • Voice-cloned translated audio 🎤
                • High-quality rendering 🎨
                • Original video quality preserved 📺
              </p>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <h4>❌ Error:</h4>
              <p>{error}</p>
            </div>
          )}

          {/* Success Message */}
          {isCompleted && (
            <div className="success-message">
              <h3>🎉 Processing Completed Successfully!</h3>
              <p>All AI modules have finished processing your video.</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-buttons">
            {/* Download Translated Audio (Available after Module 2 - 80%) */}
            {modules.s2sTranslation.status === 'completed' && (
              <button className="download-btn secondary" onClick={handleDownloadTranslatedAudio}>
                🎵 Download Audio (Voice Cloned)
              </button>
            )}
            
            {/* Download Final Video (Available after Module 3 - 100%) */}
            {isCompleted && videoData?.processed_video_url && (
              <button className="download-btn primary" onClick={handleDownloadVideo}>
                🎬 Download Translated Video (Lip Synced)
              </button>
            )}
            
            <button className="back-btn" onClick={handleBackToDashboard}>
              🏠 Back to Dashboard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Progress;