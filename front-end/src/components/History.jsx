import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import apiService from "../services/api";
import "../styles/History.css";

export default function History() {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (!apiService.isAuthenticated()) {
      navigate('/login');
      return;
    }
    loadVideos();
  }, [navigate]);

  const loadVideos = async () => {
    setLoading(true);
    try {
      const result = await apiService.getVideos();
      if (result.success) {
        setVideos(result.data.results || result.data || []);
      } else {
        setError('Failed to load videos');
      }
    } catch (error) {
      setError('Error loading videos: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (videoId) => {
    try {
      const result = await apiService.getVideoDetails(videoId);
      if (result.success && result.data.processed_video) {
        const downloadUrl = `http://127.0.0.1:8000${result.data.processed_video}`;
        window.open(downloadUrl, '_blank');
      } else {
        alert('Processed video not available yet.');
      }
    } catch (error) {
      alert('Download failed: ' + error.message);
    }
  };

  const handleDelete = async (videoId) => {
    if (window.confirm('Are you sure you want to delete this video?')) {
      try {
        const result = await apiService.deleteVideo(videoId);
        if (result.success) {
          loadVideos();
        } else {
          alert('Failed to delete video');
        }
      } catch (error) {
        alert('Delete failed: ' + error.message);
      }
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "processing": return "⏳";
      case "completed": return "✅";
      case "failed": return "❌";
      case "pending": return "⏸️";
      default: return "📄";
    }
  };

  const formatStatus = (status, progress = 0) => {
    switch (status) {
      case "processing": return `Processing ${progress}%`;
      case "completed": return "Completed";
      case "failed": return "Failed";
      case "pending": return "Pending";
      default: return "Unknown";
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="history-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history" className="active">
            History
          </Link>
          <Link to="/settings">Settings</Link>
        </nav>
      </aside>

      <main className="history-main">
        <h2 className="section-title">Translation History</h2>
        
        {error && (
          <div className="alert alert-error">
            ❌ {error}
          </div>
        )}

        {loading ? (
          <div className="loading-spinner">⏳ Loading videos...</div>
        ) : videos.length === 0 ? (
          <div className="no-videos">
            <p>📹 No videos found. <Link to="/upload">Upload your first video</Link> to get started!</p>
          </div>
        ) : (
          <div className="history-list">
            {videos.map((video) => (
              <div key={video.id} className="history-item">
                <div className="history-info">
                  <h4>{video.title}</h4>
                  <p><strong>Uploaded:</strong> {formatDate(video.created_at)}</p>
                  <p><strong>Status:</strong> {getStatusIcon(video.status)} {formatStatus(video.status, video.progress)}</p>
                  {video.source_language && video.target_language && (
                    <p><strong>Translation:</strong> {video.source_language.toUpperCase()} → {video.target_language.toUpperCase()}</p>
                  )}
                </div>
                
                <div className="history-actions">
                  {video.status === 'completed' && (
                    <button onClick={() => handleDownload(video.id)} className="btn btn-primary">
                      📥 Download
                    </button>
                  )}
                  {video.status === 'processing' && (
                    <Link to={`/progress/${video.id}`} className="btn btn-secondary">
                      👁️ View Progress
                    </Link>
                  )}
                  {video.status === 'failed' && (
                    <button className="btn btn-warning" disabled>
                      ❌ Failed
                    </button>
                  )}
                  <button onClick={() => handleDelete(video.id)} className="btn btn-danger">
                    🗑️ Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
