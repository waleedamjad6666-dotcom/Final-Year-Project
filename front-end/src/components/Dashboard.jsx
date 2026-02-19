import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import apiService from "../services/api";
import "../styles/Dashboard.css";

export default function Dashboard() {
  const [dashboardData, setDashboardData] = useState(null);
  const [videos, setVideos] = useState([]);
  // const [userQuota, setUserQuota] = useState(null); // Disabled for testing
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is authenticated
    if (!apiService.isAuthenticated()) {
      console.error('User not authenticated, redirecting to login');
      navigate('/login');
      return;
    }

    console.log('Loading dashboard data...');
    loadDashboardData();
  }, [navigate]);

  const loadDashboardData = async () => {
    console.log('loadDashboardData called');
    setLoading(true);
    try {
      // Load all dashboard data in parallel
      const [dashboardResult, videosResult] = await Promise.all([
        apiService.getDashboardStats(),
        apiService.getVideos(),
        // apiService.getUserQuota() // Disabled for testing
      ]);

      console.log('Dashboard result:', dashboardResult);
      console.log('Videos result:', videosResult);

      if (dashboardResult.success) {
        setDashboardData(dashboardResult.data);
      }

      if (videosResult.success) {
        setVideos(videosResult.data.results || videosResult.data || []);
      }

      // Quota handling disabled for testing
      // if (quotaResult.success) {
      //   setUserQuota(quotaResult.data);
      // }
    } catch (error) {
      console.error('Dashboard error:', error);
      setError('Failed to load dashboard data: ' + error.message);
    } finally {
      console.log('Loading complete');
      setLoading(false);
    }
  };

  const handleDownload = async (videoId) => {
    try {
      const result = await apiService.getVideoDetails(videoId);
      if (result.success && result.data.processed_video) {
        // Create download URL - check if it's already a full URL
        let downloadUrl = result.data.processed_video;
        if (!downloadUrl.startsWith('http')) {
          downloadUrl = `http://127.0.0.1:8000${downloadUrl}`;
        }
        window.open(downloadUrl, '_blank');
      } else {
        alert('Processed video not available yet. Please wait for processing to complete.');
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
          // Reload dashboard data
          loadDashboardData();
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

  if (loading) {
    return (
      <div className="dashboard-container animate-fade-in">
        <aside className="sidebar">
          <h2 className="sidebar-title">🎬 AI VIDEO TRANSLATION</h2>
          <nav className="menu">
            <Link to="/dashboard" className="active">📊 Dashboard</Link>
            <Link to="/upload">📤 New Upload</Link>
            <Link to="/history">📚 History</Link>
            <Link to="/settings">⚙️ Settings</Link>
          </nav>
        </aside>
        <main className="dashboard-main">
          <div className="loading-spinner">⏳ Loading dashboard...</div>
        </main>
      </div>
    );
  }

  return (
    <div className="dashboard-container animate-fade-in">
      <aside className="sidebar">
        <h2 className="sidebar-title">🎬 AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard" className="active">
            📊 Dashboard
          </Link>
          <Link to="/upload">
            📤 New Upload
          </Link>
          <Link to="/history">
            📚 History
          </Link>
          <Link to="/settings">
            ⚙️ Settings
          </Link>
        </nav>
      </aside>

      <main className="dashboard-main">
        <div className="dashboard-header">
          <h1 className="section-title">Dashboard</h1>
          <Link to="/upload" className="btn btn-primary">
            🚀 New Translation
          </Link>
        </div>

        {error && (
          <div className="alert alert-error">
            ❌ {error}
          </div>
        )}

        <div className="dashboard-stats">
          <div className="stat-card">
            <div className="stat-number">{dashboardData?.total_videos || videos.length}</div>
            <div className="stat-label">Total Videos</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{dashboardData?.completed_translations || videos.filter(v => v.status === 'completed').length}</div>
            <div className="stat-label">Completed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{dashboardData?.processing_jobs || videos.filter(v => v.status === 'processing').length}</div>
            <div className="stat-label">Processing</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">∞</div>
            <div className="stat-label">Monthly Limit</div>
          </div>
        </div>

        {/* Quota section disabled for testing
        {userQuota && (
          <div className="quota-section">
            <h3>Monthly Usage</h3>
            <div className="quota-bar">
              <div 
                className="quota-fill" 
                style={{ width: `${(userQuota.videos_translated / userQuota.max_videos_per_month) * 100}%` }}
              />
            </div>
            <span>{userQuota.videos_translated}/{userQuota.max_videos_per_month} videos used</span>
          </div>
        )}
        */}

        <div className="recent-translations">
          <h3>Recent Translations</h3>
          {videos.length === 0 ? (
            <div className="no-translations">
              <p>🎬 No translations yet. <Link to="/upload">Upload your first video</Link> to get started!</p>
            </div>
          ) : (
            <div className="translations-list">
              {videos.slice(0, 5).map((video) => (
                <div key={video.id} className="translation-card">
                  <div className="translation-info">
                    <h4>{video.title || `Video ${video.id}`}</h4>
                    <p>
                      {video.source_language?.toUpperCase()} → {video.target_language?.toUpperCase()}
                    </p>
                    <span className="translation-date">
                      {formatDate(video.created_at)}
                    </span>
                  </div>
                  
                  <div className="translation-status">
                    <span className={`status-badge ${video.status}`}>
                      {getStatusIcon(video.status)} {formatStatus(video.status, video.progress)}
                    </span>
                  </div>
                  
                  <div className="translation-actions">
                    {video.status === 'completed' && (
                      <button 
                        onClick={() => handleDownload(video.id)}
                        className="btn btn-sm btn-primary"
                      >
                        📥 Download
                      </button>
                    )}
                    {video.status === 'processing' && (
                      <div className="progress-mini">
                        <div 
                          className="progress-fill" 
                          style={{ width: `${video.progress || 0}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="recent-videos">
          <h3>Recent Videos</h3>
          {videos.length === 0 ? (
            <div className="no-videos">
              <p>📹 No videos uploaded yet.</p>
            </div>
          ) : (
            <div className="video-grid">
              {videos.slice(0, 4).map((video) => (
                <div key={video.id} className="video-card">
                  <div className="video-thumbnail">
                    {video.thumbnail ? (
                      <img src={`http://127.0.0.1:8000${video.thumbnail}`} alt={video.title} />
                    ) : (
                      <div className="video-placeholder">
                        🎬
                      </div>
                    )}
                    <span className={`status-badge-small ${video.status}`}>
                      {getStatusIcon(video.status)}
                    </span>
                  </div>
                  
                  <div className="video-info">
                    <h4>{video.title}</h4>
                    <p>{formatStatus(video.status, video.progress)}</p>
                    <span className="video-date">
                      {formatDate(video.created_at)}
                    </span>
                  </div>
                  
                  <div className="video-actions">
                    {video.status === 'completed' && (
                      <button 
                        onClick={() => handleDownload(video.id)}
                        className="btn btn-sm btn-primary"
                      >
                        📥 Download
                      </button>
                    )}
                    {video.status === 'processing' && (
                      <Link 
                        to={`/progress/${video.id}`} 
                        className="btn btn-sm btn-secondary"
                      >
                        👁️ View Progress
                      </Link>
                    )}
                    <button 
                      onClick={() => handleDelete(video.id)}
                      className="btn btn-sm btn-danger"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
