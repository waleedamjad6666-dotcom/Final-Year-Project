import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import apiService from '../services/api';
import '../styles/Upload.css';

export default function Upload() {
  const [fileName, setFileName] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationProgress, setTranslationProgress] = useState(0);
  const [sourceLanguage, setSourceLanguage] = useState('english');
  const [targetLanguage, setTargetLanguage] = useState('urdu');
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [translationJob, setTranslationJob] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  // const [userQuota, setUserQuota] = useState(null); // Disabled for testing
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is authenticated
    if (!apiService.isAuthenticated()) {
      navigate('/login');
      return;
    }

    // Clear any stale errors on mount
    setError('');
    
    // Load user quota - disabled for testing
    // loadUserQuota();
  }, [navigate]);

  // Auto-dismiss error messages after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError('');
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // const loadUserQuota = async () => {
  //   const result = await apiService.getUserQuota();
  //   if (result.success) {
  //     setUserQuota(result.data);
  //   }
  // };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      processFile(droppedFile);
    } else {
      setError('Please select a valid video file');
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (selectedFile) => {
    // Check file size (100MB limit)
    if (selectedFile.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      return;
    }

    // Check file type
    if (!selectedFile.type.startsWith('video/')) {
      setError('Please select a valid video file');
      return;
    }

    // Check video duration (1 minute max)
    const video = document.createElement('video');
    video.preload = 'metadata';
    
    video.onloadedmetadata = function() {
      window.URL.revokeObjectURL(video.src);
      const duration = video.duration;
      
      if (duration > 60) {
        setError(`Video duration must be 1 minute or less. Your video is ${Math.floor(duration / 60)}:${Math.floor(duration % 60).toString().padStart(2, '0')} minutes.`);
        return;
      }
      
      // Duration is valid, proceed with file
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setFileDetails({
        name: selectedFile.name,
        size: (selectedFile.size / 1024 / 1024).toFixed(2) + ' MB',
        type: selectedFile.type,
        duration: `${Math.floor(duration / 60)}:${Math.floor(duration % 60).toString().padStart(2, '0')}`
      });
      setError('');
    };
    
    video.onerror = function() {
      setError('Unable to read video file. Please try a different file.');
    };
    
    video.src = URL.createObjectURL(selectedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file to upload.');
      return;
    }

    if (!apiService.isAuthenticated()) {
      setError('Please log in to upload videos.');
      navigate('/login');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setError('');
    setSuccess('');

    try {
      // Create form data
      const formData = new FormData();
      formData.append('original_video', file);
      formData.append('title', file.name.split('.')[0]);
      formData.append('description', `Video for ${sourceLanguage} to ${targetLanguage} translation`);
      formData.append('source_language', sourceLanguage === 'english' ? 'en' : 'ur');
      formData.append('target_language', targetLanguage === 'english' ? 'en' : 'ur');

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      // Upload video
      const uploadResult = await apiService.uploadVideo(formData);
      clearInterval(progressInterval);
      setUploadProgress(100);

      console.log('Upload result:', uploadResult); // Debug logging

      if (uploadResult.success) {
        setUploadedVideo(uploadResult.data);
        setSuccess('Video uploaded successfully! Redirecting to processing...');
        setIsUploading(false);
        
        // Redirect to progress page with video ID
        setTimeout(() => {
          navigate(`/progress/${uploadResult.data.id}`);
        }, 1000);
      } else {
        console.error('Upload failed:', uploadResult); // Debug logging
        setError(uploadResult.data?.detail || uploadResult.error || 'Upload failed. Please try again.');
        setIsUploading(false);
      }
    } catch (error) {
      setError('Upload failed: ' + error.message);
      setIsUploading(false);
    }
  };

  const startTranslation = async (videoId) => {
    setIsTranslating(true);
    setTranslationProgress(0);

    try {
      const translationData = {
        video: videoId,
        source_language: sourceLanguage === 'english' ? 'en' : 'ur',
        target_language: targetLanguage === 'english' ? 'en' : 'ur'
      };

      console.log('Starting translation with data:', translationData); // Debug
      const result = await apiService.createTranslation(translationData);
      
      if (result.success) {
        setTranslationJob(result.data);
        setSuccess('Translation started! Redirecting to AI processing page...');
        
        // Redirect to progress page immediately
        setTimeout(() => {
          navigate(`/progress/${result.data.processing_job}`);
        }, 1500);
      } else {
        setError(result.data?.detail || 'Translation failed to start.');
        setIsTranslating(false);
      }
    } catch (error) {
      setError('Translation failed: ' + error.message);
      setIsTranslating(false);
    }
  };

  const clearForm = () => {
    setFileName(null);
    setFileDetails(null);
    setFile(null);
    setIsUploading(false);
    setUploadProgress(0);
    setIsTranslating(false);
    setTranslationProgress(0);
    setUploadedVideo(null);
    setTranslationJob(null);
    setError('');
    setSuccess('');
  };

  const canUpload = () => {
    // Quota limit disabled for testing
    return true;
  };

  const languages = [
    { value: 'english', label: 'English' },
    { value: 'urdu', label: 'Urdu' }
  ];

  return (
    <div className="upload-container animate-fade-in">
      <aside className="sidebar">
        <h2 className="sidebar-title">🎬 AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">📊 Dashboard</Link>
          <Link to="/upload" className="active">📤 New Upload</Link>
          <Link to="/history">📚 History</Link>
          <Link to="/settings">⚙️ Settings</Link>
        </nav>
      </aside>

      <main className="upload-main">
        <h1 className="section-title">📤 New Upload</h1>
        
        {error && (
          <div className="alert alert-error">
            ❌ {error}
          </div>
        )}

        {success && (
          <div className="alert alert-success">
            ✅ {success}
          </div>
        )}
        
        <div
          className={`upload-box ${isDragging ? 'dragging' : ''} ${fileName ? 'has-file' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="upload-icon">
            {fileName ? '✅' : '📹'}
          </div>
          <p>
            {fileName ? `✨ ${fileName} ready for translation` : "Drag & drop your video file here"}
          </p>
          <label htmlFor="file-upload" className="browse-btn">
            {fileName ? '📁 Choose Different File' : '📁 Browse Files'}
          </label>
          <input
            type="file"
            id="file-upload"
            style={{ display: "none" }}
            onChange={handleFileChange}
            accept="video/*"
            disabled={isUploading || isTranslating}
          />
          <div style={{ marginTop: 'var(--space-md)', fontSize: 'var(--font-size-sm)', color: 'var(--text-muted)' }}>
            Supported formats: MP4, AVI, MOV, MKV (max 100MB, 1 minute)
          </div>
        </div>

        {fileDetails && (
          <div className="file-info card">
            <div className="file-icon">🎬</div>
            <div className="file-details">
              <h4>{fileDetails.name}</h4>
              <div className="file-meta">
                📊 {fileDetails.size} • 🎥 {fileDetails.type} {fileDetails.duration && `• ⏱️ ${fileDetails.duration}`}
              </div>
            </div>
          </div>
        )}

        <div className="language-selection card">
          <h3>🌍 Language Settings</h3>
          <div className="dropdowns">
            <div className="dropdown">
              <label>Source Language</label>
              <select 
                value={sourceLanguage} 
                onChange={(e) => setSourceLanguage(e.target.value)}
                disabled={isUploading || isTranslating}
              >
                {languages.map(lang => (
                  <option key={lang.value} value={lang.value}>{lang.label}</option>
                ))}
              </select>
            </div>
            <div className="dropdown">
              <label>Target Language</label>
              <select 
                value={targetLanguage} 
                onChange={(e) => setTargetLanguage(e.target.value)}
                disabled={isUploading || isTranslating}
              >
                {languages.map(lang => (
                  <option key={lang.value} value={lang.value}>{lang.label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="upload-actions">
          <button 
            className="btn btn-secondary"
            onClick={clearForm}
            disabled={isUploading || isTranslating}
          >
            🗑️ Clear
          </button>
          
          {!uploadedVideo && (
            <button 
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={!fileName || isUploading || isTranslating || !canUpload()}
            >
              {isUploading ? '⏳ Uploading...' : isTranslating ? '🧠 Translating...' : '🚀 Start Translation'}
            </button>
          )}

          {uploadedVideo && !isTranslating && (
            <button 
              className="btn btn-primary"
              onClick={() => navigate('/dashboard')}
            >
              📊 Go to Dashboard
            </button>
          )}
        </div>

        <div className={`upload-progress ${(isUploading || isTranslating) ? 'active' : ''}`}>
          {isUploading && (
            <>
              <div className="progress-header">
                <span className="progress-title">📤 Uploading Video</span>
                <span className="progress-percentage">{uploadProgress}%</span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </>
          )}

          {isTranslating && (
            <>
              <div className="progress-header">
                <span className="progress-title">🧠 Processing Translation</span>
                <span className="progress-percentage">{translationProgress}%</span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${translationProgress}%` }}
                ></div>
              </div>
              <div className="progress-steps">
                <span>🎤 Speech Analysis</span>
                <span>🔄 Translation</span>
                <span>🎭 Lip Sync</span>
                <span>✅ Complete</span>
              </div>
              <p style={{ marginTop: '10px', color: 'var(--text-muted)', fontSize: 'var(--font-size-sm)' }}>
                This may take several minutes depending on video length...
              </p>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
