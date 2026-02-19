// API Service for AI Video Translation Platform
const API_BASE_URL = 'http://127.0.0.1:8000/api';

class ApiService {
  constructor() {
    this.token = localStorage.getItem('access_token');
  }

  // Helper method to get headers
  getHeaders(includeAuth = true) {
    const headers = {
      'Content-Type': 'application/json',
    };
    
    if (includeAuth && this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    return headers;
  }

  // Helper method to get headers for file upload
  getFileHeaders() {
    const headers = {};
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    return headers;
  }

  // Authentication Methods
  async register(userData) {
    try {
      const response = await fetch(`${API_BASE_URL}/accounts/register/`, {
        method: 'POST',
        headers: this.getHeaders(false),
        body: JSON.stringify(userData),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        this.token = data.access;
        localStorage.setItem('access_token', data.access);
        localStorage.setItem('refresh_token', data.refresh);
        localStorage.setItem('user', JSON.stringify(data.user));
      }
      
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async login(credentials) {
    try {
      const response = await fetch(`${API_BASE_URL}/accounts/login/`, {
        method: 'POST',
        headers: this.getHeaders(false),
        body: JSON.stringify(credentials),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        this.token = data.access;
        localStorage.setItem('access_token', data.access);
        localStorage.setItem('refresh_token', data.refresh);
        if (data.user) {
          localStorage.setItem('user', JSON.stringify(data.user));
        }
      }
      
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async logout() {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      const response = await fetch(`${API_BASE_URL}/accounts/logout/`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ refresh: refreshToken }),
      });
      
      this.token = null;
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      
      return { success: response.ok };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Video Management Methods
  async uploadVideo(formData) {
    try {
      console.log('Uploading video with token:', this.token ? 'Token present' : 'No token'); // Debug
      const response = await fetch(`${API_BASE_URL}/videos/`, {
        method: 'POST',
        headers: this.getFileHeaders(),
        body: formData,
      });
      
      const data = await response.json();
      console.log('Upload response:', { status: response.status, ok: response.ok, data }); // Debug
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      console.error('Upload error:', error); // Debug
      return { success: false, error: error.message };
    }
  }

  async getVideos() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getVideoDetails(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async deleteVideo(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/`, {
        method: 'DELETE',
        headers: this.getHeaders(),
      });
      
      return { success: response.ok, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Translation Methods
  async createTranslation(translationData) {
    try {
      console.log('Creating translation:', translationData); // Debug
      const response = await fetch(`${API_BASE_URL}/videos/translate/`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(translationData),
      });
      
      const data = await response.json();
      console.log('Translation response:', { status: response.status, ok: response.ok, data }); // Debug
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      console.error('Translation error:', error); // Debug
      return { success: false, error: error.message };
    }
  }

  async getTranslations() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/translations/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getTranslationDetails(translationId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/translations/${translationId}/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async downloadTranslatedVideo(translationId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/translations/${translationId}/download/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Processing Job Methods
  async getProcessingJobs() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/jobs/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getProcessingStatus(jobId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/jobs/${jobId}/status/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async cancelProcessing(jobId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/jobs/${jobId}/cancel/`, {
        method: 'POST',
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // User Data Methods
  async getUserProfile() {
    try {
      const response = await fetch(`${API_BASE_URL}/accounts/profile/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getUserQuota() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/quota/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getDashboardStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/stats/user/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Health Check
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health/`);
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Lip Sync Methods
  async startLipSync(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/lip-sync/start/`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ video_id: videoId }),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getLipSyncJobs(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/lip-sync/by-video/${videoId}/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async getLipSyncStatus(jobId) {
    try {
      const response = await fetch(`${API_BASE_URL}/lip-sync/${jobId}/status/`, {
        headers: this.getHeaders(),
      });
      
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // Utility Methods
  isAuthenticated() {
    return !!this.token;
  }

  getCurrentUser() {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  }

  clearAuth() {
    this.token = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  }
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;