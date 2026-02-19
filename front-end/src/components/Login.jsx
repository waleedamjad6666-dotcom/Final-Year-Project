import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import apiService from "../services/api";
import "../styles/Login.css";

export default function Login() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    email: "",
    password: ""
  });
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError(''); // Clear error when user types
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    try {
      const result = await apiService.login({
        email: formData.email,
        password: formData.password
      });

      if (result.success) {
        // Redirect to dashboard on successful login
        navigate('/dashboard');
      } else {
        setError(result.data?.detail || 'Login failed. Please check your credentials.');
      }
    } catch (error) {
      setError('Login failed: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container animate-fade-in">
      <form className="login-form card-glass" onSubmit={handleSubmit}>
        <div className="form-header">
          <h2>🎬 AI Video Translate</h2>
          <h3>Welcome Back</h3>
        </div>
        
        {error && (
          <div className="alert alert-error">
            ❌ {error}
          </div>
        )}
        
        <div className="form-group">
          <input
            type="email"
            name="email"
            placeholder="📧 Enter your email"
            value={formData.email}
            onChange={handleInputChange}
            required
            className="form-input"
            disabled={isLoading}
          />
        </div>
        
        <div className="form-group">
          <input
            type="password"
            name="password"
            placeholder="🔒 Enter your password"
            value={formData.password}
            onChange={handleInputChange}
            required
            className="form-input"
            disabled={isLoading}
          />
        </div>
        
        <button 
          type="submit" 
          className={`btn btn-primary btn-lg ${isLoading ? 'loading' : ''}`}
          disabled={isLoading}
        >
          {isLoading ? '⏳ Signing In...' : '🚀 Sign In'}
        </button>
        
        <div className="form-footer">
          <p>
            Don't have an account? <Link to="/signup">Create one here</Link>
          </p>
        </div>
      </form>
    </div>
  );
}
