import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import apiService from "../services/api";
import "../styles/SignUp.css";

export default function SignUp() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [formData, setFormData] = useState({
    first_name: "",
    last_name: "",
    email: "",
    password: "",
    password_confirm: "",
    agreeToTerms: false
  });
  const [passwordStrength, setPasswordStrength] = useState("weak");
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value
    });

    // Check password strength
    if (name === "password") {
      const strength = getPasswordStrength(value);
      setPasswordStrength(strength);
    }

    setError(''); // Clear error when user types
  };

  const getPasswordStrength = (password) => {
    if (password.length < 6) return "weak";
    if (password.length < 10) return "medium";
    return "strong";
  };

  const validateForm = () => {
    if (!formData.first_name.trim()) {
      setError('First name is required');
      return false;
    }
    if (!formData.last_name.trim()) {
      setError('Last name is required');
      return false;
    }
    if (!formData.email.trim()) {
      setError('Email is required');
      return false;
    }
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      return false;
    }
    if (formData.password !== formData.password_confirm) {
      setError('Passwords do not match');
      return false;
    }
    if (!formData.agreeToTerms) {
      setError('Please agree to the terms and conditions');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    setError('');
    setSuccess('');
    
    try {
      // Generate username from email (before @ symbol)
      const username = formData.email.split('@')[0].toLowerCase().replace(/[^a-z0-9]/g, '');
      
      const userData = {
        username: username,
        first_name: formData.first_name,
        last_name: formData.last_name,
        email: formData.email,
        password: formData.password,
        password2: formData.password_confirm
      };

      const result = await apiService.register(userData);

      if (result.success) {
        setSuccess('Account created successfully! Redirecting to dashboard...');
        setTimeout(() => {
          navigate('/dashboard');
        }, 2000);
      } else {
        // Handle all possible error fields
        const errors = result.data;
        if (errors) {
          const errorMessages = [];
          
          if (errors.username) errorMessages.push('Username: ' + (Array.isArray(errors.username) ? errors.username[0] : errors.username));
          if (errors.email) errorMessages.push('Email: ' + (Array.isArray(errors.email) ? errors.email[0] : errors.email));
          if (errors.password) errorMessages.push('Password: ' + (Array.isArray(errors.password) ? errors.password[0] : errors.password));
          if (errors.password2) errorMessages.push('Password confirmation: ' + (Array.isArray(errors.password2) ? errors.password2[0] : errors.password2));
          if (errors.first_name) errorMessages.push('First name: ' + (Array.isArray(errors.first_name) ? errors.first_name[0] : errors.first_name));
          if (errors.last_name) errorMessages.push('Last name: ' + (Array.isArray(errors.last_name) ? errors.last_name[0] : errors.last_name));
          if (errors.detail) errorMessages.push(errors.detail);
          if (errors.non_field_errors) errorMessages.push(Array.isArray(errors.non_field_errors) ? errors.non_field_errors[0] : errors.non_field_errors);
          
          setError(errorMessages.length > 0 ? errorMessages.join(' | ') : 'Registration failed. Please try again.');
        } else {
          setError('Registration failed. Please check your connection and try again.');
        }
      }
    } catch (error) {
      setError('Registration failed: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="signup-container animate-fade-in">
      <form className="signup-form card-glass" onSubmit={handleSubmit}>
        <div className="form-header">
          <h2>🎬 AI Video Translate</h2>
          <h3>Create Your Account</h3>
        </div>
        
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
        
        <input
          type="text"
          name="first_name"
          placeholder="👤 First Name"
          value={formData.first_name}
          onChange={handleInputChange}
          required
          disabled={isLoading}
        />

        <input
          type="text"
          name="last_name"
          placeholder="👤 Last Name"
          value={formData.last_name}
          onChange={handleInputChange}
          required
          disabled={isLoading}
        />
        
        <input
          type="email"
          name="email"
          placeholder="📧 Email Address"
          value={formData.email}
          onChange={handleInputChange}
          required
          disabled={isLoading}
        />
        
        <div>
          <input
            type="password"
            name="password"
            placeholder="🔒 Create Password"
            value={formData.password}
            onChange={handleInputChange}
            required
            disabled={isLoading}
          />
          {formData.password && (
            <div className="password-strength">
              <div className={`password-strength-fill ${passwordStrength}`}></div>
            </div>
          )}
        </div>

        <input
          type="password"
          name="password_confirm"
          placeholder="🔒 Confirm Password"
          value={formData.password_confirm}
          onChange={handleInputChange}
          required
          disabled={isLoading}
        />
        
        <div className="terms-checkbox">
          <input
            type="checkbox"
            name="agreeToTerms"
            checked={formData.agreeToTerms}
            onChange={handleInputChange}
            required
            disabled={isLoading}
          />
          <label>
            I agree to the <Link to="/terms">Terms of Service</Link> and{" "}
            <Link to="/privacy">Privacy Policy</Link>
          </label>
        </div>
        
        <button 
          type="submit" 
          className={`btn btn-primary btn-lg ${isLoading ? 'loading' : ''}`}
          disabled={isLoading}
        >
          {isLoading ? '⏳ Creating Account...' : '🚀 Create Account'}
        </button>
        
        <p>
          Already have an account? <Link to="/login">Sign in here</Link>
        </p>
      </form>
    </div>
  );
}
