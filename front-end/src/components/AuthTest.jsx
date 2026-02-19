import React, { useState } from 'react';

const AuthTest = () => {
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const testAuth = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('authToken');
      console.log('Testing with token:', token ? 'Token exists' : 'No token');
      
      if (!token) {
        setResult('❌ No auth token found in localStorage');
        setLoading(false);
        return;
      }

      const response = await fetch('http://127.0.0.1:8000/api/v1/videos/jobs/', {
        headers: {
          'Authorization': `Token ${token}`,
          'Content-Type': 'application/json',
        },
      });

      console.log('Auth test response:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        setResult(`✅ Authentication successful! Found ${data.length} jobs`);
      } else {
        const errorData = await response.text();
        setResult(`❌ Auth failed: ${response.status} - ${errorData}`);
      }
    } catch (error) {
      setResult(`❌ Error: ${error.message}`);
    }
    setLoading(false);
  };

  const clearAuth = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    setResult('🗑️ Auth data cleared');
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', margin: '20px' }}>
      <h3>🔐 Authentication Test</h3>
      <button onClick={testAuth} disabled={loading}>
        {loading ? '⏳ Testing...' : '🧪 Test Authentication'}
      </button>
      <button onClick={clearAuth} style={{ marginLeft: '10px' }}>
        🗑️ Clear Auth
      </button>
      <div style={{ marginTop: '10px', padding: '10px', backgroundColor: '#f5f5f5' }}>
        {result || 'Click "Test Authentication" to check your login status'}
      </div>
    </div>
  );
};

export default AuthTest;