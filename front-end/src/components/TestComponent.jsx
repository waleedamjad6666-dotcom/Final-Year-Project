import React from "react";

function TestComponent() {
  return (
    <div style={{ 
      padding: '20px', 
      backgroundColor: '#f0f0f0', 
      color: '#333',
      textAlign: 'center',
      fontSize: '24px'
    }}>
      <h1>🎬 AI Video Translation Platform</h1>
      <p>Frontend is working! ✅</p>
      <div style={{ marginTop: '20px' }}>
        <a href="/login" style={{ margin: '10px', padding: '10px 20px', backgroundColor: '#007bff', color: 'white', textDecoration: 'none', borderRadius: '5px' }}>
          Go to Login
        </a>
        <a href="/signup" style={{ margin: '10px', padding: '10px 20px', backgroundColor: '#28a745', color: 'white', textDecoration: 'none', borderRadius: '5px' }}>
          Go to Signup
        </a>
      </div>
    </div>
  );
}

export default TestComponent;