import React from "react";
import { Link } from "react-router-dom";
import "../styles/Help.css";

export default function Help() {
  return (
    <div className="help-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history">History</Link>
          <Link to="/settings">Settings</Link>
          <Link to="/pricing">Pricing</Link>
          <Link to="/processing">Processing</Link>
          <Link to="/features">Features</Link>
          <Link to="/help" className="active">
            Help
          </Link>
        </nav>
      </aside>

      <main className="help-main">
        <h2 className="section-title">Need Help?</h2>
        <div className="help-content">
          <p>
            <strong>How to Upload:</strong> Go to "New Upload", drag or select
            your video, choose source and target language, and click start.
          </p>
          <p>
            <strong>Translation Time:</strong> Usually 30-60 seconds depending
            on video length.
          </p>
          <p>
            <strong>Languages Supported:</strong> English and Urdu with 
            seamless bidirectional translation.
          </p>
          <p>
            <strong>Download Issues:</strong> If download fails, retry or check
            your connection.
          </p>
          <p>
            <strong>Contact Support:</strong> Reach us at
            support@aivideotranslate.io
          </p>
        </div>
      </main>
    </div>
  );
}
