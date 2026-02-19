import React from "react";
import { Link } from "react-router-dom";
import "../styles/Processing.css";

export default function Processing() {
  return (
    <div className="processing-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history">History</Link>
          <Link to="/settings">Settings</Link>
          <Link to="/pricing">Pricing</Link>
          <Link to="/processing" className="active">
            Processing
          </Link>
        </nav>
      </aside>

      <main className="processing-main">
        <h2 className="section-title">Processing Video</h2>
        <div className="progress-box">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: "70%" }}></div>
          </div>
          <p className="status">🔄 Syncing lips and translating voice...</p>
          <p className="eta">Estimated Time: 45 seconds</p>
        </div>
      </main>
    </div>
  );
}
