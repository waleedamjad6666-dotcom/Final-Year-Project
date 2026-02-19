import React from "react";
import { Link } from "react-router-dom";
import "../styles/Features.css";

export default function Features() {
  return (
    <div className="features-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history">History</Link>
          <Link to="/settings">Settings</Link>
          <Link to="/pricing">Pricing</Link>
          <Link to="/processing">Processing</Link>
          <Link to="/features" className="active">
            Features
          </Link>
        </nav>
      </aside>

      <main className="features-main">
        <h2 className="section-title">Key Features</h2>
        <ul className="feature-list">
          <li>
            <strong>🎯 Realistic Lip Sync:</strong> Match voice with lip
            movements for English and Urdu languages
          </li>
          <li>
            <strong>🌍 Bilingual Support:</strong> Seamless translation between
            English and Urdu languages
          </li>
          <li>
            <strong>⚡ Fast Processing:</strong> Upload and translate videos in
            under a minute
          </li>
          <li>
            <strong>🎤 AI Voice Over:</strong> Customize voice style and emotion
          </li>
          <li>
            <strong>📂 Dashboard & History:</strong> Manage and re-download all
            past translations
          </li>
        </ul>
      </main>
    </div>
  );
}
