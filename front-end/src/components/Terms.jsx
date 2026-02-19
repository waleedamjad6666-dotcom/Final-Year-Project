import React from "react";
import { Link } from "react-router-dom";
import "../styles/Terms.css";

export default function Terms() {
  return (
    <div className="terms-container">
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
          <Link to="/help">Help</Link>
          <Link to="/terms" className="active">
            Terms
          </Link>
        </nav>
      </aside>

      <main className="terms-main">
        <h2 className="section-title">Terms & Conditions</h2>
        <p>
          By using AI Video Translation, you agree to the following terms and
          conditions. You are responsible for your own content. We do not store
          videos permanently. Your usage must comply with international
          copyright and privacy laws.
        </p>
      </main>
    </div>
  );
}
