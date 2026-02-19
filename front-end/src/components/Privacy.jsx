import React from "react";
import { Link } from "react-router-dom";
import "../styles/Privacy.css";

export default function Privacy() {
  return (
    <div className="privacy-container">
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
          <Link to="/privacy" className="active">
            Privacy
          </Link>
        </nav>
      </aside>

      <main className="privacy-main">
        <h2 className="section-title">Privacy Policy</h2>
        <p>
          We respect your privacy. We do not share your data with third parties.
          All uploaded videos are processed temporarily and deleted after
          translation. For questions, email privacy@aivideotranslate.io
        </p>
      </main>
    </div>
  );
}
