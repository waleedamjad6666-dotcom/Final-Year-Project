import React, { useState } from "react";
import { Link } from "react-router-dom";
import "../styles/Settings.css";

export default function Settings() {
  const [username, setUsername] = useState("john_doe");
  const [email, setEmail] = useState("john@example.com");
  const [language, setLanguage] = useState("English");

  return (
    <div className="settings-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history">History</Link>
          <Link to="/settings" className="active">
            Settings
          </Link>
        </nav>
      </aside>

      <main className="settings-main">
        <h2 className="section-title">User Settings</h2>
        <form className="settings-form">
          <label>Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />

          <label>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <label>Preferred Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option>English</option>
            <option>Urdu</option>
          </select>

          <button className="btn">Save Changes</button>
        </form>
      </main>
    </div>
  );
}
