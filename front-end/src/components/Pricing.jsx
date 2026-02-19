import React from "react";
import { Link } from "react-router-dom";
import "../styles/Pricing.css";

export default function Pricing() {
  return (
    <div className="pricing-container">
      <aside className="sidebar">
        <h2 className="sidebar-title">AI VIDEO TRANSLATION</h2>
        <nav className="menu">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/upload">New Upload</Link>
          <Link to="/history">History</Link>
          <Link to="/settings">Settings</Link>
          <Link to="/pricing" className="active">
            Pricing
          </Link>
        </nav>
      </aside>

      <main className="pricing-main">
        <h2 className="section-title">Choose Your Plan</h2>
        <div className="pricing-cards">
          <div className="card">
            <h3>Free</h3>
            <p className="price">$0/month</p>
            <ul>
              <li>1 Video/Month</li>
              <li>Standard Voices</li>
              <li>Basic Lip Sync</li>
            </ul>
            <button>Select</button>
          </div>

          <div className="card highlighted">
            <h3>Pro</h3>
            <p className="price">$29/month</p>
            <ul>
              <li>20 Videos</li>
              <li>Advanced Voices</li>
              <li>Realistic Lip Sync</li>
            </ul>
            <button>Select</button>
          </div>

          <div className="card">
            <h3>Enterprise</h3>
            <p className="price">Contact Us</p>
            <ul>
              <li>Unlimited Videos</li>
              <li>Team Collaboration</li>
              <li>Premium Support</li>
            </ul>
            <button>Contact</button>
          </div>
        </div>
      </main>
    </div>
  );
}
