import React from "react";
import { Link } from "react-router-dom";
import "../styles/Contact.css";

export default function Contact() {
  return (
    <div className="contact-container">
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
          <Link to="/contact" className="active">
            Contact
          </Link>
        </nav>
      </aside>

      <main className="contact-main">
        <h2 className="section-title">Contact Us</h2>
        <p>If you have any questions or feedback, feel free to reach out:</p>
        <p>Email: support@aivideotranslate.io</p>
        <p>Phone: +1 (234) 567-8901</p>
        <p>Address: 123 Translation Ave, Cloud City, Web</p>
      </main>
    </div>
  );
}
