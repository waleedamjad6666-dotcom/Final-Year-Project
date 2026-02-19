import React from "react";
import { Link } from "react-router-dom";
import "../styles/Home.css";

export default function Home() {
  return (
    <div className="home-container animate-fade-in">
      <header className="navbar glass-effect">
        <nav className="nav-left">
          <Link to="/">Home</Link>
          <Link to="/features">Features</Link>
          <Link to="/pricing">Pricing</Link>
          <Link to="/help">Help</Link>
        </nav>
        <div className="nav-right">
          <Link to="/login" className="btn btn-ghost">Login</Link>
          <Link to="/signup" className="btn btn-primary">Sign Up</Link>
        </div>
      </header>

      <main className="hero-section">
        <div className="hero-text">
          <h1 className="animate-slide-up">
            Translate Videos. Sync Lips. <span className="gradient-text">Speak to the World.</span>
          </h1>
          <p className="animate-slide-up">
            AI-powered video translation with realistic lip-syncing between
            English and Urdu. Transform your content and reach bilingual audiences.
          </p>
          <div className="hero-buttons animate-slide-up">
            <Link to="/processing" className="btn btn-primary btn-lg">
              🚀 Try Demo
            </Link>
            <Link to="/upload" className="btn btn-secondary btn-lg">
              📤 Upload Your Video
            </Link>
          </div>
        </div>
        <div className="hero-video">
          <video controls className="animate-float">
            <source
              src="https://player.vimeo.com/external/341141686.sd.mp4?s=5bfb517b6e4b9512b35964c12bfb19593f026c0&profile_id=139"
              type="video/mp4"
            />
            Your browser does not support the video tag.
          </video>
          <p className="caption">🗣️ "Hola, ¿cómo estás?"</p>
        </div>
      </main>

      <section className="features">
        <div className="feature-box card animate-slide-up">
          <div className="icon">⚡</div>
          <h3>Lightning Fast Upload</h3>
          <p>Upload and translate your videos in minutes with our optimized processing pipeline.</p>
        </div>
        <div className="feature-box card animate-slide-up">
          <div className="icon">🎭</div>
          <h3>Realistic Lip Sync</h3>
          <p>Advanced AI ensures natural-looking lip movements that match the translated audio perfectly.</p>
        </div>
        <div className="feature-box card animate-slide-up">
          <div className="icon">🎙️</div>
          <h3>Voice Customization</h3>
          <p>Choose from a variety of AI-generated voices or clone your own for personalized content.</p>
        </div>
      </section>

      <footer className="footer">
        <Link to="/terms">Terms of Service</Link>
        <Link to="/privacy">Privacy Policy</Link>
        <Link to="/contact">Contact Us</Link>
      </footer>
    </div>
  );
}
