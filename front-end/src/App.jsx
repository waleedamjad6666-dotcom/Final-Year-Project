import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./styles/globals.css";

import Home from "./components/Home";
import TestComponent from "./components/TestComponent";
import SignUp from "./components/SignUp";
import Login from "./components/Login";
import Upload from "./components/Upload";
import Pricing from "./components/Pricing";
import Dashboard from "./components/Dashboard";
import Processing from "./components/Processing";
import Progress from "./components/Progress";
import History from "./components/History";
import Settings from "./components/Settings";
import Features from "./components/Features";
import Help from "./components/Help";
import Terms from "./components/Terms";
import Privacy from "./components/Privacy";
import Contact from "./components/Contact";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/test" element={<TestComponent />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/login" element={<Login />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/progress/:videoId" element={<Progress />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/features" element={<Features />} />
        <Route path="/help" element={<Help />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>
    </Router>
  );
}
