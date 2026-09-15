import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, ShieldAlert, Cpu, Activity, BarChart2, 
  Database, GitCommit, Layers, Clock, Zap, RefreshCw, 
  CheckCircle, AlertTriangle, MessageSquare, Mail, Sparkles
} from 'lucide-react';
import { 
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, 
  Tooltip, CartesianGrid, Legend
} from 'recharts';

import TimingScopeBadge from './components/TimingScopeBadge';
import ExperimentStatusBadge from './components/ExperimentStatusBadge';
import TextRepresentationInspector from './components/TextRepresentationInspector';
import RepresentationLab from './components/RepresentationLab';
import {
  DEFAULT_REPRESENTATIONS,
  DEFAULT_PRESETS,
  DEFAULT_METRICS,
  DEFAULT_DIM_DATA,
  DEFAULT_RUNTIME_DATA,
  DEFAULT_GEOM_DATA,
  DEFAULT_TABLES_LIST,
  DEFAULT_TABLES_CONTENT
} from './data/researchFallbackData';

const DEFAULT_API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('prediction');
  const [healthStatus, setHealthStatus] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(() => localStorage.getItem('quantum_api_base') || DEFAULT_API);
  const [showApiModal, setShowApiModal] = useState(false);
  const [tempApiUrl, setTempApiUrl] = useState(apiBaseUrl);

  // Prediction State
  const [inputMode, setInputMode] = useState('single'); // 'single' | 'email'
  const [inputText, setInputText] = useState(DEFAULT_PRESETS[0].text);
  const [emailSubject, setEmailSubject] = useState(DEFAULT_PRESETS[0].subject || '');
  const [emailBody, setEmailBody] = useState(DEFAULT_PRESETS[0].body || '');
  const [selectedRepId, setSelectedRepId] = useState('tfidf');
  const [selectedDim, setSelectedDim] = useState(8);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [presets, setPresets] = useState(DEFAULT_PRESETS);
  const [representationsList, setRepresentationsList] = useState(DEFAULT_REPRESENTATIONS);

  // Research Data State (Pre-populated with Authoritative Evidence)
  const [metricsData, setMetricsData] = useState(DEFAULT_METRICS);
  const [dimData, setDimData] = useState(DEFAULT_DIM_DATA);
  const [runtimeData, setRuntimeData] = useState(DEFAULT_RUNTIME_DATA);
  const [geomData, setGeomData] = useState(DEFAULT_GEOM_DATA);
  const [tablesList, setTablesList] = useState(DEFAULT_TABLES_LIST);
  const [selectedTableId, setSelectedTableId] = useState('table_3');
  const [tableContent, setTableContent] = useState(DEFAULT_TABLES_CONTENT.table_3);
  const [isLoadingTable, setIsLoadingTable] = useState(false);

  const API_BASE = apiBaseUrl;

  // Fetch live backend updates
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => res.json())
      .then(data => setHealthStatus(data))
      .catch(() => setHealthStatus(null));

    fetch(`${API_BASE}/api/representations`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setRepresentationsList(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/samples`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setPresets(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/research/metrics`)
      .then(res => res.json())
      .then(data => {
        if (data && typeof data === 'object') setMetricsData(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/dimensionality/scaling`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setDimData(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/runtime/scaling`)
      .then(res => res.json())
      .then(data => {
        const list = Array.isArray(data) ? data : (data?.data || []);
        if (list.length > 0) setRuntimeData(list);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/geometry/diagnostics`)
      .then(res => res.json())
      .then(data => {
        if (data && typeof data === 'object') setGeomData(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/api/research/tables`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) setTablesList(data);
      })
      .catch(() => {});
  }, [API_BASE]);

  // Fetch table content on select with static fallback support
  useEffect(() => {
    if (!selectedTableId) return;
    if (DEFAULT_TABLES_CONTENT[selectedTableId]) {
      setTableContent(DEFAULT_TABLES_CONTENT[selectedTableId]);
    }
    setIsLoadingTable(true);
    fetch(`${API_BASE}/api/research/tables/${selectedTableId}`)
      .then(res => res.json())
      .then(data => {
        if (data && typeof data === 'object') setTableContent(data);
        setIsLoadingTable(false);
      })
      .catch(() => {
        setIsLoadingTable(false);
      });
  }, [selectedTableId, API_BASE]);

  // Handle live prediction
  const handlePredict = async (customPayload = null) => {
    let payload = customPayload;
    if (!payload) {
      if (inputMode === 'email') {
        if (!emailSubject.trim() && !emailBody.trim()) return;
        payload = {
          subject: emailSubject,
          body: emailBody,
          representation: selectedRepId,
          dimension: selectedDim,
          dataset: 'MeAJOR'
        };
      } else {
        if (!inputText.trim()) return;
        payload = {
          text: inputText,
          representation: selectedRepId,
          dimension: selectedDim,
          dataset: 'SMS'
        };
      }
    }

    setIsPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleSelectPreset = (preset) => {
    if (preset.dataset === 'SMS') {
      setInputMode('single');
      setInputText(preset.text);
      handlePredict({
        text: preset.text,
        representation: selectedRepId,
        dimension: selectedDim,
        dataset: 'SMS'
      });
    } else {
      setInputMode('email');
      setEmailSubject(preset.subject || '');
      setEmailBody(preset.body || preset.text);
      handlePredict({
        subject: preset.subject || '',
        body: preset.body || preset.text,
        representation: selectedRepId,
        dimension: selectedDim,
        dataset: 'MeAJOR'
      });
    }
  };

  const currentRepMeta = representationsList.find(r => r.id === selectedRepId) || {
    id: 'tfidf',
    name: 'Canonical TF-IDF + TruncatedSVD',
    is_canonical: true,
    is_sparse: true
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#f8fafc' }}>
      {/* Top Header */}
      <header className="header-clean" style={{ padding: '14px 28px' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: '#eff6ff', border: '1px solid #bfdbfe', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Cpu color="#2563eb" size={24} strokeWidth={2.2} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#0f172a' }}>
                  Quantum Text Security
                </h1>
                <span className="badge badge-blue">Exp 40 Frozen</span>
                <span className="badge badge-purple">Representation Lab V2.0</span>
              </div>
              <p style={{ fontSize: '0.9rem', color: '#64748b' }}>
                Multi-Dataset Evaluation of Representation, Geometry, Generalization & Cost
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: '6px', overflowX: 'auto', paddingBottom: '2px' }}>
            <button 
              className={`nav-tab ${activeTab === 'prediction' ? 'active' : ''}`}
              onClick={() => setActiveTab('prediction')}
            >
              <Zap size={16} /> Live Prediction
            </button>
            <button 
              className={`nav-tab ${activeTab === 'representation' ? 'active' : ''}`}
              onClick={() => setActiveTab('representation')}
            >
              <RefreshCw size={16} /> Representation Lab
            </button>
            <button 
              className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <Activity size={16} /> Findings Overview
            </button>
            <button 
              className={`nav-tab ${activeTab === 'dimensionality' ? 'active' : ''}`}
              onClick={() => setActiveTab('dimensionality')}
            >
              <Layers size={16} /> Dimensionality (2D–12D)
            </button>
            <button 
              className={`nav-tab ${activeTab === 'generalization' ? 'active' : ''}`}
              onClick={() => setActiveTab('generalization')}
            >
              <ShieldAlert size={16} /> Distribution Shift
            </button>
            <button 
              className={`nav-tab ${activeTab === 'geometry' ? 'active' : ''}`}
              onClick={() => setActiveTab('geometry')}
            >
              <GitCommit size={16} /> Geometry & Entropy
            </button>
            <button 
              className={`nav-tab ${activeTab === 'runtime' ? 'active' : ''}`}
              onClick={() => setActiveTab('runtime')}
            >
              <Clock size={16} /> Simulation Profiling
            </button>
            <button 
              className={`nav-tab ${activeTab === 'tables' ? 'active' : ''}`}
              onClick={() => setActiveTab('tables')}
            >
              <Database size={16} /> Evidence Tables
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: '1400px', margin: '0 auto', width: '100%', padding: '24px' }}>
        
        {/* VIEW 1: LIVE PREDICTION PLAYGROUND */}
        {activeTab === 'prediction' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Top Banner */}
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <span className="badge badge-purple">Inference Playground</span>
                    <span className="badge badge-blue">{selectedDim} Qubits • 2-Layer Cyclic ZZ</span>
                    <ExperimentStatusBadge status={currentRepMeta.is_canonical ? 'CANONICAL' : 'EXPLORATORY'} />
                  </div>
                  <h2 style={{ fontSize: '1.55rem', color: '#0f172a', marginBottom: '6px' }}>
                    Live Quantum vs Classical Security Text Classifier
                  </h2>
                  <p style={{ color: '#475569', fontSize: '1.02rem', maxWidth: '900px', lineHeight: '1.6' }}>
                    Select or enter a security message below to execute live comparative classification through matched {selectedDim}-dimensional representations across the <strong>Quantum Fidelity Kernel SVM</strong>, <strong>Classical Gaussian RBF SVM</strong>, and <strong>Linear SVM baseline</strong>.
                  </p>
                </div>
                <div style={{ textAlign: 'right', background: '#f8fafc', padding: '10px 16px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>Simulation Engine</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#0f172a' }}>PyTorch complex128 (CPU)</div>
                </div>
              </div>

              {/* Sample Presets Selector */}
              <div style={{ marginTop: '20px', display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.95rem', color: '#64748b', fontWeight: 600 }}>Test Presets:</span>
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset)}
                    style={{
                      background: (inputMode === 'email' ? emailSubject === preset.subject : inputText === preset.text) ? '#eff6ff' : '#ffffff',
                      border: `1px solid ${(inputMode === 'email' ? emailSubject === preset.subject : inputText === preset.text) ? '#2563eb' : '#cbd5e1'}`,
                      borderRadius: '8px',
                      padding: '7px 14px',
                      color: (inputMode === 'email' ? emailSubject === preset.subject : inputText === preset.text) ? '#1d4ed8' : '#334155',
                      fontSize: '0.9rem',
                      fontWeight: (inputMode === 'email' ? emailSubject === preset.subject : inputText === preset.text) ? 600 : 400,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {preset.category === 'Phishing / Scam' ? '🔴 ' : '🟢 '}
                    {preset.title}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Configuration & Editor Section */}
            <div className="clean-card" style={{ padding: '24px' }}>
              
              {/* Controls: Representation Selector + Dimension Selector + Input Mode Toggle */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '18px', paddingBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                
                {/* Input Format Toggle */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#334155' }}>Input Format:</span>
                  <div style={{ display: 'flex', background: '#f1f5f9', padding: '3px', borderRadius: '8px' }}>
                    <button
                      onClick={() => setInputMode('single')}
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        border: 'none',
                        background: inputMode === 'single' ? '#ffffff' : 'transparent',
                        color: inputMode === 'single' ? '#0f172a' : '#64748b',
                        fontWeight: inputMode === 'single' ? 700 : 500,
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        boxShadow: inputMode === 'single' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
                      }}
                    >
                      <MessageSquare size={14} /> SMS / Single Text
                    </button>
                    <button
                      onClick={() => setInputMode('email')}
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        border: 'none',
                        background: inputMode === 'email' ? '#ffffff' : 'transparent',
                        color: inputMode === 'email' ? '#0f172a' : '#64748b',
                        fontWeight: inputMode === 'email' ? 700 : 500,
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        boxShadow: inputMode === 'email' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
                      }}
                    >
                      <Mail size={14} /> Email (Subject + Body)
                    </button>
                  </div>
                </div>

                {/* Modular Representation Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#334155' }}>Representation:</span>
                  <select
                    value={selectedRepId}
                    onChange={(e) => setSelectedRepId(e.target.value)}
                    style={{
                      padding: '7px 12px',
                      borderRadius: '8px',
                      border: '1px solid #cbd5e1',
                      background: '#ffffff',
                      fontSize: '0.88rem',
                      fontWeight: 600,
                      color: '#0f172a',
                      cursor: 'pointer'
                    }}
                  >
                    {representationsList.map(rep => (
                      <option key={rep.id} value={rep.id}>
                        {rep.name} {rep.is_canonical ? '(Canonical)' : '(Exploratory)'}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Dimension / Qubit Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#334155' }}>Dimension:</span>
                  <select
                    value={selectedDim}
                    onChange={(e) => setSelectedDim(Number(e.target.value))}
                    style={{
                      padding: '7px 12px',
                      borderRadius: '8px',
                      border: '1px solid #cbd5e1',
                      background: '#ffffff',
                      fontSize: '0.88rem',
                      fontWeight: 600,
                      color: '#0f172a',
                      cursor: 'pointer'
                    }}
                  >
                    {[2, 4, 6, 8, 10, 12].map(d => (
                      <option key={d} value={d}>{d}D ({d} Qubits)</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Text Input Fields */}
              {inputMode === 'email' ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.92rem', fontWeight: 600, marginBottom: '6px', color: '#0f172a' }}>
                      Email Subject:
                    </label>
                    <input
                      type="text"
                      className="textarea-custom"
                      style={{ minHeight: '42px', padding: '10px 14px' }}
                      value={emailSubject}
                      onChange={(e) => setEmailSubject(e.target.value)}
                      placeholder="e.g., URGENT: Security notification regarding your payroll account"
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.92rem', fontWeight: 600, marginBottom: '6px', color: '#0f172a' }}>
                      Email Body:
                    </label>
                    <textarea
                      className="textarea-custom"
                      value={emailBody}
                      onChange={(e) => setEmailBody(e.target.value)}
                      placeholder="Enter email message body text..."
                      rows={4}
                    />
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic' }}>
                    * Combined according to paper preprocessing protocol: <code>Subject: {'<subject>'}\n\n{'<body>'}</code>
                  </div>
                </div>
              ) : (
                <div>
                  <label style={{ display: 'block', fontSize: '1.05rem', fontWeight: 600, marginBottom: '8px', color: '#0f172a' }}>
                    Message Text (SMS or Raw Text):
                  </label>
                  <textarea
                    className="textarea-custom"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter message text to classify (e.g., SMS alerts, phishing links, or normal messages)..."
                    rows={4}
                  />
                </div>
              )}

              {/* Action Button & Metadata */}
              <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ fontSize: '0.92rem', color: '#64748b' }}>
                  Encoder: <strong>{currentRepMeta.name}</strong> • Target: <strong>{selectedDim} Qubits</strong>
                </div>
                <button
                  className="btn-primary"
                  onClick={() => handlePredict()}
                  disabled={isPredicting || (inputMode === 'email' ? (!emailSubject.trim() && !emailBody.trim()) : !inputText.trim())}
                  style={{ opacity: isPredicting ? 0.7 : 1 }}
                >
                  {isPredicting ? (
                    <>
                      <RefreshCw size={18} /> Simulating Quantum States...
                    </>
                  ) : (
                    <>
                      <Zap size={18} /> Run Multi-Model Inference
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Live Prediction Output Cards */}
            {predictionResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
                    Comparative Inference Results
                  </h3>
                  <TimingScopeBadge scope="live" />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
                  
                  {/* 1. Quantum Fidelity Kernel Card */}
                  <div className="clean-panel-interactive" style={{ padding: '22px', borderLeft: '5px solid #2563eb' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Cpu color="#2563eb" size={22} />
                        <h3 style={{ fontSize: '1.15rem', color: '#0f172a' }}>Quantum Fidelity Kernel</h3>
                      </div>
                      <span className="badge badge-blue">{predictionResult.dimension} Qubits</span>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '0.82rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.04em' }}>Prediction</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {predictionResult.models.quantum_fidelity_kernel.is_malicious ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#dc2626', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldAlert size={22} /> MALICIOUS / PHISHING
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#059669', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldCheck size={22} /> LEGITIMATE / HAM
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Decision Score & Probability */}
                    <div style={{ marginBottom: '14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                        <span style={{ color: '#475569' }}>Estimated Probability:</span>
                        <span style={{ fontWeight: 700, color: '#0f172a' }}>
                          {(predictionResult.models.quantum_fidelity_kernel.estimated_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${predictionResult.models.quantum_fidelity_kernel.estimated_probability * 100}%`, 
                            height: '100%', 
                            background: predictionResult.models.quantum_fidelity_kernel.is_malicious ? '#dc2626' : '#059669',
                            transition: 'width 0.4s ease'
                          }} 
                        />
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                      <div>
                        <span style={{ color: '#64748b' }}>Decision Margin: </span>
                        <strong style={{ color: '#0f172a' }}>{predictionResult.models.quantum_fidelity_kernel.decision_score}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#64748b' }}>Live Latency: </span>
                        <strong style={{ color: '#2563eb' }}>{predictionResult.models.quantum_fidelity_kernel.latency_ms} ms</strong>
                      </div>
                    </div>
                  </div>

                  {/* 2. Classical Gaussian RBF Card */}
                  <div className="clean-panel-interactive" style={{ padding: '22px', borderLeft: '5px solid #7c3aed' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity color="#7c3aed" size={22} />
                        <h3 style={{ fontSize: '1.15rem', color: '#0f172a' }}>Classical Gaussian RBF</h3>
                      </div>
                      <span className="badge badge-purple">Matched {predictionResult.dimension}D</span>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '0.82rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.04em' }}>Prediction</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {predictionResult.models.classical_rbf.is_malicious ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#dc2626', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldAlert size={22} /> MALICIOUS / PHISHING
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#059669', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldCheck size={22} /> LEGITIMATE / HAM
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Decision Score & Probability */}
                    <div style={{ marginBottom: '14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                        <span style={{ color: '#475569' }}>Estimated Probability:</span>
                        <span style={{ fontWeight: 700, color: '#0f172a' }}>
                          {(predictionResult.models.classical_rbf.estimated_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${predictionResult.models.classical_rbf.estimated_probability * 100}%`, 
                            height: '100%', 
                            background: predictionResult.models.classical_rbf.is_malicious ? '#dc2626' : '#059669',
                            transition: 'width 0.4s ease'
                          }} 
                        />
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                      <div>
                        <span style={{ color: '#64748b' }}>Decision Margin: </span>
                        <strong style={{ color: '#0f172a' }}>{predictionResult.models.classical_rbf.decision_score}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#64748b' }}>Live Latency: </span>
                        <strong style={{ color: '#7c3aed' }}>{predictionResult.models.classical_rbf.latency_ms} ms</strong>
                      </div>
                    </div>
                  </div>

                  {/* 3. Linear SVM Baseline Card */}
                  <div className="clean-panel-interactive" style={{ padding: '22px', borderLeft: '5px solid #64748b' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <BarChart2 color="#64748b" size={22} />
                        <h3 style={{ fontSize: '1.15rem', color: '#0f172a' }}>Linear SVM Baseline</h3>
                      </div>
                      <span className="badge badge-gray">{predictionResult.dimension}D Subspace</span>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '0.82rem', color: '#64748b', textTransform: 'uppercase', marginBottom: '4px', letterSpacing: '0.04em' }}>Prediction</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {predictionResult.models.linear_svm.is_malicious ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#dc2626', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldAlert size={22} /> MALICIOUS / PHISHING
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#059669', fontWeight: 700, fontSize: '1.25rem' }}>
                            <ShieldCheck size={22} /> LEGITIMATE / HAM
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Decision Score & Probability */}
                    <div style={{ marginBottom: '14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                        <span style={{ color: '#475569' }}>Estimated Probability:</span>
                        <span style={{ fontWeight: 700, color: '#0f172a' }}>
                          {(predictionResult.models.linear_svm.estimated_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${predictionResult.models.linear_svm.estimated_probability * 100}%`, 
                            height: '100%', 
                            background: predictionResult.models.linear_svm.is_malicious ? '#dc2626' : '#059669',
                            transition: 'width 0.4s ease'
                          }} 
                        />
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                      <div>
                        <span style={{ color: '#64748b' }}>Decision Margin: </span>
                        <strong style={{ color: '#0f172a' }}>{predictionResult.models.linear_svm.decision_score}</strong>
                      </div>
                      <div>
                        <span style={{ color: '#64748b' }}>Live Latency: </span>
                        <strong style={{ color: '#475569' }}>{predictionResult.models.linear_svm.latency_ms} ms</strong>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Text-Level Representation Pipeline Inspector */}
                <TextRepresentationInspector 
                  traceData={predictionResult.feature_trace} 
                  representationMeta={predictionResult.representation}
                  dimension={predictionResult.dimension}
                />
              </div>
            )}
          </div>
        )}

        {/* VIEW 2: MODULAR REPRESENTATION LAB */}
        {activeTab === 'representation' && (
          <RepresentationLab 
            currentRepId={selectedRepId} 
            onSelectRepresentation={(id) => setSelectedRepId(id)} 
          />
        )}

        {/* VIEW 3: RESEARCH DASHBOARD / EXECUTIVE FINDINGS */}
        {activeTab === 'dashboard' && metricsData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '10px' }}>
                <span className="badge badge-blue">Paper Findings Synthesis</span>
                <TimingScopeBadge scope="benchmark" />
              </div>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Principal Finding: Parity under IID, Degradation under Source Shift
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Under controlled matched conditions across 10 independent computational seeds, parameter-free quantum fidelity kernels (2-layer cyclic ZZFeatureMap) achieve in-distribution parity with classical Gaussian RBF kernels, but do not provide a consistent practical advantage (ε = 0.01 F1), suffer significant degradation under cross-source domain shift, and incur a ~64× classical simulation penalty at 12D.
              </p>
            </div>

            {/* Core KPI Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <div style={{ fontSize: '0.88rem', color: '#64748b', marginBottom: '4px' }}>MeAJOR IID (8D / 8 Qubits)</div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: '#2563eb', marginBottom: '6px' }}>
                  +0.46 pp <span style={{ fontSize: '0.95rem', color: '#64748b', fontWeight: 600 }}>(ΔF1 = +0.0046)</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: '#059669', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '4px' }}>
                  <CheckCircle size={15} /> p = 0.0016 • 95% CI [+0.0030, +0.0061]
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  Within practical-equivalence margin (ε = 0.01)
                </div>
              </div>

              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <div style={{ fontSize: '0.88rem', color: '#64748b', marginBottom: '4px' }}>MeAJOR IID (12D / 12 Qubits)</div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: '#0f172a', marginBottom: '6px' }}>
                  +0.14 pp <span style={{ fontSize: '0.95rem', color: '#64748b', fontWeight: 600 }}>(ΔF1 = +0.0014)</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: '#475569', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '4px' }}>
                  <CheckCircle size={15} /> p = 0.2824 • 95% CI [-0.0010, +0.0037]
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  Complete statistical and practical parity
                </div>
              </div>

              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <div style={{ fontSize: '0.88rem', color: '#64748b', marginBottom: '4px' }}>Cross-Source Holdout (Direction B)</div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: '#dc2626', marginBottom: '6px' }}>
                  -2.33 pp <span style={{ fontSize: '0.95rem', color: '#64748b', fontWeight: 600 }}>(ΔF1 = -0.0233)</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: '#dc2626', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '4px' }}>
                  <AlertTriangle size={15} /> p = 0.0046 (BH FDR p = 0.0069)
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  Exceeds practical threshold; classical RBF is superior
                </div>
              </div>

              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <div style={{ fontSize: '0.88rem', color: '#64748b', marginBottom: '4px' }}>Classical Simulation Overhead (12D)</div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: '#d97706', marginBottom: '6px' }}>
                  64.0× <span style={{ fontSize: '0.95rem', color: '#64748b', fontWeight: 600 }}>(108.8s vs 1.7s)</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: '#d97706', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '4px' }}>
                  <Clock size={15} /> Peak RAM: 6.4 GB vs 122 MB
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  Unfavorable engineering trade-off in simulation
                </div>
              </div>
            </div>

            {/* Key Comparison Table */}
            <div className="clean-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '14px' }}>
                Controlled 10-Seed Confirmation Summary (Exp 40)
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.95rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569', background: '#f8fafc' }}>
                      <th style={{ padding: '12px 14px' }}>Evaluation Setting</th>
                      <th style={{ padding: '12px 14px' }}>Quantum F1</th>
                      <th style={{ padding: '12px 14px' }}>Classical RBF F1</th>
                      <th style={{ padding: '12px 14px' }}>Paired ΔF1</th>
                      <th style={{ padding: '12px 14px' }}>95% Bootstrap CI</th>
                      <th style={{ padding: '12px 14px' }}>Permutation p</th>
                      <th style={{ padding: '12px 14px' }}>Status vs ε = 0.01</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>MeAJOR IID (8D)</td>
                      <td style={{ padding: '12px 14px', color: '#2563eb', fontWeight: 600 }}>0.8754 ± 0.0029</td>
                      <td style={{ padding: '12px 14px', color: '#7c3aed', fontWeight: 600 }}>0.8709 ± 0.0030</td>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#059669' }}>+0.0046</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>[+0.0030, +0.0061]</td>
                      <td style={{ padding: '12px 14px' }}>0.0016</td>
                      <td style={{ padding: '12px 14px' }}><span className="badge badge-emerald">Practically Equivalent</span></td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>MeAJOR IID (10D)</td>
                      <td style={{ padding: '12px 14px', color: '#2563eb', fontWeight: 600 }}>0.9023 ± 0.0034</td>
                      <td style={{ padding: '12px 14px', color: '#7c3aed', fontWeight: 600 }}>0.8967 ± 0.0049</td>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#059669' }}>+0.0057</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>[+0.0032, +0.0081]</td>
                      <td style={{ padding: '12px 14px' }}>0.0052</td>
                      <td style={{ padding: '12px 14px' }}><span className="badge badge-emerald">Practically Equivalent</span></td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>MeAJOR IID (12D)</td>
                      <td style={{ padding: '12px 14px', color: '#2563eb', fontWeight: 600 }}>0.9137 ± 0.0046</td>
                      <td style={{ padding: '12px 14px', color: '#7c3aed', fontWeight: 600 }}>0.9123 ± 0.0023</td>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#64748b' }}>+0.0014</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>[-0.0010, +0.0037]</td>
                      <td style={{ padding: '12px 14px' }}>0.2824</td>
                      <td style={{ padding: '12px 14px' }}><span className="badge badge-gray">Parity</span></td>
                    </tr>
                    <tr>
                      <td style={{ padding: '12px 14px', fontWeight: 600, color: '#0f172a' }}>Direction B (TREC 2007 → 2005/06)</td>
                      <td style={{ padding: '12px 14px', color: '#2563eb', fontWeight: 600 }}>0.6680 ± 0.0094</td>
                      <td style={{ padding: '12px 14px', color: '#7c3aed', fontWeight: 600 }}>0.6913 ± 0.0161</td>
                      <td style={{ padding: '12px 14px', fontWeight: 700, color: '#dc2626' }}>-0.0233</td>
                      <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>[-0.0353, -0.0117]</td>
                      <td style={{ padding: '12px 14px' }}>0.0046</td>
                      <td style={{ padding: '12px 14px' }}><span className="badge badge-rose">Classical Advantage</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 4: DIMENSIONALITY SCALING EXPLORER */}
        {activeTab === 'dimensionality' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '10px' }}>
                <span className="badge badge-purple">Bottleneck Resolution</span>
                <TimingScopeBadge scope="benchmark" />
              </div>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Dimensionality Scaling Trajectory (2D to 12D)
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Scaling representation dimension from 2D to 12D produces a monotonic +41.7% relative gain (0.6447 → 0.9137 F1), converging to parity with classical RBF at 12D. This indicates that early low-dimensional weakness was strongly associated with information bottleneck caused by TruncatedSVD compression rather than a failure of quantum geometry.
              </p>
            </div>

            <div className="clean-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '16px' }}>
                Test F1 Score vs Reduced Representation Dimension (MeAJOR Benchmark)
              </h3>
              <div style={{ height: '360px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={dimData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="dim" stroke="#475569" fontSize={13} fontWeight={500} />
                    <YAxis stroke="#475569" fontSize={13} domain={[0.6, 0.95]} tickFormatter={(v) => v.toFixed(2)} />
                    <Tooltip 
                      contentStyle={{ background: '#ffffff', borderColor: '#cbd5e1', borderRadius: '8px', color: '#0f172a', fontSize: '0.95rem', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }} 
                      formatter={(val, name) => [Number(val).toFixed(4), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.95rem', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="quantum_f1" name="Quantum Fidelity Kernel" stroke="#2563eb" strokeWidth={3} dot={{ r: 6 }} />
                    <Line type="monotone" dataKey="rbf_f1" name="Matched Classical RBF" stroke="#7c3aed" strokeWidth={3} dot={{ r: 6 }} />
                    <Line type="monotone" dataKey="linear_f1" name="Linear SVM baseline" stroke="#64748b" strokeWidth={2.2} strokeDasharray="4 4" dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 5: CROSS-SOURCE GENERALIZATION */}
        {activeTab === 'generalization' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '10px' }}>
                <span className="badge badge-rose">Domain Shift Evaluation</span>
                <TimingScopeBadge scope="benchmark" />
              </div>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Cross-Source Domain Holdout (Direction B)
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Evaluating models trained on TREC 2007 and tested on TREC 2005/2006 reveals severe vocabulary drift (&gt;92% type OOV and &gt;34% token OOV). The quantum kernel degrades by 23.7% (F1 = 0.6680), underperforming classical RBF (F1 = 0.6913) by Δ = -0.0233 (p = 0.0046).
              </p>
            </div>

            <div className="clean-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '16px' }}>
                Linear SVM Vocabulary Recovery Scaling across Dimensions
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                <div className="clean-panel-interactive" style={{ padding: '18px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.88rem', color: '#64748b' }}>8D TruncatedSVD</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#dc2626', margin: '4px 0' }}>0.7562 F1</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>-12.75 pp vs full</div>
                </div>
                <div className="clean-panel-interactive" style={{ padding: '18px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.88rem', color: '#64748b' }}>16D TruncatedSVD</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#d97706', margin: '4px 0' }}>0.8125 F1</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>-7.12 pp vs full</div>
                </div>
                <div className="clean-panel-interactive" style={{ padding: '18px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.88rem', color: '#64748b' }}>32D TruncatedSVD</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#059669', margin: '4px 0' }}>0.8761 F1</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>-0.76 pp vs full</div>
                </div>
                <div className="clean-panel-interactive" style={{ padding: '18px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.88rem', color: '#64748b' }}>Full 50k TF-IDF</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#2563eb', margin: '4px 0' }}>0.8837 F1</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>Full-dimensional TF-IDF baseline</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 6: GEOMETRY & ENTROPY */}
        {activeTab === 'geometry' && geomData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '10px' }}>
                <span className="badge badge-purple">Geometric Profiling</span>
                <TimingScopeBadge scope="benchmark" />
              </div>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Decoupling Single-State Entropy and Pairwise Kernel Diversity
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Regressions demonstrate a strong inverse correlation (r = -0.78 to -0.83) between single-state von Neumann entropy and pairwise kernel diversity. Spreading statevectors across basis states is strongly inversely associated with pairwise kernel diversity, refuting the heuristic that greater single-state dispersion yields better classification.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <h3 style={{ fontSize: '1.2rem', color: '#0f172a', marginBottom: '14px' }}>
                  Kernel-Target Label Alignment Deficit
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {geomData.target_label_alignment.map((item, idx) => (
                    <div key={idx} style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '6px', color: '#0f172a' }}>{item.corpus}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem', color: '#475569' }}>
                        <span>Quantum: <strong style={{ color: '#2563eb' }}>{item.quantum}</strong></span>
                        <span>Classical RBF: <strong style={{ color: '#7c3aed' }}>{item.classical_rbf}</strong></span>
                        <span style={{ color: '#dc2626', fontWeight: 700 }}>{item.deficit_pct}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="clean-panel-interactive" style={{ padding: '22px' }}>
                <h3 style={{ fontSize: '1.2rem', color: '#0f172a', marginBottom: '14px' }}>
                  Entropy vs Pairwise Diversity Correlation
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {geomData.entropy_diversity_correlation.map((item, idx) => (
                    <div key={idx} style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '6px', color: '#0f172a' }}>{item.corpus}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.92rem' }}>
                        <span style={{ color: '#475569' }}>Pearson Correlation (r):</span>
                        <strong style={{ color: '#dc2626', fontFamily: 'var(--font-mono)' }}>{item.correlation_r}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 7: RUNTIME & SCALABILITY PROFILING */}
        {activeTab === 'runtime' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '10px' }}>
                <span className="badge badge-amber">Simulation Profiling</span>
                <TimingScopeBadge scope="benchmark" />
              </div>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Computational Execution Time and Memory Scaling
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Classical statevector simulation of quantum kernels scales steeply with dimensionality. At 12 dimensions, computing the quantum kernel requires 108.8s compared to 1.7s for RBF (64.0× ratio), while 16D encounters an unsustainable &gt;10.5 GB memory ceiling on 10,000 samples.
              </p>
            </div>

            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                <h3 style={{ fontSize: '1.25rem', color: '#0f172a', margin: 0 }}>
                  Kernel Matrix Computation Time (Seconds) vs Dimensions (10,000 Samples)
                </h3>
              </div>
              <div style={{ height: '360px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={runtimeData.filter(d => d.quantum_s !== null)} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="dim" stroke="#475569" fontSize={13} fontWeight={500} />
                    <YAxis stroke="#475569" domain={[0, 'auto']} fontSize={13} tickFormatter={(v) => `${v}s`} />
                    <Tooltip 
                      contentStyle={{ background: '#ffffff', borderColor: '#cbd5e1', borderRadius: '8px', color: '#0f172a', fontSize: '0.95rem', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                      formatter={(val, name) => [`${val}s`, name]}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.95rem', paddingTop: '10px' }} />
                    <Bar dataKey="quantum_s" name="Quantum Fidelity Kernel (PyTorch CPU)" fill="#2563eb" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="rbf_s" name="Matched Classical RBF" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Profile Comparison Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
              <div className="clean-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '6px' }}>
                  Execution Ratio (12D)
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#d97706', marginBottom: '6px' }}>
                  64.0× Slower
                </div>
                <div style={{ fontSize: '0.92rem', color: '#475569' }}>
                  Quantum requires 108.8s vs 1.7s for Classical RBF at 12 qubits/dimensions.
                </div>
              </div>

              <div className="clean-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '6px' }}>
                  Peak Memory (12D vs 16D)
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#dc2626', marginBottom: '6px' }}>
                  6.4 GB → &gt;10.5 GB
                </div>
                <div style={{ fontSize: '0.92rem', color: '#475569' }}>
                  Memory explodes exponentially with statevector dimension (2^n complex numbers).
                </div>
              </div>

              <div className="clean-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '6px' }}>
                  Classical RBF Memory
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#059669', marginBottom: '6px' }}>
                  122 MB (Constant)
                </div>
                <div style={{ fontSize: '0.92rem', color: '#475569' }}>
                  Scales linearly with feature dimensions, avoiding statevector representation explosion.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 8: EVIDENCE TABLES */}
        {activeTab === 'tables' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <span className="badge badge-blue">Paper Artifacts</span>
                    <ExperimentStatusBadge status="CANONICAL" />
                  </div>
                  <h2 style={{ fontSize: '1.55rem', color: '#0f172a' }}>Authoritative Research Tables</h2>
                  <p style={{ color: '#64748b', fontSize: '0.98rem' }}>
                    Verified numerical evidence from Experiments 30 to 40.
                  </p>
                </div>
                
                {/* Table selector buttons */}
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {tablesList.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => setSelectedTableId(t.id)}
                      style={{
                        padding: '8px 14px',
                        borderRadius: '8px',
                        border: `1px solid ${selectedTableId === t.id ? '#2563eb' : '#cbd5e1'}`,
                        background: selectedTableId === t.id ? '#eff6ff' : '#ffffff',
                        color: selectedTableId === t.id ? '#1d4ed8' : '#475569',
                        fontWeight: selectedTableId === t.id ? 700 : 500,
                        fontSize: '0.9rem',
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      {t.id.replace('_', ' ').toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {tableContent && (
              <div className="clean-card" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                  <h3 style={{ fontSize: '1.3rem', color: '#0f172a' }}>{tableContent.title}</h3>
                  <TimingScopeBadge scope="benchmark" />
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table className="table-clean" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
                    <thead>
                      <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                        {tableContent.columns.map((col, idx) => (
                          <th key={idx} style={{ padding: '12px 14px', fontWeight: 600, color: '#334155' }}>
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {tableContent.rows.map((row, rIdx) => (
                        <tr key={rIdx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          {tableContent.columns.map((col, cIdx) => (
                            <td key={cIdx} style={{ padding: '12px 14px', color: '#1e293b' }}>
                              {typeof row[col] === 'number' ? row[col].toFixed(4) : (row[col] ?? '—')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}
