import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, ShieldAlert, Cpu, Activity, BarChart2, 
  Database, GitCommit, Layers, Clock, Zap, RefreshCw, 
  CheckCircle, AlertTriangle
} from 'lucide-react';
import { 
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, 
  Tooltip, CartesianGrid, Legend
} from 'recharts';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('prediction');
  const [healthStatus, setHealthStatus] = useState(null);

  // Prediction State
  const [inputText, setInputText] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [presets, setPresets] = useState([]);

  // Research Data State
  const [metricsData, setMetricsData] = useState(null);
  const [dimData, setDimData] = useState([]);
  const [runtimeData, setRuntimeData] = useState([]);
  const [geomData, setGeomData] = useState(null);
  const [tablesList, setTablesList] = useState([]);
  const [selectedTableId, setSelectedTableId] = useState('table_3');
  const [tableContent, setTableContent] = useState(null);
  const [isLoadingTable, setIsLoadingTable] = useState(false);

  // Fetch initial data
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => res.json())
      .then(data => setHealthStatus(data))
      .catch(err => console.error("Health check error:", err));

    fetch(`${API_BASE}/api/samples`)
      .then(res => res.json())
      .then(data => {
        setPresets(data);
        if (data.length > 0) {
          setInputText(data[0].text);
        }
      })
      .catch(err => console.error("Presets error:", err));

    fetch(`${API_BASE}/api/research/metrics`)
      .then(res => res.json())
      .then(data => setMetricsData(data))
      .catch(err => console.error("Metrics error:", err));

    fetch(`${API_BASE}/api/dimensionality/scaling`)
      .then(res => res.json())
      .then(data => setDimData(data))
      .catch(err => console.error("Dim scaling error:", err));

    fetch(`${API_BASE}/api/runtime/scaling`)
      .then(res => res.json())
      .then(data => setRuntimeData(data))
      .catch(err => console.error("Runtime scaling error:", err));

    fetch(`${API_BASE}/api/geometry/diagnostics`)
      .then(res => res.json())
      .then(data => setGeomData(data))
      .catch(err => console.error("Geometry diagnostics error:", err));

    fetch(`${API_BASE}/api/research/tables`)
      .then(res => res.json())
      .then(data => setTablesList(data))
      .catch(err => console.error("Tables list error:", err));
  }, []);

  // Fetch table content on select
  useEffect(() => {
    if (!selectedTableId) return;
    setIsLoadingTable(true);
    fetch(`${API_BASE}/api/research/tables/${selectedTableId}`)
      .then(res => res.json())
      .then(data => {
        setTableContent(data);
        setIsLoadingTable(false);
      })
      .catch(err => {
        console.error("Table fetch error:", err);
        setIsLoadingTable(false);
      });
  }, [selectedTableId]);

  // Handle live prediction
  const handlePredict = async (textToPredict = inputText) => {
    if (!textToPredict.trim()) return;
    setIsPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToPredict })
      });
      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsPredicting(false);
    }
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
              className={`nav-tab ${activeTab === 'representation' ? 'active' : ''}`}
              onClick={() => setActiveTab('representation')}
            >
              <RefreshCw size={16} /> Representation
            </button>
            <button 
              className={`nav-tab ${activeTab === 'generalization' ? 'active' : ''}`}
              onClick={() => setActiveTab('generalization')}
            >
              <ShieldAlert size={16} /> Domain Shift
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
                    <span className="badge badge-blue">8 Qubits • 2-Layer Cyclic ZZ</span>
                  </div>
                  <h2 style={{ fontSize: '1.55rem', color: '#0f172a', marginBottom: '6px' }}>
                    Live Quantum vs Classical Security Text Classifier
                  </h2>
                  <p style={{ color: '#475569', fontSize: '1.02rem', maxWidth: '900px', lineHeight: '1.6' }}>
                    Select or enter an email/SMS message below to execute live comparative classification through matched 8-dimensional representations across the <strong>Quantum Fidelity Kernel SVM</strong>, <strong>Classical Gaussian RBF SVM</strong>, and <strong>Contextual Linear SVM</strong>.
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
                    onClick={() => {
                      setInputText(preset.text);
                      handlePredict(preset.text);
                    }}
                    style={{
                      background: inputText === preset.text ? '#eff6ff' : '#ffffff',
                      border: `1px solid ${inputText === preset.text ? '#2563eb' : '#cbd5e1'}`,
                      borderRadius: '8px',
                      padding: '7px 14px',
                      color: inputText === preset.text ? '#1d4ed8' : '#334155',
                      fontSize: '0.9rem',
                      fontWeight: inputText === preset.text ? 600 : 400,
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

            {/* Input and Action Section */}
            <div className="clean-card" style={{ padding: '24px' }}>
              <label style={{ display: 'block', fontSize: '1.05rem', fontWeight: 600, marginBottom: '8px', color: '#0f172a' }}>
                Message Text (Email Subject + Body or SMS):
              </label>
              <textarea
                className="textarea-custom"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to classify (e.g., security alerts, prize notifications, or corporate emails)..."
                rows={4}
              />
              <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                <div style={{ fontSize: '0.92rem', color: '#64748b' }}>
                  Characters: <strong>{inputText.length}</strong> • Words: <strong>{inputText.split(/\s+/).filter(Boolean).length}</strong>
                </div>
                <button
                  className="btn-primary"
                  onClick={() => handlePredict()}
                  disabled={isPredicting || !inputText.trim()}
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

            {/* Live Prediction Cards */}
            {predictionResult && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
                {/* 1. Quantum Fidelity Kernel Card */}
                <div className="clean-panel-interactive" style={{ padding: '22px', borderLeft: '5px solid #2563eb' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Cpu color="#2563eb" size={22} />
                      <h3 style={{ fontSize: '1.15rem', color: '#0f172a' }}>Quantum Fidelity Kernel</h3>
                    </div>
                    <span className="badge badge-blue">8 Qubits</span>
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

                  {/* Probability Bar */}
                  <div style={{ marginBottom: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                      <span style={{ color: '#475569' }}>Malicious Probability:</span>
                      <span style={{ fontWeight: 700, color: '#0f172a' }}>
                        {(predictionResult.models.quantum_fidelity_kernel.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${predictionResult.models.quantum_fidelity_kernel.probability * 100}%`, 
                          height: '100%', 
                          background: predictionResult.models.quantum_fidelity_kernel.is_malicious ? '#dc2626' : '#059669',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                    <div>
                      <span style={{ color: '#64748b' }}>Decision Score: </span>
                      <strong style={{ color: '#0f172a' }}>{predictionResult.models.quantum_fidelity_kernel.decision_score}</strong>
                    </div>
                    <div>
                      <span style={{ color: '#64748b' }}>Latency: </span>
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
                    <span className="badge badge-purple">Matched 8D</span>
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

                  {/* Probability Bar */}
                  <div style={{ marginBottom: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                      <span style={{ color: '#475569' }}>Malicious Probability:</span>
                      <span style={{ fontWeight: 700, color: '#0f172a' }}>
                        {(predictionResult.models.classical_rbf.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${predictionResult.models.classical_rbf.probability * 100}%`, 
                          height: '100%', 
                          background: predictionResult.models.classical_rbf.is_malicious ? '#dc2626' : '#059669',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                    <div>
                      <span style={{ color: '#64748b' }}>Decision Score: </span>
                      <strong style={{ color: '#0f172a' }}>{predictionResult.models.classical_rbf.decision_score}</strong>
                    </div>
                    <div>
                      <span style={{ color: '#64748b' }}>Latency: </span>
                      <strong style={{ color: '#7c3aed' }}>{predictionResult.models.classical_rbf.latency_ms} ms</strong>
                    </div>
                  </div>
                </div>

                {/* 3. Contextual Linear SVM Card */}
                <div className="clean-panel-interactive" style={{ padding: '22px', borderLeft: '5px solid #64748b' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <BarChart2 color="#64748b" size={22} />
                      <h3 style={{ fontSize: '1.15rem', color: '#0f172a' }}>Contextual Linear SVM</h3>
                    </div>
                    <span className="badge badge-gray">8D SVD</span>
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

                  {/* Probability Bar */}
                  <div style={{ marginBottom: '14px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', marginBottom: '5px' }}>
                      <span style={{ color: '#475569' }}>Malicious Probability:</span>
                      <span style={{ fontWeight: 700, color: '#0f172a' }}>
                        {(predictionResult.models.linear_svm.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${predictionResult.models.linear_svm.probability * 100}%`, 
                          height: '100%', 
                          background: predictionResult.models.linear_svm.is_malicious ? '#dc2626' : '#059669',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                    <div>
                      <span style={{ color: '#64748b' }}>Decision Score: </span>
                      <strong style={{ color: '#0f172a' }}>{predictionResult.models.linear_svm.decision_score}</strong>
                    </div>
                    <div>
                      <span style={{ color: '#64748b' }}>Latency: </span>
                      <strong style={{ color: '#475569' }}>{predictionResult.models.linear_svm.latency_ms} ms</strong>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Quantum Statevector Amplitudes Visualization */}
            {predictionResult && (
              <div className="clean-card" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Zap color="#2563eb" size={20} /> Quantum Statevector Amplitude Distribution
                    </h3>
                    <p style={{ fontSize: '0.92rem', color: '#64748b', marginTop: '3px' }}>
                      Exact double-precision statevector |ψ(x)⟩ across 256 complex basis states (8 qubits).
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <span className="badge badge-blue">Entropy: {predictionResult.quantum_diagnostics.state_entropy_bits} bits</span>
                    <span className="badge badge-purple">Hilbert Dim: {predictionResult.quantum_diagnostics.hilbert_dimension}</span>
                  </div>
                </div>

                <div style={{ height: '240px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={predictionResult.quantum_diagnostics.top_basis_probabilities} margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="basis" stroke="#475569" fontSize={13} fontWeight={600} fontFamily="var(--font-mono)" />
                      <YAxis stroke="#475569" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip 
                        contentStyle={{ background: '#ffffff', borderColor: '#cbd5e1', borderRadius: '8px', color: '#0f172a', fontSize: '0.95rem', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                        formatter={(val) => [`${(val * 100).toFixed(2)}%`, 'Basis Measurement Probability']}
                      />
                      <Bar dataKey="probability" fill="#2563eb" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* VIEW 2: RESEARCH DASHBOARD / EXECUTIVE FINDINGS */}
        {activeTab === 'dashboard' && metricsData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <span className="badge badge-blue" style={{ marginBottom: '10px' }}>Paper Findings Synthesis</span>
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

        {/* VIEW 3: DIMENSIONALITY SCALING EXPLORER */}
        {activeTab === 'dimensionality' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <span className="badge badge-purple" style={{ marginBottom: '10px' }}>Bottleneck Resolution</span>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Dimensionality Scaling Trajectory (2D to 12D)
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Scaling representation dimension from 2D to 12D produces a monotonic +41.7% relative gain (0.6447 → 0.9137 F1), converging to parity with classical RBF at 12D. This indicates that early low-dimensional weakness was an information bottleneck caused by TruncatedSVD compression rather than a failure of quantum geometry.
              </p>
            </div>

            <div className="clean-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '16px' }}>
                Test F1 Score vs Reduced Representation Dimension
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
                    <Line type="monotone" dataKey="linear_f1" name="Contextual Linear SVM" stroke="#64748b" strokeWidth={2.2} strokeDasharray="4 4" dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 4: REPRESENTATION ABLATION */}
        {activeTab === 'representation' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <span className="badge badge-amber" style={{ marginBottom: '10px' }}>Representation Dominance</span>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Representation Ranking Inversion on CEAS 2008
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Switching upstream representation from 8D TF-IDF to 8D RoBERTa inverts the relative performance ranking between quantum and classical kernels by a net margin of <strong>3.90 percentage points</strong> (+0.95 pp on TF-IDF vs -2.95 pp on RoBERTa). Representation choice dominates kernel choice by an order of magnitude.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
              <div className="clean-panel-interactive" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '1.2rem', color: '#2563eb', marginBottom: '14px' }}>8D TF-IDF + TruncatedSVD</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1rem' }}>
                    <span style={{ color: '#475569' }}>Quantum Kernel:</span>
                    <strong style={{ color: '#0f172a' }}>0.9736 F1</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1rem' }}>
                    <span style={{ color: '#475569' }}>Classical RBF:</span>
                    <strong style={{ color: '#0f172a' }}>0.9641 F1</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #f1f5f9', paddingTop: '10px', fontSize: '1rem' }}>
                    <span style={{ color: '#059669', fontWeight: 600 }}>Quantum Advantage:</span>
                    <strong style={{ color: '#059669' }}>+0.95 pp (Δ = +0.0095)</strong>
                  </div>
                </div>
              </div>

              <div className="clean-panel-interactive" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '1.2rem', color: '#7c3aed', marginBottom: '14px' }}>8D Dense RoBERTa Embeddings</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1rem' }}>
                    <span style={{ color: '#475569' }}>Quantum Kernel:</span>
                    <strong style={{ color: '#0f172a' }}>0.9601 F1</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1rem' }}>
                    <span style={{ color: '#475569' }}>Classical RBF:</span>
                    <strong style={{ color: '#0f172a' }}>0.9896 F1</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #f1f5f9', paddingTop: '10px', fontSize: '1rem' }}>
                    <span style={{ color: '#dc2626', fontWeight: 600 }}>Classical Advantage:</span>
                    <strong style={{ color: '#dc2626' }}>-2.95 pp (Δ = -0.0295)</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 5: CROSS-SOURCE GENERALIZATION */}
        {activeTab === 'generalization' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <span className="badge badge-rose" style={{ marginBottom: '10px' }}>Domain Shift Evaluation</span>
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
                  <div style={{ fontSize: '0.82rem', color: '#64748b' }}>Full Vocabulary Ceiling</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 6: GEOMETRY & ENTROPY */}
        {activeTab === 'geometry' && geomData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <span className="badge badge-purple" style={{ marginBottom: '10px' }}>Geometric Profiling</span>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Decoupling Single-State Entropy and Pairwise Kernel Diversity
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Regressions demonstrate a strong inverse correlation (r = -0.78 to -0.83) between single-state von Neumann entropy and pairwise kernel diversity. Spreading statevectors across basis states causes pairwise fidelities to concentrate, refuting the heuristic that greater single-state dispersion yields better classification.
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
              <span className="badge badge-amber" style={{ marginBottom: '10px' }}>Simulation Profiling</span>
              <h2 style={{ fontSize: '1.6rem', color: '#0f172a', marginBottom: '8px' }}>
                Computational Execution Time and Memory Scaling
              </h2>
              <p style={{ color: '#475569', fontSize: '1.02rem', lineHeight: '1.6' }}>
                Classical statevector simulation of quantum kernels scales steeply with dimensionality. At 12 dimensions, computing the quantum kernel requires 108.8s compared to 1.7s for RBF (64.0× ratio), while 16D encounters an unsustainable &gt;10.5 GB memory ceiling on 10,000 samples.
              </p>
            </div>

            <div className="clean-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '16px' }}>
                Kernel Matrix Computation Time (Seconds) vs Qubits / Dimensions
              </h3>
              <div style={{ height: '360px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={runtimeData.filter(d => d.quantum_s !== null)} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="dim" stroke="#475569" fontSize={13} fontWeight={500} />
                    <YAxis stroke="#475569" scale="log" domain={['auto', 'auto']} fontSize={13} tickFormatter={(v) => `${v}s`} />
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
          </div>
        )}

        {/* VIEW 8: EVIDENCE TABLES */}
        {activeTab === 'tables' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <span className="badge badge-blue" style={{ marginBottom: '8px' }}>Audited Research Data</span>
                  <h2 style={{ fontSize: '1.6rem', color: '#0f172a' }}>Publication Evidence Tables</h2>
                </div>
                {/* Table Selector */}
                <select
                  value={selectedTableId}
                  onChange={(e) => setSelectedTableId(e.target.value)}
                  style={{
                    background: '#ffffff',
                    color: '#0f172a',
                    border: '1px solid #cbd5e1',
                    borderRadius: '8px',
                    padding: '9px 16px',
                    fontSize: '0.98rem',
                    fontWeight: 600,
                    outline: 'none',
                    cursor: 'pointer'
                  }}
                >
                  {tablesList.map(t => (
                    <option key={t.id} value={t.id}>{t.title}</option>
                  ))}
                </select>
              </div>
            </div>

            {tableContent && !isLoadingTable && (
              <div className="clean-card" style={{ padding: '24px', overflowX: 'auto' }}>
                <h3 style={{ fontSize: '1.25rem', color: '#0f172a', marginBottom: '16px' }}>
                  {tablesList.find(t => t.id === selectedTableId)?.title}
                </h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.95rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569', background: '#f8fafc' }}>
                      {tableContent.columns.map((col, idx) => (
                        <th key={idx} style={{ padding: '12px 14px', whiteSpace: 'nowrap' }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableContent.rows.map((row, rIdx) => (
                      <tr key={rIdx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        {tableContent.columns.map((col, cIdx) => (
                          <td key={cIdx} style={{ padding: '12px 14px', color: cIdx === 0 ? '#0f172a' : '#475569' }}>
                            {row[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid #e2e8f0', background: '#ffffff', padding: '20px 28px', textAlign: 'center', fontSize: '0.9rem', color: '#64748b' }}>
        <p>Quantum Text Security Research Platform • Multi-Dataset Empirical Benchmark • Apple Silicon ARM64 / PyTorch complex128</p>
      </footer>
    </div>
  );
}
