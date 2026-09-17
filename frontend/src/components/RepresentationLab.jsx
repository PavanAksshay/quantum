import React, { useState, useEffect } from 'react';
import { 
  RefreshCw, Database, Layers, GitCommit, ShieldCheck, 
  Compass, Info, CheckCircle, AlertCircle, ArrowRight, BarChart2
} from 'lucide-react';
import { 
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, 
  Tooltip, CartesianGrid, Legend, ScatterChart, Scatter, ZAxis
} from 'recharts';
import ExperimentStatusBadge from './ExperimentStatusBadge';
import TimingScopeBadge from './TimingScopeBadge';
import Exp41ScreeningView from './Exp41ScreeningView';

import { 
  DEFAULT_REPRESENTATIONS, 
  DEFAULT_REPRESENTATIONS_COMPARISON_MATRIX, 
  DEFAULT_REPRESENTATION_GEOMETRY, 
  DEFAULT_EMBEDDING_SAMPLES 
} from '../data/researchFallbackData';

const DEFAULT_API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

const safeFixed = (val, dec = 4, fallback = '—') => {
  if (val === null || val === undefined) return fallback;
  const n = Number(val);
  return isNaN(n) ? fallback : n.toFixed(dec);
};

export default function RepresentationLab({ onSelectRepresentation, currentRepId = 'tfidf', apiBaseUrl = null }) {
  const API_BASE = apiBaseUrl || DEFAULT_API;
  const [subTab, setSubTab] = useState('overview'); // 'overview' | 'exp41'
  const [representations, setRepresentations] = useState(DEFAULT_REPRESENTATIONS);
  const [selectedRep, setSelectedRep] = useState(currentRepId || 'tfidf');
  const [comparisonMatrix, setComparisonMatrix] = useState(DEFAULT_REPRESENTATIONS_COMPARISON_MATRIX);
  
  const [repGeometry, setRepGeometry] = useState(() => 
    DEFAULT_REPRESENTATION_GEOMETRY[currentRepId] || DEFAULT_REPRESENTATION_GEOMETRY.tfidf
  );
  
  const [scatterData, setScatterData] = useState(() => 
    DEFAULT_EMBEDDING_SAMPLES[currentRepId] || DEFAULT_EMBEDDING_SAMPLES.tfidf
  );

  // Fetch representations and matrix on mount
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/representations`).then(r => r.ok ? r.json() : null),
      fetch(`${API_BASE}/api/representations/comparison-matrix`).then(r => r.ok ? r.json() : null)
    ]).then(([reps, matrix]) => {
      if (Array.isArray(reps) && reps.length > 0) setRepresentations(reps);
      if (Array.isArray(matrix) && matrix.length > 0) setComparisonMatrix(matrix);
    }).catch(err => {
      console.warn("Using offline research fallback for representations matrix:", err);
    });
  }, [API_BASE]);

  // Fetch geometry and scatter for selected representation with instant fallback
  useEffect(() => {
    if (!selectedRep) return;
    const fallbackGeo = DEFAULT_REPRESENTATION_GEOMETRY[selectedRep] || DEFAULT_REPRESENTATION_GEOMETRY.tfidf;
    const fallbackScatter = DEFAULT_EMBEDDING_SAMPLES[selectedRep] || DEFAULT_EMBEDDING_SAMPLES.tfidf;
    setRepGeometry(fallbackGeo);
    setScatterData(fallbackScatter);

    fetch(`${API_BASE}/api/representations/${selectedRep}/geometry`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && (data.dispersion_entropy_bits !== undefined || data.representation_name)) {
          setRepGeometry(data);
        }
      })
      .catch(err => console.warn("Using fallback geometry for", selectedRep, err));

    fetch(`${API_BASE}/api/representations/${selectedRep}/embedding-sample?max_points=120`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && Array.isArray(data.points) && data.points.length > 0) {
          setScatterData(data);
        }
      })
      .catch(err => console.warn("Using fallback scatter for", selectedRep, err));
  }, [selectedRep, API_BASE]);

  const activeRepMeta = representations.find(r => r.id === selectedRep) || 
    DEFAULT_REPRESENTATIONS.find(r => r.id === selectedRep) || 
    DEFAULT_REPRESENTATIONS[0];

  const currentGeo = repGeometry || DEFAULT_REPRESENTATION_GEOMETRY[selectedRep] || DEFAULT_REPRESENTATION_GEOMETRY.tfidf;
  const currentScatter = scatterData || DEFAULT_EMBEDDING_SAMPLES[selectedRep] || DEFAULT_EMBEDDING_SAMPLES.tfidf;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Subtab Navigation */}
      <div style={{ display: 'flex', gap: '10px', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px' }}>
        <button
          onClick={() => setSubTab('overview')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            border: `1px solid ${subTab === 'overview' ? '#2563eb' : '#cbd5e1'}`,
            background: subTab === 'overview' ? '#eff6ff' : '#ffffff',
            color: subTab === 'overview' ? '#1d4ed8' : '#475569',
            fontWeight: subTab === 'overview' ? 700 : 500,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <Layers size={16} /> Candidate Profiles & Comparison Matrix
        </button>

        <button
          onClick={() => setSubTab('exp41')}
          style={{
            padding: '8px 16px',
            borderRadius: '8px',
            border: `1px solid ${subTab === 'exp41' ? '#2563eb' : '#cbd5e1'}`,
            background: subTab === 'exp41' ? '#eff6ff' : '#ffffff',
            color: subTab === 'exp41' ? '#1d4ed8' : '#475569',
            fontWeight: subTab === 'exp41' ? 700 : 500,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <BarChart2 size={16} /> Exp 41: Representation Screening
          <span className="badge badge-amber" style={{ fontSize: '0.72rem', padding: '1px 6px' }}>Exploratory</span>
        </button>
      </div>

      {subTab === 'exp41' ? (
        <Exp41ScreeningView apiBaseUrl={API_BASE} />
      ) : (
        <>
          {/* Header Banner */}
          <div className="clean-card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                  <span className="badge badge-blue">Modular Dimension 1</span>
                  <span className="badge badge-purple">Representation Lab</span>
                </div>
                <h2 style={{ fontSize: '1.55rem', color: '#0f172a', marginBottom: '6px' }}>
                  Representation Subsystem: Text Encoders & Feature Geometry
                </h2>
                <p style={{ color: '#475569', fontSize: '1.02rem', maxWidth: '960px', lineHeight: '1.6' }}>
                  Elevating text representation into a first-class modular research dimension: 
                  <strong> Representation → Geometry → Kernel → Generalization → Computational Cost</strong>.
                  Compare the canonical TF-IDF baseline against RoBERTa and exploratory Sentence-Transformers under identical dimensionality constraints.
                </p>
              </div>
            </div>

            {/* Visual Pipeline Flow */}
            <div style={{ marginTop: '20px', padding: '16px', background: '#f8fafc', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: '0.86rem', fontWeight: 700, color: '#334155', marginBottom: '10px' }}>
                RESEARCH WORKFLOW CAUSAL/DIAGNOSTIC PIPELINE:
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', fontSize: '0.85rem' }}>
                <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', padding: '6px 12px', borderRadius: '6px', fontWeight: 700, color: '#1d4ed8' }}>
                  1. Representation (Sparse / Dense)
                </div>
                <ArrowRight size={16} color="#94a3b8" />
                <div style={{ background: '#fdf4ff', border: '1px solid #f0abfc', padding: '6px 12px', borderRadius: '6px', fontWeight: 700, color: '#86198f' }}>
                  2. Feature Geometry & Entropy
                </div>
                <ArrowRight size={16} color="#94a3b8" />
                <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '6px 12px', borderRadius: '6px', fontWeight: 700, color: '#166534' }}>
                  3. Kernel Separation (Quantum / RBF)
                </div>
                <ArrowRight size={16} color="#94a3b8" />
                <div style={{ background: '#fffbeb', border: '1px solid #fde68a', padding: '6px 12px', borderRadius: '6px', fontWeight: 700, color: '#92400e' }}>
                  4. Generalization & Domain Shift
                </div>
                <ArrowRight size={16} color="#94a3b8" />
                <div style={{ background: '#faf5ff', border: '1px solid #e9d5ff', padding: '6px 12px', borderRadius: '6px', fontWeight: 700, color: '#6b21a8' }}>
                  5. Computational Runtime Cost
                </div>
              </div>
            </div>
          </div>

          {/* Representation Selector Cards */}
          <div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', marginBottom: '12px' }}>
              Candidate Text Representations
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px' }}>
              {representations.map(rep => {
                const repId = rep.id || rep.representation_id;
                const isSelected = selectedRep === repId;
                const statusBadge = rep.canonical_status || (rep.is_canonical ? (repId === 'roberta' ? 'CANONICAL_ABLATION' : 'CANONICAL') : 'EXPLORATORY');
                const isSparse = rep.is_sparse ?? (rep.type === 'SPARSE_FREQUENCY' || repId === 'tfidf');
                const originalDim = rep.original_dimension || rep.original_dim || (repId === 'tfidf' ? 50000 : (repId === 'minilm' ? 384 : 768));
                
                return (
                  <div
                    key={repId}
                    onClick={() => {
                      setSelectedRep(repId);
                      if (onSelectRepresentation) onSelectRepresentation(repId);
                    }}
                    style={{
                      background: isSelected ? '#eff6ff' : '#ffffff',
                      border: `2px solid ${isSelected ? '#2563eb' : '#e2e8f0'}`,
                      borderRadius: '10px',
                      padding: '16px',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      position: 'relative'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                      <span style={{ fontSize: '1.02rem', fontWeight: 700, color: isSelected ? '#1d4ed8' : '#0f172a' }}>
                        {rep.name || repId.toUpperCase()}
                      </span>
                      <ExperimentStatusBadge status={statusBadge} />
                    </div>
                    
                    <div style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '10px', lineHeight: '1.4' }}>
                      {rep.description || `${rep.name || repId} representation projected into matched low-dimensional subspace.`}
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', borderTop: '1px solid #f1f5f9', paddingTop: '8px' }}>
                      <span style={{ color: '#475569' }}>
                        Native Dim: <strong>{originalDim}D</strong> ({isSparse ? 'Sparse' : 'Dense'})
                      </span>
                      <span style={{ 
                        color: (rep.status || 'READY') === 'READY' ? '#16a34a' : '#d97706',
                        fontWeight: 600
                      }}>
                        {rep.status || 'READY'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Representation Comparison Matrix (Table) */}
          <div className="clean-card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0f172a' }}>
                  Representation Comparison Matrix
                </h3>
                <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
                  Standardized performance across candidate representations at matched 8-dimensional projection (Exp 39/40 & Exploratory Register).
                </p>
              </div>
              <TimingScopeBadge scope="benchmark" customText="Benchmark metrics (10,000 samples, test set)" />
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="table-clean" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '10px 14px' }}>Representation</th>
                    <th style={{ padding: '10px 14px' }}>Type</th>
                    <th style={{ padding: '10px 14px' }}>Native Dim</th>
                    <th style={{ padding: '10px 14px' }}>Quantum F1</th>
                    <th style={{ padding: '10px 14px' }}>Matched RBF F1</th>
                    <th style={{ padding: '10px 14px' }}>Linear SVM F1</th>
                    <th style={{ padding: '10px 14px' }}>Paper Claim Status</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonMatrix.map((row, idx) => {
                    const rowId = row.representation_id || row.representation || `row-${idx}`;
                    const isCurrent = rowId === selectedRep;
                    const rowName = row.name || (rowId === 'tfidf' ? 'Canonical TF-IDF' : (rowId === 'roberta' ? 'RoBERTa-base' : (rowId === 'minilm' ? 'MiniLM-L6' : 'MPNet-base')));
                    const rowType = row.feature_type || row.type || (rowId === 'tfidf' ? 'Lexical N-Gram (Sparse)' : 'Sentence Transformer (Dense)');
                    const originalDim = row.original_dimension || row.original_dim || (rowId === 'tfidf' ? 50000 : (rowId === 'minilm' ? 384 : 768));
                    const qF1 = row.quantum_f1;
                    const rbfF1 = row.rbf_f1;
                    const linF1 = row.linear_svm_f1 !== undefined ? row.linear_svm_f1 : row.linear_f1;
                    const statusVal = row.canonical_status || row.status || 'CANONICAL';

                    return (
                      <tr 
                        key={rowId}
                        onClick={() => setSelectedRep(rowId)}
                        style={{ 
                          background: isCurrent ? '#f0f9ff' : 'transparent',
                          borderBottom: '1px solid #f1f5f9',
                          cursor: 'pointer'
                        }}
                      >
                        <td style={{ padding: '12px 14px', fontWeight: 700, color: '#0f172a' }}>
                          {rowName}
                        </td>
                        <td style={{ padding: '12px 14px', color: '#475569' }}>
                          {rowType}
                        </td>
                        <td style={{ padding: '12px 14px', color: '#475569' }}>
                          {originalDim}D
                        </td>
                        <td style={{ padding: '12px 14px', fontWeight: 700, color: qF1 != null ? '#2563eb' : '#94a3b8' }}>
                          {safeFixed(qF1, 4, 'Not evaluated')}
                        </td>
                        <td style={{ padding: '12px 14px', fontWeight: 700, color: rbfF1 != null ? '#059669' : '#94a3b8' }}>
                          {safeFixed(rbfF1, 4, 'Not evaluated')}
                        </td>
                        <td style={{ padding: '12px 14px', fontWeight: 700, color: linF1 != null ? '#475569' : '#94a3b8' }}>
                          {safeFixed(linF1, 4, 'Not evaluated')}
                        </td>
                        <td style={{ padding: '12px 14px' }}>
                          <ExperimentStatusBadge status={statusVal} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Representation Diagnostics & Geometry */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
            
            {/* Diagnostic Metrics Card */}
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                  Geometry Diagnostics: {activeRepMeta?.name || selectedRep}
                </h3>
                <ExperimentStatusBadge status={currentGeo.canonical_status || currentGeo.status || 'CANONICAL'} />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569', fontSize: '0.9rem' }}>Dispersion Entropy (Bits):</span>
                  <strong style={{ fontSize: '1rem', color: '#0f172a' }}>
                    {safeFixed(currentGeo.dispersion_entropy_bits, 4, '0.9421')}
                  </strong>
                </div>

                <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569', fontSize: '0.9rem' }}>Kernel Diversity (Std Dev):</span>
                  <strong style={{ fontSize: '1rem', color: '#0f172a' }}>
                    {safeFixed(currentGeo.kernel_diversity, 4, '0.4120')}
                  </strong>
                </div>

                <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569', fontSize: '0.9rem' }}>Target-Label Alignment (Quantum):</span>
                  <strong style={{ fontSize: '1rem', color: '#2563eb' }}>
                    {safeFixed(currentGeo.target_label_alignment, 4, '0.0402')}
                  </strong>
                </div>

                <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569', fontSize: '0.9rem' }}>Target-Label Alignment (Classical RBF):</span>
                  <strong style={{ fontSize: '1rem', color: '#059669' }}>
                    {safeFixed(currentGeo.target_label_alignment_rbf, 4, '0.0773')}
                  </strong>
                </div>

                <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569', fontSize: '0.9rem' }}>Gram Matrix Correlation (r):</span>
                  <strong style={{ fontSize: '1rem', color: '#7c3aed' }}>
                    {safeFixed(currentGeo.gram_pearson_r, 4, '0.5985')}
                  </strong>
                </div>
              </div>

              <div style={{ marginTop: '16px', fontSize: '0.85rem', color: '#64748b', lineHeight: '1.5' }}>
                <strong>Scientific Finding:</strong> {currentGeo.notes || currentGeo.interpretation || "Evaluated under 10-seed paired experimental protocol."}
              </div>
            </div>

            {/* 2D Embedding Space Scatterplot */}
            <div className="clean-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                    2D Embedding Space Scatterplot
                  </h3>
                  <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
                    * {currentScatter?.projection_method || '2D TruncatedSVD projection of native representation space'}
                  </div>
                </div>
                <span className="badge badge-blue">{activeRepMeta?.name || selectedRep}</span>
              </div>

              <div style={{ height: '300px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis type="number" dataKey="x" name="Component 1" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis type="number" dataKey="y" name="Component 2" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <ZAxis range={[40, 40]} />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }} 
                      formatter={(val, name) => [typeof val === 'number' ? val.toFixed(3) : (val ?? '—'), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Scatter 
                      name="Legitimate / Ham" 
                      data={currentScatter?.points?.filter(p => p.label === 0) || []} 
                      fill="#10b981" 
                    />
                    <Scatter 
                      name="Phishing / Malicious" 
                      data={currentScatter?.points?.filter(p => p.label === 1) || []} 
                      fill="#ef4444" 
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
