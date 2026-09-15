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

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export default function RepresentationLab({ onSelectRepresentation, currentRepId = 'tfidf' }) {
  const [subTab, setSubTab] = useState('exp41'); // 'overview' | 'exp41'
  const [representations, setRepresentations] = useState([]);
  const [selectedRep, setSelectedRep] = useState(currentRepId);
  const [comparisonMatrix, setComparisonMatrix] = useState([]);
  const [repGeometry, setRepGeometry] = useState(null);
  const [scatterData, setScatterData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch representations and matrix on mount
  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/api/representations`).then(r => r.json()),
      fetch(`${API_BASE}/api/representations/comparison-matrix`).then(r => r.json())
    ]).then(([reps, matrix]) => {
      setRepresentations(Array.isArray(reps) ? reps : []);
      setComparisonMatrix(Array.isArray(matrix) ? matrix : []);
      setLoading(false);
    }).catch(err => {
      console.error("Error loading representations:", err);
      setRepresentations([]);
      setComparisonMatrix([]);
      setLoading(false);
    });
  }, []);

  // Fetch geometry and scatter for selected representation
  useEffect(() => {
    if (!selectedRep) return;
    fetch(`${API_BASE}/api/representations/${selectedRep}/geometry`)
      .then(r => r.json())
      .then(data => setRepGeometry(data))
      .catch(err => console.error("Error loading representation geometry:", err));

    fetch(`${API_BASE}/api/representations/${selectedRep}/embedding-sample?max_points=120`)
      .then(r => r.json())
      .then(data => setScatterData(data))
      .catch(err => console.error("Error loading embedding sample:", err));
  }, [selectedRep]);

  const activeRepMeta = representations.find(r => r.id === selectedRep) || representations[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Subtab Navigation */}
      <div style={{ display: 'flex', gap: '10px', borderBottom: '1px solid #e2e8f0', paddingBottom: '10px' }}>
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
      </div>

      {subTab === 'exp41' ? (
        <Exp41ScreeningView />
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
            const isSelected = selectedRep === rep.id;
            return (
              <div
                key={rep.id}
                onClick={() => {
                  setSelectedRep(rep.id);
                  if (onSelectRepresentation) onSelectRepresentation(rep.id);
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
                    {rep.name}
                  </span>
                  <ExperimentStatusBadge status={rep.is_canonical ? (rep.id === 'roberta' ? 'CANONICAL_ABLATION' : 'CANONICAL') : 'EXPLORATORY'} />
                </div>
                
                <div style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '10px', lineHeight: '1.4' }}>
                  {rep.description}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', borderTop: '1px solid #f1f5f9', paddingTop: '8px' }}>
                  <span style={{ color: '#475569' }}>
                    Native Dim: <strong>{rep.original_dimension}D</strong> ({rep.is_sparse ? 'Sparse' : 'Dense'})
                  </span>
                  <span style={{ 
                    color: rep.status === 'READY' ? '#16a34a' : (rep.status === 'UNAVAILABLE' ? '#dc2626' : '#d97706'),
                    fontWeight: 600
                  }}>
                    {rep.status}
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
              {comparisonMatrix.map(row => {
                const isCurrent = row.representation_id === selectedRep;
                return (
                  <tr 
                    key={row.representation_id}
                    onClick={() => setSelectedRep(row.representation_id)}
                    style={{ 
                      background: isCurrent ? '#f0f9ff' : 'transparent',
                      borderBottom: '1px solid #f1f5f9',
                      cursor: 'pointer'
                    }}
                  >
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: '#0f172a' }}>
                      {row.name}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#475569' }}>
                      {row.feature_type}
                    </td>
                    <td style={{ padding: '12px 14px', color: '#475569' }}>
                      {row.original_dimension}D
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: row.quantum_f1 ? '#2563eb' : '#94a3b8' }}>
                      {row.quantum_f1 !== null ? row.quantum_f1.toFixed(4) : 'Not evaluated'}
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: row.rbf_f1 ? '#059669' : '#94a3b8' }}>
                      {row.rbf_f1 !== null ? row.rbf_f1.toFixed(4) : 'Not evaluated'}
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: row.linear_svm_f1 ? '#475569' : '#94a3b8' }}>
                      {row.linear_svm_f1 !== null ? row.linear_svm_f1.toFixed(4) : 'Not evaluated'}
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <ExperimentStatusBadge status={row.canonical_status} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Representation Diagnostics & Geometry */}
      {repGeometry && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '20px' }}>
          
          {/* Diagnostic Metrics Card */}
          <div className="clean-card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                Geometry Diagnostics: {activeRepMeta?.name}
              </h3>
              <ExperimentStatusBadge status={repGeometry.canonical_status} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '0.9rem' }}>Dispersion Entropy (Bits):</span>
                <strong style={{ fontSize: '1rem', color: '#0f172a' }}>
                  {repGeometry.dispersion_entropy_bits !== null ? repGeometry.dispersion_entropy_bits.toFixed(4) : 'N/A'}
                </strong>
              </div>

              <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '0.9rem' }}>Kernel Diversity (Std Dev):</span>
                <strong style={{ fontSize: '1rem', color: '#0f172a' }}>
                  {repGeometry.kernel_diversity !== null ? repGeometry.kernel_diversity.toFixed(4) : 'N/A'}
                </strong>
              </div>

              <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '0.9rem' }}>Target-Label Alignment (Quantum):</span>
                <strong style={{ fontSize: '1rem', color: '#2563eb' }}>
                  {repGeometry.target_label_alignment !== null ? repGeometry.target_label_alignment.toFixed(4) : 'N/A'}
                </strong>
              </div>

              <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '0.9rem' }}>Target-Label Alignment (Classical RBF):</span>
                <strong style={{ fontSize: '1rem', color: '#059669' }}>
                  {repGeometry.target_label_alignment_rbf !== null ? repGeometry.target_label_alignment_rbf.toFixed(4) : 'N/A'}
                </strong>
              </div>

              <div style={{ background: '#f8fafc', padding: '12px 16px', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '0.9rem' }}>Gram Matrix Correlation (r):</span>
                <strong style={{ fontSize: '1rem', color: '#7c3aed' }}>
                  {repGeometry.gram_pearson_r !== null ? repGeometry.gram_pearson_r.toFixed(4) : 'N/A'}
                </strong>
              </div>
            </div>

            <div style={{ marginTop: '16px', fontSize: '0.85rem', color: '#64748b', lineHeight: '1.5' }}>
              <strong>Scientific Finding:</strong> {repGeometry.notes}
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
                  * {scatterData?.projection_method || '2D TruncatedSVD projection of native representation space'}
                </div>
              </div>
              <span className="badge badge-blue">{activeRepMeta?.name}</span>
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
                    formatter={(val, name, item) => [val.toFixed(3), name]}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Scatter 
                    name="Legitimate / Ham" 
                    data={scatterData?.points?.filter(p => p.label === 0) || []} 
                    fill="#10b981" 
                  />
                  <Scatter 
                    name="Phishing / Malicious" 
                    data={scatterData?.points?.filter(p => p.label === 1) || []} 
                    fill="#ef4444" 
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </>
  )}
</div>
  );
}
