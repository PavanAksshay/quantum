import React, { useState, useEffect } from 'react';
import { 
  BarChart2, Activity, Layers, ShieldCheck, Compass, 
  CheckCircle, AlertTriangle, ArrowRight, RefreshCw, Filter, GitCommit
} from 'lucide-react';
import { 
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, 
  Tooltip, CartesianGrid, Legend, ReferenceLine, LineChart, Line
} from 'recharts';

import ExperimentStatusBadge from './ExperimentStatusBadge';
import TimingScopeBadge from './TimingScopeBadge';

import { DEFAULT_EXP41_COMPARISON, DEFAULT_EXP41_GEOMETRY, DEFAULT_EXP41_DATA } from '../data/researchFallbackData';

const DEFAULT_API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export default function Exp41ScreeningView({ apiBaseUrl = null }) {
  const API_BASE = apiBaseUrl || DEFAULT_API;
  const [expData, setExpData] = useState(DEFAULT_EXP41_DATA);
  const [comparisonData, setComparisonData] = useState(DEFAULT_EXP41_COMPARISON);
  const [geometryData, setGeometryData] = useState(DEFAULT_EXP41_GEOMETRY);
  const [statusData, setStatusData] = useState({ status: 'READY', canonical_status: 'EXPLORATORY' });
  const [loading, setLoading] = useState(false);

  // Filters
  const [filterDataset, setFilterDataset] = useState('all');
  const [filterRep, setFilterRep] = useState('all');
  const [filterModel, setFilterModel] = useState('all');
  const [heatmapDataset, setHeatmapDataset] = useState('meajor');

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/research/representation/exp41`).then(r => r.json()),
      fetch(`${API_BASE}/api/research/representation/comparison`).then(r => r.json()),
      fetch(`${API_BASE}/api/research/representation/geometry`).then(r => r.json()),
      fetch(`${API_BASE}/api/research/representation/status`).then(r => r.json())
    ]).then(([exp, comp, geom, stat]) => {
      if (exp && typeof exp === 'object' && Array.isArray(exp.summary) && exp.summary.length > 0) setExpData(exp);
      if (Array.isArray(comp) && comp.length > 0) setComparisonData(comp);
      if (Array.isArray(geom) && geom.length > 0) setGeometryData(geom);
      if (stat && typeof stat === 'object') setStatusData(stat);
    }).catch(() => {});
  }, [API_BASE]);

  // Filtered raw records
  const rawResults = (expData?.screening_results && expData.screening_results.length > 0) 
    ? expData.screening_results 
    : DEFAULT_EXP41_DATA.screening_results;

  const filteredRecords = rawResults.filter(row => {
    if (filterDataset !== 'all' && row?.dataset?.toLowerCase() !== filterDataset.toLowerCase()) return false;
    if (filterRep !== 'all' && row?.representation?.toLowerCase() !== filterRep.toLowerCase()) return false;
    if (filterModel !== 'all' && !row?.model?.toLowerCase().includes(filterModel.toLowerCase())) return false;
    return true;
  });

  // Data for Representation-conditioned Q-RBF Difference Chart
  const chartDatasets = ['sms', 'ceas', 'meajor'];
  const representationsList = ['tfidf', 'minilm', 'roberta', 'mpnet'];
  
  const diffChartData = representationsList.map(rep => {
    const row = { representation: rep.toUpperCase() };
    chartDatasets.forEach(ds => {
      const match = (comparisonData || []).find(c => c?.dataset?.toLowerCase() === ds && c?.representation?.toLowerCase() === rep);
      row[ds] = match ? match.delta_f1_q_minus_rbf : null;
    });
    return row;
  });

  // Heatmap rows for selected dataset
  const summary = (expData?.summary && expData.summary.length > 0) 
    ? expData.summary 
    : DEFAULT_EXP41_DATA.summary;

  const heatmapRows = representationsList.map(rep => {
    const lin = summary.find(s => s?.dataset?.toLowerCase() === heatmapDataset.toLowerCase() && s?.representation?.toLowerCase() === rep && s?.model?.includes('Linear'));
    const rbf = summary.find(s => s?.dataset?.toLowerCase() === heatmapDataset.toLowerCase() && s?.representation?.toLowerCase() === rep && s?.model?.includes('RBF'));
    const q = summary.find(s => s?.dataset?.toLowerCase() === heatmapDataset.toLowerCase() && s?.representation?.toLowerCase() === rep && s?.model?.includes('Quantum'));

    return {
      representation: rep.toUpperCase(),
      linear_f1: lin ? lin.f1_mean : null,
      rbf_f1: rbf ? rbf.f1_mean : null,
      quantum_f1: q ? q.f1_mean : null,
      delta: (q && rbf) ? roundNumber(q.f1_mean - rbf.f1_mean, 4) : null
    };
  });

  function roundNumber(num, dec = 4) {
    return num !== null && num !== undefined ? Number(num.toFixed(dec)) : null;
  }

  function getCellColor(val) {
    if (val === null || val === undefined) return '#f1f5f9';
    if (val >= 0.95) return '#dcfce7'; // high green
    if (val >= 0.88) return '#eff6ff'; // mid blue
    if (val >= 0.78) return '#fdf4ff'; // purple
    return '#fffbeb'; // amber
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Banner */}
      <div className="clean-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '14px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <span className="badge badge-purple">Exploratory Extension</span>
              <span className="badge badge-amber">Exp 41 Screening</span>
              <ExperimentStatusBadge status="EXPLORATORY" />
            </div>
            <h2 style={{ fontSize: '1.55rem', color: '#0f172a', marginBottom: '6px' }}>
              Experiment 41: Controlled Representation Screening
            </h2>
            <p style={{ color: '#475569', fontSize: '1.02rem', maxWidth: '960px', lineHeight: '1.6' }}>
              Investigating the exploratory research question: 
              <em> “Does quantum-kernel behavior depend more strongly on text representation than on kernel choice?”</em>
              <br />
              Screening 3 datasets (SMS, CEAS, MeAJOR) across 4 representations (TF-IDF, MiniLM, RoBERTa, MPNet) and 3 models (Linear SVM, RBF SVM, Quantum Fidelity Kernel) over seeds 42, 123, 456 at matched 8D.
            </p>
          </div>
          <TimingScopeBadge scope="benchmark" customText="Research benchmark runtime (3-seed screening)" />
        </div>
      </div>

      {/* Top Screening KPI Highlights */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
        <div className="clean-panel-interactive" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.86rem', color: '#64748b', marginBottom: '4px' }}>Screening Scope</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0f172a', marginBottom: '4px' }}>
            108 Runs
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
            3 Datasets × 4 Encoders × 3 Models × 3 Seeds
          </div>
        </div>

        <div className="clean-panel-interactive" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.86rem', color: '#64748b', marginBottom: '4px' }}>Representation Range</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#2563eb', marginBottom: '4px' }}>
            11.52 pp
          </div>
          <div style={{ fontSize: '0.82rem', color: '#2563eb' }}>
            max(ΔF1) - min(ΔF1) on SMS (-5.93 pp to +5.59 pp)
          </div>
        </div>

        <div className="clean-panel-interactive" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.86rem', color: '#64748b', marginBottom: '4px' }}>Model Rank Reversals</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#86198f', marginBottom: '4px' }}>
            Confirmed (SMS & CEAS)
          </div>
          <div style={{ fontSize: '0.82rem', color: '#86198f' }}>
            Changing encoder inverts Quantum vs RBF rank
          </div>
        </div>

        <div className="clean-panel-interactive" style={{ padding: '20px' }}>
          <div style={{ fontSize: '0.86rem', color: '#64748b', marginBottom: '4px' }}>Equivalence Threshold</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#059669', marginBottom: '4px' }}>
            ε = ±0.01 F1
          </div>
          <div style={{ fontSize: '0.82rem', color: '#059669' }}>
            Region of practical equivalence
          </div>
        </div>
      </div>

      {/* Primary Chart: Representation-conditioned Q-RBF Difference */}
      <div className="clean-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
              Representation-Conditioned Q-RBF Difference (ΔF1 = Quantum F1 - Classical RBF F1)
            </h3>
            <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
              Mean ΔF1 across candidate representations. Bars inside the green shaded area (|ΔF1| ≤ 0.01) indicate practical equivalence.
            </p>
          </div>
          <span className="badge badge-purple">Exp 41 Screening</span>
        </div>

        <div style={{ height: '340px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={diffChartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="representation" stroke="#475569" fontSize={13} fontWeight={600} />
              <YAxis stroke="#475569" fontSize={13} domain={[-0.08, 0.08]} tickFormatter={(v) => `${(v * 100).toFixed(1)} pp`} />
              <Tooltip 
                contentStyle={{ background: '#ffffff', borderColor: '#cbd5e1', borderRadius: '8px', fontSize: '0.9rem' }}
                formatter={(val, name) => [val !== null ? `${(val * 100).toFixed(2)} pp (Δ = ${val > 0 ? '+' : ''}${val.toFixed(4)})` : 'N/A', name.toUpperCase()]}
              />
              <Legend wrapperStyle={{ fontSize: '0.92rem', paddingTop: '8px' }} />
              
              {/* Zero Reference Line */}
              <ReferenceLine y={0} stroke="#334155" strokeWidth={1.5} />
              
              {/* Practical Equivalence Bands (+0.01 and -0.01) */}
              <ReferenceLine y={0.01} stroke="#10b981" strokeDasharray="4 4" label={{ value: "+ε = +0.01 threshold", fill: "#059669", fontSize: 11, position: 'top' }} />
              <ReferenceLine y={-0.01} stroke="#10b981" strokeDasharray="4 4" label={{ value: "-ε = -0.01 threshold", fill: "#059669", fontSize: 11, position: 'bottom' }} />

              <Bar dataKey="sms" name="SMS Dataset" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ceas" name="CEAS Dataset" fill="#7c3aed" radius={[4, 4, 0, 0]} />
              <Bar dataKey="meajor" name="MeAJOR Dataset" fill="#059669" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Representation × Kernel Heatmap */}
      <div className="clean-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
              Representation × Model F1 Matrix (Heatmap)
            </h3>
            <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
              Mean F1 score across models for the selected screening dataset.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.88rem', fontWeight: 600, color: '#475569' }}>Dataset:</span>
            <select
              value={heatmapDataset}
              onChange={(e) => setHeatmapDataset(e.target.value)}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                background: '#ffffff',
                fontWeight: 600,
                fontSize: '0.88rem',
                color: '#0f172a'
              }}
            >
              <option value="sms">SMS Spam Collection</option>
              <option value="ceas">CEAS 2008 Corpus</option>
              <option value="meajor">MeAJOR Archive</option>
            </select>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="table-clean" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '0.95rem' }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '12px 16px', textAlign: 'left' }}>Representation</th>
                <th style={{ padding: '12px 16px' }}>Linear SVM baseline</th>
                <th style={{ padding: '12px 16px' }}>Classical RBF SVM</th>
                <th style={{ padding: '12px 16px' }}>Quantum Fidelity Kernel</th>
                <th style={{ padding: '12px 16px' }}>Observed ΔF1 (Q - RBF)</th>
                <th style={{ padding: '12px 16px' }}>Screening Classification</th>
              </tr>
            </thead>
            <tbody>
              {heatmapRows.map((row, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 700, color: '#0f172a' }}>
                    {row.representation}
                  </td>
                  <td style={{ padding: '12px 16px', background: getCellColor(row.linear_f1), fontWeight: 600 }}>
                    {row.linear_f1 !== null ? row.linear_f1.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '12px 16px', background: getCellColor(row.rbf_f1), fontWeight: 600 }}>
                    {row.rbf_f1 !== null ? row.rbf_f1.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '12px 16px', background: getCellColor(row.quantum_f1), fontWeight: 700, color: '#2563eb' }}>
                    {row.quantum_f1 !== null ? row.quantum_f1.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '12px 16px', fontWeight: 700, color: row.delta > 0.01 ? '#16a34a' : (row.delta < -0.01 ? '#dc2626' : '#64748b') }}>
                    {row.delta !== null ? (row.delta > 0 ? `+${row.delta.toFixed(4)}` : row.delta.toFixed(4)) : '—'}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {row.delta !== null ? (
                      row.delta > 0.01 ? (
                        <span className="badge badge-emerald">Observed Quantum Edge</span>
                      ) : row.delta < -0.01 ? (
                        <span className="badge badge-rose">Classical Advantage</span>
                      ) : (
                        <span className="badge badge-gray">Practical Parity</span>
                      )
                    ) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Geometry Diagnostics Breakdown */}
      <div className="clean-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
          Representation × Geometry Association Analysis
        </h3>
        <p style={{ fontSize: '0.88rem', color: '#64748b', marginBottom: '16px' }}>
          Observing how text representation geometry correlates with kernel diversity and target-label alignment.
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table className="table-clean" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '10px 14px' }}>Dataset</th>
                <th style={{ padding: '10px 14px' }}>Representation</th>
                <th style={{ padding: '10px 14px' }}>Seed</th>
                <th style={{ padding: '10px 14px' }}>Dispersion Entropy (Bits)</th>
                <th style={{ padding: '10px 14px' }}>Kernel Diversity (Std)</th>
                <th style={{ padding: '10px 14px' }}>Target Alignment (Q)</th>
                <th style={{ padding: '10px 14px' }}>Target Alignment (RBF)</th>
                <th style={{ padding: '10px 14px' }}>Gram Pearson (r)</th>
              </tr>
            </thead>
            <tbody>
              {geometryData.slice(0, 12).map((g, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#0f172a' }}>{g.dataset.toUpperCase()}</td>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#2563eb' }}>{g.representation.toUpperCase()}</td>
                  <td style={{ padding: '10px 14px', color: '#64748b' }}>{g.seed}</td>
                  <td style={{ padding: '10px 14px' }}>{g.dispersion_entropy_bits.toFixed(4)}</td>
                  <td style={{ padding: '10px 14px' }}>{g.kernel_diversity.toFixed(4)}</td>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#2563eb' }}>{g.target_label_alignment.toFixed(4)}</td>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#7c3aed' }}>{g.target_label_alignment_rbf.toFixed(4)}</td>
                  <td style={{ padding: '10px 14px', fontWeight: 600, color: '#059669' }}>{g.gram_pearson_r.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Raw Screening Results Table with Filters */}
      <div className="clean-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
              Full 108-Run Screening Replicates
            </h3>
            <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
              Individual replicate runs across seeds 42, 123, 456 under zero test leakage.
            </p>
          </div>

          {/* Interactive Filter Controls */}
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <select
              value={filterDataset}
              onChange={(e) => setFilterDataset(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
            >
              <option value="all">All Datasets</option>
              <option value="sms">SMS</option>
              <option value="ceas">CEAS</option>
              <option value="meajor">MeAJOR</option>
            </select>

            <select
              value={filterRep}
              onChange={(e) => setFilterRep(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
            >
              <option value="all">All Encoders</option>
              <option value="tfidf">TF-IDF</option>
              <option value="minilm">MiniLM</option>
              <option value="roberta">RoBERTa</option>
              <option value="mpnet">MPNet</option>
            </select>

            <select
              value={filterModel}
              onChange={(e) => setFilterModel(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
            >
              <option value="all">All Models</option>
              <option value="linear">Linear SVM</option>
              <option value="rbf">Classical RBF</option>
              <option value="quantum">Quantum Fidelity</option>
            </select>
          </div>
        </div>

        <div style={{ overflowX: 'auto', maxHeight: '420px' }}>
          <table className="table-clean" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead style={{ position: 'sticky', top: 0, background: '#f8fafc', zIndex: 1 }}>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Dataset</th>
                <th style={{ padding: '8px 12px' }}>Encoder</th>
                <th style={{ padding: '8px 12px' }}>Model</th>
                <th style={{ padding: '8px 12px' }}>Seed</th>
                <th style={{ padding: '8px 12px' }}>F1</th>
                <th style={{ padding: '8px 12px' }}>PR-AUC</th>
                <th style={{ padding: '8px 12px' }}>ROC-AUC</th>
                <th style={{ padding: '8px 12px' }}>Accuracy</th>
                <th style={{ padding: '8px 12px' }}>Runtime (s)</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((row, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{row?.dataset?.toUpperCase() ?? '—'}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#2563eb' }}>{row?.representation?.toUpperCase() ?? '—'}</td>
                  <td style={{ padding: '8px 12px', color: '#0f172a' }}>{row?.model ?? '—'}</td>
                  <td style={{ padding: '8px 12px', color: '#64748b' }}>{row?.seed ?? '—'}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: '#0f172a' }}>{row?.f1 != null ? row.f1.toFixed(4) : '—'}</td>
                  <td style={{ padding: '8px 12px' }}>{row?.pr_auc != null ? row.pr_auc.toFixed(4) : '—'}</td>
                  <td style={{ padding: '8px 12px' }}>{row?.roc_auc != null ? row.roc_auc.toFixed(4) : '—'}</td>
                  <td style={{ padding: '8px 12px' }}>{row?.accuracy != null ? (row.accuracy * 100).toFixed(1) + '%' : '—'}</td>
                  <td style={{ padding: '8px 12px', color: '#64748b' }}>{row?.runtime_sec != null ? row.runtime_sec.toFixed(2) + 's' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
