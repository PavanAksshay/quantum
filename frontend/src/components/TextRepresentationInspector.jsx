import React, { useState } from 'react';
import { Layers, ChevronRight, Hash, Cpu, ArrowRight, Eye, Code, Zap } from 'lucide-react';
import ExperimentStatusBadge from './ExperimentStatusBadge';

/**
 * TextRepresentationInspector
 * Transparent step-by-step feature trace from raw text to quantum phase encoding.
 */
export default function TextRepresentationInspector({ traceData, representationMeta, dimension = 8 }) {
  const [activeStep, setActiveStep] = useState(3);

  if (!traceData) {
    return null;
  }

  const steps = [
    {
      id: 1,
      title: "1. Raw Text Input",
      desc: "Raw security message string (Email Subject + Body or SMS message).",
      dataSummary: `${traceData.character_count} characters • ${traceData.word_count} words`
    },
    {
      id: 2,
      title: "2. Native Representation",
      desc: `Extracted features in original native space before projection.`,
      dataSummary: `${traceData.representation_name} (${traceData.original_dimension} native dimensions)`
    },
    {
      id: 3,
      title: "3. SVD Coordinate Vector",
      desc: `TruncatedSVD + StandardScaler projection into matched ${dimension}-dimensional subspace.`,
      dataSummary: `${dimension} standardized continuous features`
    },
    {
      id: 4,
      title: "4. Quantum Phase Encoding",
      desc: `MinMax normalization mapped to [0, π] quantum rotation angles for Pauli-Z phase gates.`,
      dataSummary: `${dimension} rotation angles (radians)`
    },
    {
      id: 5,
      title: "5. Quantum State Preparation",
      desc: `2-layer cyclic ZZFeatureMap entangling circuit generating 2^${dimension} (${Math.pow(2, dimension)}) statevector amplitudes.`,
      dataSummary: `|ψ(x)⟩ ∈ ℂ^${Math.pow(2, dimension)} Hilbert state space`
    }
  ];

  return (
    <div className="clean-card" style={{ padding: '24px', marginTop: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <Layers size={18} color="#2563eb" />
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#0f172a' }}>
              Text-Level Representation & Encoding Pipeline Trace
            </h3>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#64748b' }}>
            Mathematical progression from raw security message to quantum phase angles and statevector.
          </p>
        </div>
        {representationMeta && (
          <ExperimentStatusBadge status={representationMeta.is_canonical ? 'CANONICAL' : 'EXPLORATORY'} />
        )}
      </div>

      {/* Interactive Step Navigator */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '10px', marginBottom: '20px' }}>
        {steps.map((step) => (
          <button
            key={step.id}
            onClick={() => setActiveStep(step.id)}
            style={{
              textAlign: 'left',
              padding: '12px 14px',
              borderRadius: '8px',
              border: `1px solid ${activeStep === step.id ? '#2563eb' : '#e2e8f0'}`,
              background: activeStep === step.id ? '#eff6ff' : '#ffffff',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <div style={{ fontSize: '0.86rem', fontWeight: 700, color: activeStep === step.id ? '#1d4ed8' : '#334155' }}>
              {step.title}
            </div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: '3px' }}>
              {step.dataSummary}
            </div>
          </button>
        ))}
      </div>

      {/* Step Detail Content */}
      <div style={{ background: '#f8fafc', borderRadius: '10px', border: '1px solid #e2e8f0', padding: '18px' }}>
        {activeStep === 1 && (
          <div>
            <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Step 1: Input Message Preprocessing
            </h4>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '12px', lineHeight: '1.5' }}>
              {traceData.dataset === 'SMS' ? 
                'Raw SMS message normalized as a single text sequence.' : 
                'Email formatted according to paper preprocessing rules: Subject and Body concatenated with standardized header formatting.'}
            </p>
            <div style={{ display: 'flex', gap: '20px', fontSize: '0.85rem' }}>
              <div style={{ background: '#ffffff', padding: '8px 14px', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <span style={{ color: '#64748b' }}>Character Count:</span> <strong>{traceData.character_count}</strong>
              </div>
              <div style={{ background: '#ffffff', padding: '8px 14px', borderRadius: '6px', border: '1px solid #cbd5e1' }}>
                <span style={{ color: '#64748b' }}>Word Count:</span> <strong>{traceData.word_count}</strong>
              </div>
            </div>
          </div>
        )}

        {activeStep === 2 && (
          <div>
            <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Step 2: Native Representation Extraction
            </h4>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '12px', lineHeight: '1.5' }}>
              Extracted via <strong>{traceData.representation_name}</strong> into its native {traceData.original_dimension}-dimensional vector space.
            </p>
            <div style={{ background: '#1e293b', color: '#f8fafc', padding: '12px', borderRadius: '6px', fontFamily: 'monospace', fontSize: '0.85rem' }}>
              {`Representation: ${traceData.representation_name}\nNative Dimension: ${traceData.original_dimension}D\nFeature Type: ${representationMeta?.is_sparse ? 'Sparse TF-IDF (Unigram + Bigram, Sublinear TF)' : 'Dense Transformer Mean-Pooled Embeddings'}`}
            </div>
          </div>
        )}

        {activeStep === 3 && (
          <div>
            <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Step 3: Dimensionality Projection ({dimension}D Standardized Coordinates)
            </h4>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '12px', lineHeight: '1.5' }}>
              TruncatedSVD projects high-dimensional features to {dimension} components, followed by StandardScaler zero-mean unit-variance transformation:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '8px', marginBottom: '12px' }}>
              {traceData.projected_coordinates?.map((val, idx) => (
                <div key={idx} style={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '6px', padding: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600 }}>x_{idx + 1}</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0f172a' }}>{val.toFixed(4)}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: '0.82rem', color: '#64748b', fontStyle: 'italic' }}>
              * These coordinates are passed directly into the Classical Gaussian RBF and Linear SVM baselines.
            </div>
          </div>
        )}

        {activeStep === 4 && (
          <div>
            <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Step 4: Quantum Phase Coordinate Mapping [0, π]
            </h4>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '12px', lineHeight: '1.5' }}>
              Coordinates are linearly scaled into the interval [0, π] for parameterized Pauli-Z and ZZ entanglement rotations:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: '8px', marginBottom: '12px' }}>
              {traceData.phase_coordinates_rad?.map((val, idx) => (
                <div key={idx} style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '6px', padding: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.72rem', color: '#166534', fontWeight: 600 }}>θ_{idx + 1} (rad)</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#15803d' }}>{val.toFixed(4)}</div>
                  <div style={{ fontSize: '0.7rem', color: '#16a34a' }}>{(val / Math.PI).toFixed(2)}π</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
              Formula: <code>θ_i = π · (x_i - min(X)) / (max(X) - min(X))</code>
            </div>
          </div>
        )}

        {activeStep === 5 && (
          <div>
            <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a', marginBottom: '8px' }}>
              Step 5: Quantum Statevector & Fidelity Kernel Evaluation
            </h4>
            <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '12px', lineHeight: '1.5' }}>
              The 2-layer cyclic ZZFeatureMap prepares state |ψ(x)⟩ across {dimension} qubits in a {Math.pow(2, dimension)}-dimensional complex Hilbert space:
            </p>
            <div style={{ background: '#ffffff', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.85rem', color: '#334155' }}>
              <div style={{ marginBottom: '8px' }}>
                <strong>Circuit Architecture:</strong> 2-Layer Cyclic ZZFeatureMap with Hadamard superposition, single-qubit Rz(2θ_i) rotations, and nearest-neighbor + cyclic CNOT-Rz(2(π-θ_i)(π-θ_j))-CNOT entangling gates.
              </div>
              <div>
                <strong>Kernel Evaluation:</strong> Fidelity similarity <code>K(x, x') = |⟨ψ(x)|ψ(x')⟩|²</code> computed via exact complex128 statevector inner product against support vectors.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
