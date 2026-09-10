import React from 'react';
import { ShieldCheck, Compass, AlertTriangle, CheckCircle } from 'lucide-react';

/**
 * ExperimentStatusBadge
 * Explicitly partitions CANONICAL paper findings from EXPLORATORY representations.
 */
export default function ExperimentStatusBadge({ status = 'CANONICAL', showTooltip = true }) {
  const isCanonical = status === 'CANONICAL' || status === 'CANONICAL_ABLATION';
  const isCanonicalAblation = status === 'CANONICAL_ABLATION';
  const isExploratory = status === 'EXPLORATORY';

  if (isCanonical) {
    return (
      <span 
        title={showTooltip ? "Canonical finding reported in main paper tables (Exp 39/40)" : ""}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '5px',
          background: isCanonicalAblation ? '#fdf4ff' : '#eff6ff',
          border: `1px solid ${isCanonicalAblation ? '#f0abfc' : '#bfdbfe'}`,
          color: isCanonicalAblation ? '#86198f' : '#1e40af',
          padding: '2px 8px',
          borderRadius: '6px',
          fontSize: '0.78rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.03em'
        }}
      >
        <ShieldCheck size={12} color={isCanonicalAblation ? '#a21caf' : '#2563eb'} />
        {isCanonicalAblation ? 'Canonical Ablation' : 'Canonical Claim'}
      </span>
    );
  }

  if (isExploratory) {
    return (
      <span 
        title={showTooltip ? "Exploratory candidate representation (Representation Lab ablation, decoupled from paper claims)" : ""}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '5px',
          background: '#fffbeb',
          border: '1px solid #fde68a',
          color: '#92400e',
          padding: '2px 8px',
          borderRadius: '6px',
          fontSize: '0.78rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.03em'
        }}
      >
        <Compass size={12} color="#d97706" />
        Exploratory
      </span>
    );
  }

  return (
    <span 
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '5px',
        background: '#f8fafc',
        border: '1px solid #e2e8f0',
        color: '#64748b',
        padding: '2px 8px',
        borderRadius: '6px',
        fontSize: '0.78rem',
        fontWeight: 600
      }}
    >
      {status}
    </span>
  );
}
