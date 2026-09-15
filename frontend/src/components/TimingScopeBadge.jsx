import React from 'react';
import { Clock, Zap, AlertCircle } from 'lucide-react';

/**
 * TimingScopeBadge
 * Explicitly disambiguates Live Request Latency from Research Benchmark Runtimes.
 */
export default function TimingScopeBadge({ scope = 'live', customText = null }) {
  if (scope === 'fallback') {
    return (
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        background: '#fffbeb',
        border: '1px solid #fde68a',
        color: '#b45309',
        padding: '3px 9px',
        borderRadius: '6px',
        fontSize: '0.82rem',
        fontWeight: 600
      }}>
        <AlertCircle size={13} color="#d97706" />
        {customText || "Client-side simulation fallback"}
      </span>
    );
  }

  if (scope === 'live') {
    return (
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        background: '#f0fdf4',
        border: '1px solid #bbf7d0',
        color: '#166534',
        padding: '3px 9px',
        borderRadius: '6px',
        fontSize: '0.82rem',
        fontWeight: 600
      }}>
        <Zap size={13} color="#16a34a" />
        {customText || "Live request latency (single sample)"}
      </span>
    );
  }

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      background: '#eff6ff',
      border: '1px solid #bfdbfe',
      color: '#1e40af',
      padding: '3px 9px',
      borderRadius: '6px',
      fontSize: '0.82rem',
      fontWeight: 600
    }}>
      <Clock size={13} color="#2563eb" />
      {customText || "Research benchmark runtime — 10,000 samples"}
    </span>
  );
}
