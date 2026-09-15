// Authoritative Frozen Research Evidence & Platform Fallbacks
// Ensures 100% data visibility for all graphs, matrices, and tables on static CDN hosting.

export const DEFAULT_REPRESENTATIONS = [
  {
    id: "tfidf",
    name: "Canonical TF-IDF + TruncatedSVD (8D)",
    type: "SPARSE_FREQUENCY",
    canonical_status: "CANONICAL",
    target_dim: 8,
    original_dimension: 50000,
    description: "50,000 max features, sublinear term-frequency scaling, unigram+bigram, TruncatedSVD to 8D."
  },
  {
    id: "minilm",
    name: "MiniLM-L6 Contextual (8D)",
    type: "DENSE_TRANSFORMER",
    canonical_status: "EXPLORATORY",
    target_dim: 8,
    original_dimension: 384,
    description: "All-MiniLM-L6-v2 sentence embeddings projected to 8D via TruncatedSVD."
  },
  {
    id: "roberta",
    name: "RoBERTa-base Contextual (8D)",
    type: "DENSE_TRANSFORMER",
    canonical_status: "CANONICAL_ABLATION",
    target_dim: 8,
    original_dimension: 768,
    description: "RoBERTa-base mean-pooled embeddings projected to 8D via TruncatedSVD."
  },
  {
    id: "mpnet",
    name: "MPNet-base Contextual (8D)",
    type: "DENSE_TRANSFORMER",
    canonical_status: "EXPLORATORY",
    target_dim: 8,
    original_dimension: 768,
    description: "All-MPNet-base-v2 sentence embeddings projected to 8D via TruncatedSVD."
  }
];

export const DEFAULT_PRESETS = [
  {
    id: "sample_phish_1",
    title: "Urgent Account Suspension Alert",
    category: "Phishing / Scam",
    subject: "CRITICAL: Immediate Account Security Verification Required",
    body: "Dear Valued Customer,\n\nWe detected unauthorized login attempts to your corporate portal from an unrecognized IP address. Your access will be suspended within 24 hours unless you re-verify your identity.\n\nPlease follow the secure verification link: http://auth-portal-secure-update.com/verify?token=938210\n\nIT Security Department",
    text: "Subject: CRITICAL: Immediate Account Security Verification Required\n\nDear Valued Customer, We detected unauthorized login attempts to your corporate portal. Your access will be suspended within 24 hours unless you re-verify your identity at http://auth-portal-secure-update.com/verify"
  },
  {
    id: "sample_phish_2",
    title: "International Lottery Prize Award",
    category: "Phishing / Scam",
    subject: "NOTIFICATION OF AWARD WINNINGS: Ref #8491/2026",
    body: "CONGRATULATIONS! Your email address was selected as the lucky winner of 1,500,000 USD in the 2026 Global Cyber Draw.\n\nTo claim your prize funds, submit your full name, banking details, and telephone contact immediately to our fiduciary agent at agent-release@intl-claims-payout.org.",
    text: "Subject: NOTIFICATION OF AWARD WINNINGS: Ref #8491/2026\n\nCONGRATULATIONS! Your email address was selected as the lucky winner of 1,500,000 USD in the 2026 Global Cyber Draw. Submit details to agent-release@intl-claims-payout.org."
  },
  {
    id: "sample_ham_1",
    title: "GitHub Commit & CI Notification",
    category: "Legitimate / Ham",
    subject: "[GitHub] Build succeeded on main branch (Commit 4b91f0a)",
    body: "The continuous integration workflow for repository 'PavanAksshay/quantum' completed successfully.\n\nAll unit tests (14 passed), linting checks, and artifact verification audits passed without warnings.\n\nView build execution details at https://github.com/PavanAksshay/quantum/actions/runs/849102.",
    text: "Subject: [GitHub] Build succeeded on main branch (Commit 4b91f0a)\n\nThe continuous integration workflow for repository 'PavanAksshay/quantum' completed successfully. All unit tests (14 passed) passed."
  },
  {
    id: "sample_ham_2",
    title: "Weekly Research Engineering Sync",
    category: "Legitimate / Ham",
    subject: "Agenda: Weekly Quantum NLP Research Alignment Meeting",
    body: "Hi team,\n\nAttached is the agenda for our synchronization meeting on Thursday at 3:00 PM EST:\n1. Review of Exp 40 confirmation experiments across 10 independent computational seeds.\n2. Discussion of cross-source generalization results on the MeAJOR benchmark.\n3. Final draft review of Manuscript Section 3.\n\nBest regards,\nPavan",
    text: "Subject: Agenda: Weekly Quantum NLP Research Alignment Meeting\n\nHi team, Attached is the agenda for our synchronization meeting on Thursday at 3:00 PM EST: Review of Exp 40 confirmation experiments."
  }
];

export const DEFAULT_METRICS = {
  meajor_iid_8d: {
    quantum_f1: 0.8754,
    rbf_f1: 0.8709,
    linear_f1: 0.8445,
    delta_f1: 0.0046,
    ci_lower: 0.0030,
    ci_upper: 0.0061,
    p_value: 0.0016,
    status: "PRACTICALLY_EQUIVALENT",
    status_label: "Practically Equivalent (Within ±0.01 F1 margin)"
  },
  meajor_iid_12d: {
    quantum_f1: 0.9137,
    rbf_f1: 0.9123,
    linear_f1: 0.8892,
    delta_f1: 0.0014,
    ci_lower: -0.0010,
    ci_upper: 0.0037,
    p_value: 0.2824,
    status: "PARITY",
    status_label: "Parity (No statistically significant difference)"
  },
  source_holdout_direction_b: {
    quantum_f1: 0.6680,
    rbf_f1: 0.6913,
    linear_f1: 0.7562,
    delta_f1: -0.0233,
    p_value: 0.0046,
    p_fdr: 0.0069,
    status: "CLASSICAL_RBF_SUPERIOR",
    status_label: "Classical RBF Superior (Significant degradation under shift)"
  },
  simulation_ratio_12d: 64.0,
  linear_svm_50k_tfidf: 0.8837
};

export const DEFAULT_DIM_DATA = [
  { dim: "2D", qubits: 2, quantum_f1: 0.6447, rbf_f1: 0.6735, linear_f1: 0.6853, delta: -0.0288, ci_lower: -0.0345, ci_upper: -0.0231, status: "CANONICAL" },
  { dim: "4D", qubits: 4, quantum_f1: 0.7876, rbf_f1: 0.7918, linear_f1: 0.7584, delta: -0.0042, ci_lower: -0.0078, ci_upper: -0.0006, status: "CANONICAL" },
  { dim: "6D", qubits: 6, quantum_f1: 0.8253, rbf_f1: 0.8254, linear_f1: 0.7946, delta: -0.0001, ci_lower: -0.0035, ci_upper: 0.0033, status: "CANONICAL" },
  { dim: "8D", qubits: 8, quantum_f1: 0.8754, rbf_f1: 0.8709, linear_f1: 0.8445, delta: 0.0046, ci_lower: 0.0030, ci_upper: 0.0061, status: "CANONICAL" },
  { dim: "10D", qubits: 10, quantum_f1: 0.9023, rbf_f1: 0.8967, linear_f1: 0.8730, delta: 0.0057, ci_lower: 0.0032, ci_upper: 0.0081, status: "CANONICAL" },
  { dim: "12D", qubits: 12, quantum_f1: 0.9137, rbf_f1: 0.9123, linear_f1: 0.8892, delta: 0.0014, ci_lower: -0.0010, ci_upper: 0.0037, status: "CANONICAL" }
];

export const DEFAULT_RUNTIME_DATA = [
  { dim: "2D", qubits: 2, quantum_s: 3.4, rbf_s: 1.3, ratio: 2.6, quantum_ram_mb: 142, rbf_ram_mb: 95 },
  { dim: "4D", qubits: 4, quantum_s: 5.8, rbf_s: 1.3, ratio: 4.5, quantum_ram_mb: 185, rbf_ram_mb: 98 },
  { dim: "6D", qubits: 6, quantum_s: 9.4, rbf_s: 1.4, ratio: 6.7, quantum_ram_mb: 310, rbf_ram_mb: 102 },
  { dim: "8D", qubits: 8, quantum_s: 17.2, rbf_s: 1.4, ratio: 12.3, quantum_ram_mb: 680, rbf_ram_mb: 108 },
  { dim: "10D", qubits: 10, quantum_s: 44.6, rbf_s: 1.5, ratio: 29.7, quantum_ram_mb: 2150, rbf_ram_mb: 115 },
  { dim: "12D", qubits: 12, quantum_s: 108.8, rbf_s: 1.7, ratio: 64.0, quantum_ram_mb: 6400, rbf_ram_mb: 122 },
  { dim: "16D", qubits: 16, quantum_s: null, rbf_s: 2.1, ratio: null, quantum_ram_mb: 10500, rbf_ram_mb: 135 }
];

export const DEFAULT_GEOM_DATA = {
  datasets: {
    sms: { dataset: "SMS Spam Collection", quantum_kta: 0.0382, rbf_kta: 0.0768, deficit_pct: -50.3, entropy_vs_div_r: -0.8257 },
    ceas: { dataset: "CEAS 2008 Phishing Corpus", quantum_kta: 0.0220, rbf_kta: 0.0603, deficit_pct: -63.5, entropy_vs_div_r: -0.8170 },
    meajor: { dataset: "MeAJOR Email Archive", quantum_kta: 0.0402, rbf_kta: 0.0773, deficit_pct: -48.0, entropy_vs_div_r: -0.7822 }
  },
  pearson_scaling: [
    { dimension: "2D", qubits: 2, pearson_r: 0.7120 },
    { dimension: "4D", qubits: 4, pearson_r: 0.6740 },
    { dimension: "6D", qubits: 6, pearson_r: 0.6350 },
    { dimension: "8D", qubits: 8, pearson_r: 0.5985 },
    { dimension: "10D", qubits: 10, pearson_r: 0.5740 },
    { dimension: "12D", qubits: 12, pearson_r: 0.5510 },
    { dimension: "16D", qubits: 16, pearson_r: 0.4566 }
  ]
};

export const DEFAULT_TABLES_LIST = [
  { id: "table_1", name: "Table 1: Dataset Characteristics" },
  { id: "table_2", name: "Table 2: High-Dimensional Baselines" },
  { id: "table_3", name: "Table 3: Canonical Comparison (8D)" },
  { id: "table_4", name: "Table 4: Dimensionality Scaling Sweep" },
  { id: "table_5", name: "Table 5: Source Domain Holdout" },
  { id: "table_6", name: "Table 6: Statistical Significance & CI" },
  { id: "table_7", name: "Table 7: Kernel Geometry Diagnostics" },
  { id: "table_8", name: "Table 8: Computational Runtime & Memory" }
];

export const DEFAULT_TABLES_CONTENT = {
  table_1: {
    id: "table_1",
    name: "Table 1: Dataset Characteristics and Leakage Audit",
    columns: ["Corpus Name", "Total Records", "Usable Split", "Legitimate (Ham)", "Malicious (Spam/Phish)", "Positive Prevalence", "Median Doc Length"],
    rows: [
      ["UCI SMS Spam Collection", "5,574", "5,572", "4,825", "747", "13.41%", "12 words"],
      ["CEAS 2008 Phishing Corpus", "39,154", "15,000 (canonical)", "12,165", "2,835", "18.90%", "48 words"],
      ["MeAJOR Email Archive", "108,685", "108,684 (total)", "87,678", "21,006", "19.33%", "84 words"]
    ]
  },
  table_2: {
    id: "table_2",
    name: "Table 2: High-Dimensional Baseline Performance",
    columns: ["Model", "Feature Space", "Dimensionality", "MeAJOR IID F1", "Direction A F1", "Direction B F1"],
    rows: [
      ["Linear SVM", "TF-IDF (50k max)", "50,000D", "0.9721", "0.8922", "0.8837"],
      ["Classical RBF", "TF-IDF (50k max)", "50,000D", "0.9688", "0.8845", "0.8710"],
      ["Linear SVM", "TruncatedSVD Subspace", "32D", "0.9410", "0.8520", "0.8761"],
      ["Linear SVM", "TruncatedSVD Subspace", "16D", "0.9022", "0.8140", "0.8125"],
      ["Linear SVM", "TruncatedSVD Subspace", "8D", "0.8445", "0.7610", "0.7562"]
    ]
  },
  table_3: {
    id: "table_3",
    name: "Table 3: Canonical Comparison (8D Matched Models)",
    columns: ["Dataset", "Representation", "Linear SVM F1", "Classical RBF F1", "Quantum Kernel F1", "ΔF1 (Q - RBF)", "Practical Equivalence (ε=0.01)"],
    rows: [
      ["MeAJOR IID", "TF-IDF + SVD (8D)", "0.8445 ± 0.0042", "0.8709 ± 0.0030", "0.8754 ± 0.0029", "+0.0046 ± 0.0027", "EQUIVALENT (|Δ| ≤ 0.01)"],
      ["CEAS 2008", "TF-IDF + SVD (8D)", "0.9528 ± 0.0021", "0.9416 ± 0.0019", "0.9504 ± 0.0022", "+0.0088 ± 0.0020", "EQUIVALENT (|Δ| ≤ 0.01)"],
      ["SMS Spam", "TF-IDF + SVD (8D)", "0.7927 ± 0.0051", "0.8091 ± 0.0048", "0.7969 ± 0.0052", "-0.0122 ± 0.0044", "RBF ADVANTAGE"]
    ]
  },
  table_4: {
    id: "table_4",
    name: "Table 4: Dimensionality Scaling Trajectory (2D to 12D)",
    columns: ["Dimension", "Qubits", "Linear SVM F1", "Classical RBF F1", "Quantum Kernel F1", "ΔF1 (Q - RBF)", "95% Bootstrap CI"],
    rows: [
      ["2D", "2 Qubits", "0.6853", "0.6735", "0.6447", "-0.0288", "[-0.0345, -0.0231]"],
      ["4D", "4 Qubits", "0.7584", "0.7918", "0.7876", "-0.0042", "[-0.0078, -0.0006]"],
      ["6D", "6 Qubits", "0.7946", "0.8254", "0.8253", "-0.0001", "[-0.0035, +0.0033]"],
      ["8D", "8 Qubits", "0.8445", "0.8709", "0.8754", "+0.0046", "[+0.0030, +0.0061]"],
      ["10D", "10 Qubits", "0.8730", "0.8967", "0.9023", "+0.0057", "[+0.0032, +0.0081]"],
      ["12D", "12 Qubits", "0.8892", "0.9123", "0.9137", "+0.0014", "[-0.0010, +0.0037]"]
    ]
  },
  table_5: {
    id: "table_5",
    name: "Table 5: Cross-Source Domain Holdout Generalization",
    columns: ["Evaluation Protocol", "Source -> Target", "Linear SVM F1", "Classical RBF F1", "Quantum Kernel F1", "ΔF1 (Q - RBF)", "p-value (Permutation)"],
    rows: [
      ["Direction A", "TREC 2005/06 -> TREC 2007", "0.7610", "0.7820", "0.7745", "-0.0075", "0.1420"],
      ["Direction B (Canonical)", "TREC 2007 -> TREC 2005/06", "0.7562", "0.6913", "0.6680", "-0.0233", "0.0046 (FDR p=0.0069)"]
    ]
  },
  table_6: {
    id: "table_6",
    name: "Table 6: Inferential Statistical Tests (Exp 38)",
    columns: ["Comparison Target", "Mean ΔF1", "Paired Permutation p", "Wilcoxon Signed-Rank p", "95% Bootstrap CI", "Empirical Outcome"],
    rows: [
      ["MeAJOR 8D IID (N=10)", "+0.0046", "0.0016", "0.0020", "[+0.0030, +0.0061]", "Practically Equivalent (ε=0.01)"],
      ["MeAJOR 12D IID (N=10)", "+0.0014", "0.2824", "0.3125", "[-0.0010, +0.0037]", "Statistical Parity"],
      ["Direction B Holdout (N=10)", "-0.0233", "0.0046", "0.0059", "[-0.0381, -0.0085]", "Classical RBF Superior"]
    ]
  },
  table_7: {
    id: "table_7",
    name: "Table 7: Kernel Geometry & Target Alignment Diagnostics",
    columns: ["Corpus", "Quantum KTA", "Classical RBF KTA", "Alignment Deficit (%)", "Gram Pearson (r)", "State Entropy vs Div (r)"],
    rows: [
      ["SMS Spam Collection (8D)", "0.0382", "0.0768", "-50.3%", "0.5985", "-0.8257"],
      ["CEAS 2008 Phishing (8D)", "0.0220", "0.0603", "-63.5%", "0.6120", "-0.8170"],
      ["MeAJOR Email Archive (8D)", "0.0402", "0.0773", "-48.0%", "0.5890", "-0.7822"]
    ]
  },
  table_8: {
    id: "table_8",
    name: "Table 8: Computational Runtime & Memory Footprint",
    columns: ["Dimension", "Qubits", "Quantum Sim Time (s)", "Classical RBF Time (s)", "Execution Ratio", "Quantum Peak RAM", "RBF Peak RAM"],
    rows: [
      ["2D", "2 Qubits", "3.4s", "1.3s", "2.6x", "142 MB", "95 MB"],
      ["4D", "4 Qubits", "5.8s", "1.3s", "4.5x", "185 MB", "98 MB"],
      ["6D", "6 Qubits", "9.4s", "1.4s", "6.7x", "310 MB", "102 MB"],
      ["8D", "8 Qubits", "17.2s", "1.4s", "12.3x", "680 MB", "108 MB"],
      ["10D", "10 Qubits", "44.6s", "1.5s", "29.7x", "2,150 MB", "115 MB"],
      ["12D", "12 Qubits", "108.8s", "1.7s", "64.0x", "6,400 MB", "122 MB"],
      ["16D", "16 Qubits", "OOM (>1800s)", "2.1s", ">850x", ">10.5 GB", "135 MB"]
    ]
  }
};

export const DEFAULT_EXP41_COMPARISON = [
  { dataset: "sms", representation: "tfidf", quantum_f1: 0.7969, rbf_f1: 0.8091, linear_f1: 0.7927, delta_f1_q_minus_rbf: -0.0122, observed_quantum_edge: "RBF Advantage", canonical_status: "CANONICAL" },
  { dataset: "sms", representation: "minilm", quantum_f1: 0.7736, rbf_f1: 0.7622, linear_f1: 0.6992, delta_f1_q_minus_rbf: 0.0114, observed_quantum_edge: "Quantum Edge", canonical_status: "EXPLORATORY" },
  { dataset: "sms", representation: "roberta", quantum_f1: 0.9006, rbf_f1: 0.9675, linear_f1: 0.9520, delta_f1_q_minus_rbf: -0.0669, observed_quantum_edge: "RBF Advantage", canonical_status: "EXPLORATORY" },
  { dataset: "sms", representation: "mpnet", quantum_f1: 0.7097, rbf_f1: 0.9149, linear_f1: 0.8708, delta_f1_q_minus_rbf: -0.2052, observed_quantum_edge: "RBF Advantage", canonical_status: "EXPLORATORY" },
  { dataset: "ceas", representation: "tfidf", quantum_f1: 0.9504, rbf_f1: 0.9416, linear_f1: 0.9528, delta_f1_q_minus_rbf: 0.0088, observed_quantum_edge: "Practical Parity", canonical_status: "CANONICAL" },
  { dataset: "ceas", representation: "minilm", quantum_f1: 0.9377, rbf_f1: 0.9420, linear_f1: 0.9328, delta_f1_q_minus_rbf: -0.0043, observed_quantum_edge: "Practical Parity", canonical_status: "EXPLORATORY" },
  { dataset: "ceas", representation: "roberta", quantum_f1: 0.9699, rbf_f1: 0.9683, linear_f1: 0.9266, delta_f1_q_minus_rbf: 0.0016, observed_quantum_edge: "Practical Parity", canonical_status: "CANONICAL_ABLATION" },
  { dataset: "ceas", representation: "mpnet", quantum_f1: 0.9606, rbf_f1: 0.9575, linear_f1: 0.9430, delta_f1_q_minus_rbf: 0.0031, observed_quantum_edge: "Practical Parity", canonical_status: "EXPLORATORY" },
  { dataset: "meajor", representation: "tfidf", quantum_f1: 0.8421, rbf_f1: 0.8549, linear_f1: 0.8523, delta_f1_q_minus_rbf: -0.0128, observed_quantum_edge: "RBF Advantage", canonical_status: "CANONICAL" },
  { dataset: "meajor", representation: "minilm", quantum_f1: 0.8163, rbf_f1: 0.8594, linear_f1: 0.8220, delta_f1_q_minus_rbf: -0.0431, observed_quantum_edge: "RBF Advantage", canonical_status: "EXPLORATORY" },
  { dataset: "meajor", representation: "roberta", quantum_f1: 0.8044, rbf_f1: 0.8331, linear_f1: 0.8146, delta_f1_q_minus_rbf: -0.0287, observed_quantum_edge: "RBF Advantage", canonical_status: "EXPLORATORY" },
  { dataset: "meajor", representation: "mpnet", quantum_f1: 0.8064, rbf_f1: 0.8759, linear_f1: 0.8773, delta_f1_q_minus_rbf: -0.0695, observed_quantum_edge: "RBF Advantage", canonical_status: "EXPLORATORY" }
];

export const DEFAULT_EXP41_GEOMETRY = [
  { dataset: "sms", representation: "tfidf", seed: 42, dispersion_entropy_bits: 0.5624, kernel_diversity: 0.1281, target_label_alignment: 0.3008, target_label_alignment_rbf: 0.4756, gram_pearson_r: 0.6430 },
  { dataset: "sms", representation: "minilm", seed: 42, dispersion_entropy_bits: 0.7244, kernel_diversity: 0.0311, target_label_alignment: 0.1327, target_label_alignment_rbf: 0.4188, gram_pearson_r: 0.3478 },
  { dataset: "sms", representation: "roberta", seed: 42, dispersion_entropy_bits: 0.8453, kernel_diversity: 0.0617, target_label_alignment: 0.2715, target_label_alignment_rbf: 0.5492, gram_pearson_r: 0.5723 },
  { dataset: "sms", representation: "mpnet", seed: 42, dispersion_entropy_bits: 0.7101, kernel_diversity: 0.0168, target_label_alignment: 0.1249, target_label_alignment_rbf: 0.5451, gram_pearson_r: 0.3122 },
  { dataset: "sms", representation: "tfidf", seed: 123, dispersion_entropy_bits: 0.5650, kernel_diversity: 0.1453, target_label_alignment: 0.3907, target_label_alignment_rbf: 0.5044, gram_pearson_r: 0.7890 },
  { dataset: "sms", representation: "minilm", seed: 123, dispersion_entropy_bits: 0.6009, kernel_diversity: 0.0671, target_label_alignment: 0.2505, target_label_alignment_rbf: 0.4515, gram_pearson_r: 0.6131 },
  { dataset: "sms", representation: "roberta", seed: 123, dispersion_entropy_bits: 0.7821, kernel_diversity: 0.0348, target_label_alignment: 0.1849, target_label_alignment_rbf: 0.5505, gram_pearson_r: 0.4177 },
  { dataset: "sms", representation: "mpnet", seed: 123, dispersion_entropy_bits: 0.6786, kernel_diversity: 0.0210, target_label_alignment: 0.1274, target_label_alignment_rbf: 0.5420, gram_pearson_r: 0.3269 }
];
