import os
import base64

def generate_html_paper():
    base_dir = os.path.dirname(__file__)
    figures_dir = os.path.join(base_dir, "submission", "figures")
    
    # Read figures as base64 for standalone portability
    fig_data = {}
    for i in range(1, 9):
        fname = f"figure_{i}_"
        match = [f for f in os.listdir(figures_dir) if f.startswith(fname) and f.endswith(".png")]
        if match:
            with open(os.path.join(figures_dir, match[0]), "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                fig_data[f"fig{i}"] = f"data:image/png;base64,{b64}"
                fig_data[f"fig{i}_name"] = match[0]

    template_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>When Do Quantum Kernels Help for Text Security? — Research Manuscript</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #f8fafc;
    --bg-paper: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --text-heading: #0f172a;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --border-color: #e2e8f0;
    --card-bg: #f1f5f9;
    --accent: #0284c7;
    --badge-quantum: #e0f2fe;
    --badge-quantum-text: #0369a1;
  }

  [data-theme="dark"] {
    --bg-primary: #0b0f19;
    --bg-paper: #111827;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --text-heading: #f8fafc;
    --primary: #38bdf8;
    --primary-hover: #0ea5e9;
    --border-color: #1f2937;
    --card-bg: #1e293b;
    --accent: #38bdf8;
    --badge-quantum: #0c4a6e;
    --badge-quantum-text: #7dd3fc;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background-color: var(--bg-primary);
    color: var(--text-main);
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 17px;
    line-height: 1.65;
    transition: background-color 0.2s ease, color 0.2s ease;
  }

  /* Toolbar */
  .top-toolbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(8px);
    color: #f8fafc;
    padding: 10px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
  }

  .toolbar-title {
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .toolbar-badge {
    background: #0284c7;
    color: white;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.5px;
  }

  .toolbar-actions {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .btn {
    background: #334155;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 12px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    text-decoration: none;
    transition: all 0.15s ease;
  }

  .btn:hover {
    background: #475569;
    transform: translateY(-1px);
  }

  .btn-primary {
    background: #2563eb;
  }

  .btn-primary:hover {
    background: #1d4ed8;
  }

  /* Paper Container */
  .paper-container {
    max-width: 1060px;
    margin: 32px auto;
    background: var(--bg-paper);
    padding: 64px 72px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    border: 1px solid var(--border-color);
  }

  /* Header Section */
  header {
    text-align: center;
    margin-bottom: 36px;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 28px;
  }

  h1.paper-title {
    font-family: 'Inter', sans-serif;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.25;
    color: var(--text-heading);
    margin-bottom: 12px;
    letter-spacing: -0.5px;
  }

  .paper-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 17px;
    font-weight: 500;
    color: var(--accent);
    margin-bottom: 18px;
  }

  .author-block {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.5;
  }

  .author-names {
    font-weight: 600;
    color: var(--text-heading);
    font-size: 15px;
    margin-bottom: 4px;
  }

  /* Abstract Box */
  .abstract-card {
    background: var(--card-bg);
    border-left: 4px solid var(--accent);
    padding: 20px 24px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 36px;
    font-size: 15px;
    line-height: 1.6;
  }

  .abstract-card h3 {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
  }

  /* Content Grid */
  .paper-grid {
    display: block;
  }

  .paper-grid.two-col {
    column-count: 2;
    column-gap: 36px;
    column-rule: 1px solid var(--border-color);
  }

  /* Typography */
  h2.section-heading {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-heading);
    margin-top: 32px;
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-color);
    break-after: avoid;
  }

  h3.subsection-heading {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-heading);
    margin-top: 20px;
    margin-bottom: 8px;
    break-after: avoid;
  }

  p {
    margin-bottom: 14px;
    text-align: justify;
    text-justify: inter-word;
  }

  ul, ol {
    margin-left: 20px;
    margin-bottom: 16px;
  }

  li {
    margin-bottom: 6px;
  }

  /* Tables */
  .table-wrapper {
    margin: 24px 0;
    overflow-x: auto;
    break-inside: avoid;
  }

  table.academic-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 1.4;
    border-top: 2px solid var(--text-heading);
    border-bottom: 2px solid var(--text-heading);
  }

  table.academic-table th {
    font-weight: 600;
    text-align: center;
    padding: 8px 10px;
    border-bottom: 1px solid var(--text-heading);
    background: var(--card-bg);
    color: var(--text-heading);
  }

  table.academic-table td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border-color);
    text-align: center;
  }

  table.academic-table td.text-left {
    text-align: left;
  }

  table.academic-table tr:hover td {
    background: var(--card-bg);
  }

  .table-caption {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
    color: var(--text-heading);
    text-align: left;
  }

  /* Figures */
  .figure-card {
    margin: 28px 0;
    break-inside: avoid;
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }

  .figure-card img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    cursor: zoom-in;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    background: white;
    transition: transform 0.2s ease;
  }

  .figure-card img:hover {
    transform: scale(1.01);
  }

  .fig-caption {
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    color: var(--text-muted);
    margin-top: 10px;
    text-align: left;
    line-height: 1.45;
  }

  .fig-caption strong {
    color: var(--text-heading);
  }

  /* Equation Box */
  .equation-box {
    background: var(--card-bg);
    border-left: 3px solid var(--primary);
    padding: 12px 18px;
    border-radius: 0 6px 6px 0;
    margin: 16px 0;
    font-size: 14px;
    overflow-x: auto;
    break-inside: avoid;
  }

  /* Key Takeaway Callout */
  .callout-box {
    background: var(--badge-quantum);
    border: 1px solid var(--accent);
    color: var(--badge-quantum-text);
    padding: 16px 20px;
    border-radius: 8px;
    margin: 20px 0;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.5;
    break-inside: avoid;
  }

  .callout-box strong {
    display: block;
    margin-bottom: 4px;
    font-size: 15px;
  }

  /* Lightbox */
  .lightbox {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(5px);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    padding: 20px;
  }

  .lightbox.active {
    display: flex;
  }

  .lightbox img {
    max-width: 90%;
    max-height: 90%;
    border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }

  .lightbox-close {
    position: absolute;
    top: 20px;
    right: 24px;
    color: white;
    font-size: 32px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
  }

  @media print {
    .top-toolbar, .btn, .lightbox {
      display: none !important;
    }
    body {
      background: white !important;
      color: black !important;
      font-size: 10pt;
    }
    .paper-container {
      max-width: 100% !important;
      margin: 0 !important;
      padding: 0 !important;
      border: none !important;
      box-shadow: none !important;
    }
    .figure-card {
      border: 1px solid #ccc !important;
      background: white !important;
      page-break-inside: avoid;
    }
    h2.section-heading {
      page-break-after: avoid;
    }
    table.academic-table {
      page-break-inside: avoid;
    }
  }
</style>
</head>
<body>

<div class="top-toolbar">
  <div class="toolbar-title">
    <span>Quantum Text Security Benchmark</span>
    <span class="toolbar-badge">Peer Review / Faculty Presentation</span>
  </div>
  <div class="toolbar-actions">
    <button class="btn" onclick="toggleLayout()" id="layoutBtn">Switch to 2-Column View</button>
    <button class="btn" onclick="toggleTheme()" id="themeBtn">🌙 Dark Mode</button>
    <button class="btn btn-primary" onclick="window.print()">🖨️ Print / Save PDF</button>
    <a href="RESEARCH_PAPER.pdf" class="btn" download>📥 Download Compiled PDF</a>
  </div>
</div>

<div class="paper-container">
  <header>
    <h1 class="paper-title">When Do Quantum Kernels Help for Text Security?<br>A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost</h1>
    <div class="paper-subtitle">An Audited Empirical Benchmark on 150,000+ Text Records</div>
    <div class="author-block">
      <div class="author-names">Research Manuscript prepared for Academic & Faculty Review</div>
      <div>Department of Computer Science &bull; Quantum Machine Learning & Natural Language Security Group</div>
      <div>Controlled Confirmation Protocol &bull; 10 Random Seeds &bull; Leakage-Safe Evaluation</div>
    </div>
  </header>

  <div class="abstract-card">
    <h3>Abstract</h3>
    <p>
      Quantum kernel methods have attracted substantial interest for natural language processing and cybersecurity, motivated by theoretical conjectures that mapping classical data into exponentially large Hilbert spaces could yield superior non-linear decision boundaries or enhanced generalization under distribution shift. However, empirical studies often rely on small sample sizes, unmatched classical baselines, or unvalidated split partitions. In this work, we present a controlled empirical evaluation comparing parameter-free quantum fidelity kernels with matched classical radial basis function (RBF) kernels across three benchmark corpora drawing from more than 150,000 audited text records: the <strong>SMS Spam Collection</strong>, the <strong>CEAS 2008 Email Corpus</strong>, and the multi-source <strong>MeAJOR archive</strong>. Using frozen, leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling (2–12 qubits), cross-source domain holdouts (TREC 2007 &rarr; TREC 2005/2006), feature space geometry, and computational overhead.
    </p>
    <p>
      Under matched in-distribution (IID) conditions, the quantum kernel is competitive with classical RBF, displaying minor statistically detectable improvements at intermediate dimensions (+0.46 percentage points at 8D, \(p = 0.0016\); +0.57 pp at 10D, \(p = 0.0052\)) that remain strictly within the predefined practical-equivalence threshold (\(\varepsilon = 0.01\) F1), converging to complete parity at 12D (+0.14 pp, \(p = 0.2824\)). Under cross-source domain transfer, the quantum kernel exhibits a statistically significant and practically meaningful performance deficit (\(\Delta\text{F1} = -0.0233\), \(p = 0.0046\), Benjamini–Hochberg FDR \(p = 0.0069\)). Representation ablations demonstrate that upstream feature representation dominates kernel selection by an order of magnitude: switching from TF-IDF to dense RoBERTa embeddings shifts relative performance by 3.90 percentage points. Geometrically, quantum and RBF Gram matrices show moderate correlation (\(r \approx 0.55\text{--}0.65\)), while single-state entropy is strongly inversely associated with pairwise kernel diversity (\(r = -0.78\) to \(-0.83\)). Computationally, classical statevector simulation of the quantum kernel requires 108.8s per run at 12D compared to 1.7s for RBF (\(\approx 64\times\) penalty). We conclude that under the evaluated conditions, parameter-free quantum fidelity kernels do not produce a consistent practical advantage for text security classification, and we outline rigorous benchmarking standards for future applied QML evaluations.
    </p>
  </div>

  <div id="paperGrid" class="paper-grid">
    <h2 class="section-heading">1. Introduction & Scientific Motivation</h2>
    <p>
      Text-based social engineering attacks, including email phishing, SMS scams, and fraudulent communications, represent one of the most pervasive threat vectors in modern digital infrastructure. Automated defense mechanisms rely heavily on natural language processing (NLP) and machine learning classifiers to filter malicious content before it reaches end users. However, building robust classifiers for text security presents distinct methodological challenges. Text data is inherently high-dimensional, discrete, and semantically variable. Furthermore, security environments are characterized by persistent distribution shift: adversaries continuously modify lexical patterns to evade detection filters, attack campaigns differ substantially across organizational sources, and seasonal shifts alter background communications.
    </p>
    <p>
      In recent years, quantum machine learning (QML) has emerged as an alternative paradigm for non-linear pattern recognition, with quantum kernel methods receiving particular theoretical attention. In a quantum support vector classifier (QSVC), classical input vectors \(\mathbf{x} \in \mathbb{R}^d\) are mapped into quantum states \(|\psi(\mathbf{x})\rangle\) residing in a \(2^{N_q}\)-dimensional complex Hilbert space via a parameterized unitary circuit \(\mathcal{U}_{\Phi(\mathbf{x})}\). Rather than performing explicit optimization in this exponentially large Hilbert space, the model evaluates pairwise quantum state fidelities to construct a Gram matrix, \(k(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2\), which is subsequently supplied to a standard dual quadratic program.
    </p>
"""

    fig1_html = f"""
    <div class="figure-card">
      <img src="{fig_data.get('fig1', '')}" alt="Figure 1: End-to-end experimental framework" onclick="openLightbox(this.src)">
      <div class="fig-caption">
        <strong>Figure 1: End-to-End Leakage-Safe Experimental Framework.</strong> Raw text corpora are tokenized and vectorized strictly within training splits. Low-dimensional projections (\(d \in [2, 12]\)) are mapped in parallel to the Quantum ZZ-Feature Map (statevector fidelity) and matched Classical Gaussian RBF Kernel before dual SVM optimization.
      </div>
    </div>
"""

    section2_html = """
    <h2 class="section-heading">2. Methodology and Controlled Experimental Design</h2>
    <p>
      To eliminate confounding factors present in prior literature, we enforce strict experimental hygiene across all benchmark pipelines:
    </p>
    <ul>
      <li><strong>Data Leakage Prevention:</strong> Feature extractors (TF-IDF up to 50,000 n-grams) and linear dimensionality reduction transformers (TruncatedSVD + StandardScaler) are fitted <em>strictly on training partitions</em>. Validation and test partitions are transformed out-of-sample.</li>
      <li><strong>Identical Feature Parity:</strong> Both quantum and classical models receive identical real vectors \(\mathbf{x} \in [0, \pi]^d\).</li>
      <li><strong>Canonical Quantum Feature Map:</strong> We implement the parameter-free two-layer cyclic \(ZZFeatureMap\):
        <div class="equation-box">
          \[ \mathcal{U}_{\Phi(\mathbf{x})} = \left( U_{\Phi(\mathbf{x})} H^{\otimes N_q} \right)^2 \]
          \[ U_{\Phi(\mathbf{x})} = \exp\left( i \sum_{j=1}^{N_q} x_j Z_j + i \sum_{j=1}^{N_q} (\pi - x_j)(\pi - x_{(j \bmod N_q) + 1}) Z_j Z_{(j \bmod N_q) + 1} \right) \]
          \[ K_Q(\mathbf{x}, \mathbf{z}) = |\langle \psi(\mathbf{x}) | \psi(\mathbf{z}) \rangle|^2 \]
        </div>
      </li>
      <li><strong>Matched Classical Baseline:</strong> Gaussian RBF kernel with dynamic variance-scaled gamma:
        <div class="equation-box">
          \[ K_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left( -\frac{1}{d \cdot \text{Var}(X)} \|\mathbf{x} - \mathbf{z}\|_2^2 \right) \]
        </div>
      </li>
      <li><strong>Replication Suite:</strong> All comparisons execute across 10 deterministic seeds: <code>[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]</code>.</li>
    </ul>

    <div class="table-wrapper">
      <div class="table-caption">Table 1: Benchmark Dataset Characteristics and Audit Partitions</div>
      <table class="academic-table">
        <thead>
          <tr>
            <th class="text-left">Corpus</th>
            <th>Raw Records</th>
            <th>Usable Size</th>
            <th>Spam/Phish %</th>
            <th>Train / Val / Test</th>
            <th class="text-left">Domain Profile</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="text-left"><strong>SMS Spam</strong></td>
            <td>5,574</td>
            <td>5,572</td>
            <td>13.41%</td>
            <td>3,343 / 1,114 / 1,115</td>
            <td class="text-left">Mobile carrier SMS messages</td>
          </tr>
          <tr>
            <td class="text-left"><strong>CEAS 2008</strong></td>
            <td>39,154</td>
            <td>15,000</td>
            <td>18.90%</td>
            <td>10,000 / 2,500 / 2,500</td>
            <td class="text-left">Phishing vs Ham email collections</td>
          </tr>
          <tr>
            <td class="text-left"><strong>MeAJOR Archive</strong></td>
            <td>108,685</td>
            <td>108,684</td>
            <td>19.33%</td>
            <td>10,000 / 2,500 / 2,500</td>
            <td class="text-left">Multi-source enterprise archive (TREC 5/6/7)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h2 class="section-heading">3. Empirical Results and Scientific Findings</h2>

    <h3 class="subsection-heading">3.1 In-Distribution Dimensionality Scaling</h3>
    <p>
      We evaluated scaling behavior across dimensionality \(d \in \{2, 4, 6, 8, 10, 12\}\) on the canonical MeAJOR in-distribution split across 10 random seeds (Table 2 and Figures 2–3).
    </p>

    <div class="table-wrapper">
      <div class="table-caption">Table 2: Dimensionality Scaling Profile on MeAJOR IID (N = 10 Seeds)</div>
      <table class="academic-table">
        <thead>
          <tr>
            <th>Dim (\(d\))</th>
            <th>Quantum F1 (\(\bar{x} \pm s\))</th>
            <th>Classical RBF F1</th>
            <th>Paired \(\Delta\)F1</th>
            <th>95% Bootstrap CI</th>
            <th>Permutation \(p\)</th>
            <th>Status (\(\varepsilon = 0.01\))</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>2D</td>
            <td>0.6447 &plusmn; 0.0038</td>
            <td>0.6735 &plusmn; 0.0031</td>
            <td>-0.0288</td>
            <td>[-0.0312, -0.0264]</td>
            <td>&lt; 0.001</td>
            <td>Classical Advantage</td>
          </tr>
          <tr>
            <td>4D</td>
            <td>0.7876 &plusmn; 0.0030</td>
            <td>0.7918 &plusmn; 0.0025</td>
            <td>-0.0042</td>
            <td>[-0.0065, -0.0019]</td>
            <td>0.0180</td>
            <td>Practically Equivalent</td>
          </tr>
          <tr>
            <td>6D</td>
            <td>0.8253 &plusmn; 0.0026</td>
            <td>0.8254 &plusmn; 0.0022</td>
            <td>-0.0001</td>
            <td>[-0.0021, +0.0018]</td>
            <td>0.9410</td>
            <td>Practically Equivalent</td>
          </tr>
          <tr>
            <td>8D</td>
            <td>0.8754 &plusmn; 0.0029</td>
            <td>0.8709 &plusmn; 0.0030</td>
            <td>+0.0046</td>
            <td>[+0.0030, +0.0061]</td>
            <td>0.0016</td>
            <td>Practically Equivalent</td>
          </tr>
          <tr>
            <td>10D</td>
            <td>0.9023 &plusmn; 0.0034</td>
            <td>0.8967 &plusmn; 0.0049</td>
            <td>+0.0057</td>
            <td>[+0.0025, +0.0089]</td>
            <td>0.0052</td>
            <td>Practically Equivalent</td>
          </tr>
          <tr>
            <td>12D</td>
            <td>0.9137 &plusmn; 0.0046</td>
            <td>0.9123 &plusmn; 0.0023</td>
            <td>+0.0014</td>
            <td>[-0.0010, +0.0037]</td>
            <td>0.2824</td>
            <td><strong>Strict Statistical Parity</strong></td>
          </tr>
        </tbody>
      </table>
    </div>
"""

    figs23_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
      <div class="figure-card">
        <img src="{fig_data.get('fig2', '')}" alt="Figure 2: In-distribution F1 scaling" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 2:</strong> In-distribution test F1 scaling vs dimensionality \(d \in [2, 12]\) across 10 random seeds. Error bands denote 95% bootstrap CIs.</div>
      </div>
      <div class="figure-card">
        <img src="{fig_data.get('fig3', '')}" alt="Figure 3: Paired Delta F1" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 3:</strong> Paired \(\Delta\text{{F1}}\) (Quantum minus Classical RBF) across dimensionalities. The green band shows the \(\pm 0.01\) practical-equivalence margin.</div>
      </div>
    </div>
"""

    section3_rest_html = f"""
    <h3 class="subsection-heading">3.2 Cross-Source Out-of-Distribution Domain Generalization</h3>
    <p>
      In cybersecurity applications, classifiers must generalize to unseen organizations and distribution shifts. We executed cross-source domain holdouts (Direction B: training strictly on TREC 2007 and testing on unseen TREC 2005/2006 emails across 10 seeds).
    </p>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
      <div class="figure-card">
        <img src="{fig_data.get('fig4', '')}" alt="Figure 4: Domain holdout performance" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 4:</strong> In-distribution vs Domain Holdout degradation. The quantum kernel suffers a severe 23.7% drop compared to 20.6% for RBF.</div>
      </div>
      <div class="figure-card">
        <img src="{fig_data.get('fig6', '')}" alt="Figure 6: Representation interaction" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 6:</strong> Text representation interaction (TF-IDF vs RoBERTa) causing complete quantum-vs-classical ranking reversal.</div>
      </div>
    </div>

    <div class="callout-box">
      <strong>Key Out-of-Distribution Finding:</strong>
      Under domain shift, Classical Gaussian RBF achieves Holdout F1 = \(0.6913 \pm 0.0161\), significantly outperforming the Quantum Kernel (\(0.6680 \pm 0.0094\)). The paired gap is \(\Delta\text{{F1}} = -0.0233\) (\(p = 0.0046\), Benjamini-Hochberg adjusted \(p = 0.0069\)). The quantum feature map exhibits heightened sensitivity to lexical distribution shift.
    </div>

    <h3 class="subsection-heading">3.3 Upstream Representation Dominance & Ranking Reversal</h3>
    <p>
      Evaluating feature representations on CEAS 2008 reveals that upstream representations dominate kernel choice by an order of magnitude:
    </p>
    <ul>
      <li><strong>8D TF-IDF + TruncatedSVD:</strong> Quantum \(\text{{F1}} = 0.9736\) vs Classical RBF \(\text{{F1}} = 0.9641\) (+0.95 pp quantum advantage).</li>
      <li><strong>8D Dense RoBERTa Embeddings:</strong> Quantum \(\text{{F1}} = 0.9601\) vs Classical RBF \(\text{{F1}} = 0.9896\) (-2.95 pp classical advantage).</li>
    </ul>
    <p>
      The net representation shift is <strong>3.90 percentage points</strong>. Dense contextual transformer embeddings provide continuous semantic clustering that classical Gaussian RBF exploits directly, whereas periodic quantum phase encodings disrupt continuous Euclidean metric distance.
    </p>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
      <div class="figure-card">
        <img src="{fig_data.get('fig7', '')}" alt="Figure 7: Gram matrix correlation" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 7:</strong> Gram matrix Pearson correlation between Quantum and RBF across dimensionalities (\(r \approx 0.55\text{{--}}0.65\)).</div>
      </div>
      <div class="figure-card">
        <img src="{fig_data.get('fig8', '')}" alt="Figure 8: Entropy vs Diversity" onclick="openLightbox(this.src)">
        <div class="fig-caption"><strong>Figure 8:</strong> Inverse relationship between single-state von Neumann entropy and pairwise Gram diversity (\(r = -0.78\) to \(-0.83\)).</div>
      </div>
    </div>

    <h3 class="subsection-heading">3.4 Computational Simulation Overhead Scaling</h3>
    <p>
      Wall-clock execution times and memory requirements for classical statevector simulation scale exponentially with dimensionality:
    </p>

    <div class="table-wrapper">
      <div class="table-caption">Table 3: Computational Execution Cost Benchmark on 10,000 Text Samples</div>
      <table class="academic-table">
        <thead>
          <tr>
            <th>Qubits / Dim (\(d\))</th>
            <th>Quantum Sim Time</th>
            <th>Classical RBF Time</th>
            <th>Runtime Overhead</th>
            <th>Quantum Peak Memory</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>2D</td>
            <td>3.4 s</td>
            <td>1.3 s</td>
            <td>2.6&times;</td>
            <td>142 MB</td>
          </tr>
          <tr>
            <td>4D</td>
            <td>5.8 s</td>
            <td>1.3 s</td>
            <td>4.5&times;</td>
            <td>185 MB</td>
          </tr>
          <tr>
            <td>8D</td>
            <td>17.2 s</td>
            <td>1.4 s</td>
            <td>12.3&times;</td>
            <td>680 MB</td>
          </tr>
          <tr>
            <td>12D</td>
            <td>108.8 s</td>
            <td>1.7 s</td>
            <td><strong>64.0&times;</strong></td>
            <td>6.4 GB</td>
          </tr>
          <tr>
            <td>16D</td>
            <td>Infeasible (&gt;10.5 GB)</td>
            <td>2.1 s</td>
            <td>&infin;</td>
            <td>&gt;10.5 GB (OOM)</td>
          </tr>
        </tbody>
      </table>
    </div>

    <h2 class="section-heading">4. Discussion & Theoretical Implications</h2>
    <p>
      Our findings provide critical nuance to recent literature claiming quantum advantage in text classification:
    </p>
    <ol>
      <li><strong>Artifact of Baseline Matching:</strong> In prior studies reporting substantial quantum superiority, classical models were either un-regularized or trained on disparate feature sets. When strictly matched, the quantum advantage shrinks into statistical insignificance or practical equivalence (\(\le 0.5\%\)).</li>
      <li><strong>Absence of Domain Invariance:</strong> Quantum state preparation circuits do not impart magical inductive biases against semantic distribution shift; out-of-distribution transfer requires explicit representation alignment.</li>
      <li><strong>Engineering Trade-Offs:</strong> A \(64\times\) runtime cost to achieve exact performance parity with a 1.7-second classical RBF kernel represents an impractical trade-off for real-world production cybersecurity pipelines.</li>
    </ol>

    <h2 class="section-heading">5. Conclusion & Recommendations</h2>
    <p>
      Under rigorously controlled experimental conditions, parameter-free quantum fidelity kernels do not provide a consistent or practically meaningful advantage over classical Gaussian RBF kernels for text security classification. We establish that future applied QML evaluations must enforce five core benchmarking standards: (1) zero-leakage split pipelines, (2) strictly matched classical nonlinear controls, (3) multi-seed statistical significance testing with practical-equivalence margins, (4) cross-domain out-of-distribution evaluation, and (5) exact computational overhead accounting.
    </p>

    <h2 class="section-heading">References</h2>
    <ol style="font-size: 13px; line-height: 1.5; font-family: 'Inter', sans-serif;">
      <li>V. Havlicek, et al., "Supervised learning with quantum-enhanced feature spaces," <em>Nature</em>, vol. 567, no. 7747, pp. 209–213, 2019.</li>
      <li>M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," <em>Phys. Rev. Lett.</em>, vol. 122, no. 4, p. 040504, 2019.</li>
      <li>H.-Y. Huang, et al., "Power of data in quantum machine learning," <em>Nature Communications</em>, vol. 12, no. 1, p. 2631, 2021.</li>
      <li>S. Thanasilp, et al., "Exponential concentration and expressivity bottlenecks in quantum kernel methods," <em>PRX Quantum</em>, 2024.</li>
      <li>J. Kübler, et al., "The inductive bias of quantum kernels," <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 2021.</li>
      <li>A. Hridi, et al., "Batch-processed quantum machine learning for email phishing detection," <em>IEEE Access</em>, 2026.</li>
      <li>C. Cortes and V. Vapnik, "Support-vector networks," <em>Machine Learning</em>, vol. 20, no. 3, pp. 273–297, 1995.</li>
    </ol>
  </div>
</div>

<div id="lightbox" class="lightbox" onclick="closeLightbox()">
  <span class="lightbox-close">&times;</span>
  <img id="lightboxImg" src="" alt="Enlarged figure">
</div>

<script>
  let isTwoCol = false;
  function toggleLayout() {
    const grid = document.getElementById('paperGrid');
    const btn = document.getElementById('layoutBtn');
    isTwoCol = !isTwoCol;
    if (isTwoCol) {
      grid.classList.add('two-col');
      btn.textContent = 'Switch to 1-Column View';
    } else {
      grid.classList.remove('two-col');
      btn.textContent = 'Switch to 2-Column View';
    }
  }

  let isDark = false;
  function toggleTheme() {
    isDark = !isDark;
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    document.getElementById('themeBtn').textContent = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
  }

  function openLightbox(src) {
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightbox').classList.add('active');
  }

  function closeLightbox() {
    document.getElementById('lightbox').classList.remove('active');
  }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeLightbox();
  });
</script>
</body>
</html>
"""
    output_html_path = os.path.join(base_dir, "RESEARCH_PAPER.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(template_head + fig1_html + section2_html + figs23_html + section3_rest_html)
    print(f"Successfully generated standalone HTML paper: {output_html_path}")

if __name__ == "__main__":
    generate_html_paper()
