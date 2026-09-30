#!/usr/bin/env python3
"""
Generate an ultra-crisp, high-contrast, perfectly spaced architecture flowchart for Figure 1.
Features:
- Sharp 90-degree rectangular corners.
- Slim, elegant arrow marks (lw=1.8, mut_scale=15).
- Clean routing: 1 -> 2 -> 3 -> 4 -> 5 -> 6 (Stage 3 goes down and directly into Stage 4).
- High text sizes with zero boundary overlap or clipping.
- Rendered at 600 DPI for pristine sharpness.
- Times New Roman / Serif typography matching the paper.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def generate_detailed_fig1():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(13.2, 9.4), dpi=600)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    def draw_sharp_box(x, y, w, h, bg_color, border_color="#0F172A", lw=2.0):
        """Draw crisp rectangular box with sharp 90-degree corners."""
        rect = Rectangle((x, y), w, h, facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
        ax.add_patch(rect)
        return rect

    # Title Banner
    ax.text(50, 97.4, "Figure 1: End-to-End Experimental Architecture & Protocol",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#000000")

    # ==========================================
    # ROW 1: DATA INGESTION & FEATURE PIPELINE
    # y = 67.5 to 92.5 (h = 25.0)
    # ==========================================
    y1, h1 = 67.5, 25.0

    # Box 1: Corpus Ingestion
    draw_sharp_box(2.0, y1, 29.0, h1, "#EFF6FF", "#1D4ED8", lw=2.0)
    ax.text(16.5, y1 + h1 - 3.2, "1. Corpus Ingestion", ha="center", va="center", fontsize=16.5, fontweight="bold", color="#1E3A8A")
    ax.text(16.5, y1 + h1 - 7.5, "153,410 Total Records", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#2563EB")
    ax.text(4.0, y1 + h1 - 12.0, "• SMS Spam (N = 5,572)", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(4.0, y1 + h1 - 16.0, "• CEAS 2008 (N = 39,154)", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(4.0, y1 + h1 - 20.0, "• MeAJOR (N = 108,684)", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")

    # Box 2: Representations
    draw_sharp_box(35.5, y1, 29.0, h1, "#FAF5FF", "#6D28D9", lw=2.0)
    ax.text(50.0, y1 + h1 - 3.2, "2. Representations", ha="center", va="center", fontsize=16.5, fontweight="bold", color="#4C1D95")
    ax.text(50.0, y1 + h1 - 7.5, "Lexical & Contextual", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#7C3AED")
    ax.text(37.5, y1 + h1 - 12.0, "• Sparse TF-IDF (50k vocab)", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(37.5, y1 + h1 - 16.0, "• Dense RoBERTa / MPNet (768D)", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(37.5, y1 + h1 - 20.0, "• Strict Train-Only Isolation", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")

    # Box 3: Information Bottleneck
    draw_sharp_box(69.0, y1, 29.0, h1, "#FFFBEB", "#B45309", lw=2.0)
    ax.text(83.5, y1 + h1 - 3.2, "3. Low-D Bottleneck", ha="center", va="center", fontsize=16.5, fontweight="bold", color="#78350F")
    ax.text(83.5, y1 + h1 - 7.5, "Projections & Angles", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#D97706")
    ax.text(71.0, y1 + h1 - 12.0, r"• Sweep: $d \in \{2, 4, 8, 12, 16\}$", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(71.0, y1 + h1 - 16.0, "• TruncatedSVD & Dense PCA", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(71.0, y1 + h1 - 20.0, r"• Angle Scaling: $[0, \pi]^d$", ha="left", va="top", fontsize=13.0, fontweight="bold", color="#0F172A")

    # ==========================================
    # ROW 2: MODEL INDUCTION & DIAGNOSTICS
    # y = 35.0 to 62.0 (h = 27.0)
    # ==========================================
    y2, h2 = 35.0, 27.0

    # Box 4: Model Induction Hierarchy
    draw_sharp_box(2.0, y2, 46.5, h2, "#F0FDF4", "#15803D", lw=2.0)
    ax.text(25.25, y2 + h2 - 3.0, "4. Model Induction Hierarchy", ha="center", va="center", fontsize=16.0, fontweight="bold", color="#14532D")
    
    # Sub-card 4A: Linear
    draw_sharp_box(3.2, y2 + 1.5, 13.2, h2 - 6.5, "#FFFFFF", "#22C55E", lw=1.4)
    ax.text(9.8, y2 + h2 - 6.5, "Linear Baseline", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#166534")
    ax.text(4.2, y2 + h2 - 9.8, "• Linear SVM\n  (L2 Dual QP)\n• Logistic Reg.\n• Fast convex", ha="left", va="top", fontsize=11.5, fontweight="bold", color="#0F172A", linespacing=1.2)
    
    # Sub-card 4B: Classical RBF
    draw_sharp_box(18.0, y2 + 1.5, 14.8, h2 - 6.5, "#FFFFFF", "#22C55E", lw=1.4)
    ax.text(25.4, y2 + h2 - 6.5, "Classical RBF", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#166534")
    ax.text(19.0, y2 + h2 - 9.8, r"• Matched RBF" "\n" r"  $\gamma = 1/(d \cdot \sigma^2)$" "\n" r"• Tuned RBF" "\n" r"  (5-fold CV)", ha="left", va="top", fontsize=11.5, fontweight="bold", color="#0F172A", linespacing=1.2)

    # Sub-card 4C: Quantum Kernel
    draw_sharp_box(34.4, y2 + 1.5, 13.0, h2 - 6.5, "#FFFFFF", "#22C55E", lw=1.4)
    ax.text(40.9, y2 + h2 - 6.5, "Quantum Kernel", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#166534")
    ax.text(35.4, y2 + h2 - 9.8, r"• ZZFeatureMap" "\n" r"  2-layer cyclic" "\n" r"• $N_q = d$ qubits" "\n" r"• Pure fidelity", ha="left", va="top", fontsize=11.5, fontweight="bold", color="#0F172A", linespacing=1.2)

    # Box 5: Geometric Diagnostics & Audit
    draw_sharp_box(51.5, y2, 46.5, h2, "#FDF2F8", "#BE185D", lw=2.0)
    ax.text(74.75, y2 + h2 - 3.0, "5. Geometric Diagnostics & Audit", ha="center", va="center", fontsize=16.0, fontweight="bold", color="#831843")
    
    ax.text(53.2, y2 + h2 - 7.5, r"• Centered Kernel-Target Alignment (CKA):", ha="left", va="top", fontsize=12.8, fontweight="bold", color="#9D174D")
    ax.text(55.0, y2 + h2 - 10.8, r"  Gram matrix alignment with target $Y = \mathbf{y}\mathbf{y}^T$", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A")
    
    ax.text(53.2, y2 + h2 - 14.5, r"• Spectral Effective Rank ($R_{\mathrm{eff}}$):", ha="left", va="top", fontsize=12.8, fontweight="bold", color="#9D174D")
    ax.text(55.0, y2 + h2 - 17.8, r"  Spectral concentration in induced Hilbert spaces", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A")
    
    ax.text(53.2, y2 + h2 - 21.5, r"• Single-State Dispersion vs Kernel Diversity:", ha="left", va="top", fontsize=12.8, fontweight="bold", color="#9D174D")
    ax.text(55.0, y2 + h2 - 24.8, r"  Entropy $S(|\psi\rangle)$ vs pairwise Gram matrix variance", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A")

    # ==========================================
    # ROW 3: MULTI-REGIME EVALUATION
    # y = 3.0 to 29.0 (h = 26.0)
    # ==========================================
    y3, h3 = 3.0, 26.0
    draw_sharp_box(2.0, y3, 96.0, h3, "#FEF2F2", "#B91C1C", lw=2.0)
    ax.text(50.0, y3 + h3 - 3.0, "6. Multi-Regime Inferential Evaluation & Scalability Limits", ha="center", va="center", fontsize=16.5, fontweight="bold", color="#7F1D1D")

    # Column A: IID Equivalence
    ax.text(4.0, y3 + h3 - 7.2, "A. In-Distribution Equivalence", ha="left", va="top", fontsize=13.8, fontweight="bold", color="#991B1B")
    ax.text(4.0, y3 + h3 - 11.0, r"• TOST ($\epsilon = \pm 0.01$ F1) on MeAJOR" "\n" r"• 10,000 non-parametric bootstrap CIs" "\n" r"• Paired permutation tests (10 seeds)", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A", linespacing=1.25)

    # Column B: Domain Transfer
    ax.text(36.0, y3 + h3 - 7.2, "B. Cross-Source Shift", ha="left", va="top", fontsize=13.8, fontweight="bold", color="#991B1B")
    ax.text(36.0, y3 + h3 - 11.0, "• Train: TREC 2007 (N = 10,000)\n• Test: TREC 2005/2006 (N = 5,000)\n• Benjamini-Hochberg FDR control", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A", linespacing=1.25)

    # Column C: Simulation Cost
    ax.text(68.0, y3 + h3 - 7.2, "C. Simulation & RAM Scaling", ha="left", va="top", fontsize=13.8, fontweight="bold", color="#991B1B")
    ax.text(68.0, y3 + h3 - 11.0, "• Exact PyTorch statevector profiling\n• 64x execution penalty at 12 qubits\n• 16-qubit RAM limit (>10.5 GB)", ha="left", va="top", fontsize=12.0, fontweight="bold", color="#0F172A", linespacing=1.25)

    # ==========================================
    # SLEEK, PROFESSIONAL CONNECTING ARROWS
    # ==========================================
    arrow_color = "#0F172A"
    arrow_lw = 1.8
    mut_scale = 15

    # Arrow 1: Box 1 -> Box 2
    ax.add_patch(FancyArrowPatch((31.0, y1 + h1/2), (35.5, y1 + h1/2),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))
    
    # Arrow 2: Box 2 -> Box 3
    ax.add_patch(FancyArrowPatch((64.5, y1 + h1/2), (69.0, y1 + h1/2),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))
    
    # Arrow 3 -> 4: From Box 3 bottom center down, left, and directly into Box 4 top!
    y_corridor = 64.75
    ax.plot([83.5, 83.5], [y1, y_corridor], color=arrow_color, lw=arrow_lw, zorder=10)
    ax.plot([25.25, 83.5], [y_corridor, y_corridor], color=arrow_color, lw=arrow_lw, zorder=10)
    ax.add_patch(FancyArrowPatch((25.25, y_corridor), (25.25, y2 + h2),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))
    
    # Arrow 4 -> 5: From Box 4 right side into Box 5 left side!
    ax.add_patch(FancyArrowPatch((48.5, y2 + h2/2), (51.5, y2 + h2/2),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))

    # Arrow 5 -> 6: Downward from Box 4 and Box 5 into Box 6 top!
    ax.add_patch(FancyArrowPatch((25.25, y2), (25.25, y3 + h3),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))
    ax.add_patch(FancyArrowPatch((74.75, y2), (74.75, y3 + h3),
                                 arrowstyle="-|>", mutation_scale=mut_scale, color=arrow_color, lw=arrow_lw, zorder=10))

    # Save at 600 DPI for pristine sharpness
    out_paths = [
        "/Users/pavanaksshay/quantum/paper/submission/figures/figure_1_experimental_framework.png",
        "/Users/pavanaksshay/quantum/results/exp39_paper/figures/figure_1_experimental_framework.png"
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=600, pad_inches=0.03)
        print(f"Saved (600 DPI): {p}")
    plt.close()

if __name__ == "__main__":
    generate_detailed_fig1()
