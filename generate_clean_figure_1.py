import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def generate_clean_figure_1():
    plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig, ax = plt.subplots(figsize=(14.5, 5.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # 5 pipeline stages with distinct spacing
    stages = [
        {
            "num": "STAGE 1",
            "title": "TEXT CORPORA",
            "subtitle": "Benchmark Datasets",
            "x": 0.02, "w": 0.165,
            "header_bg": "#1E3A8A", # Deep Royal Blue
            "card_bg": "#F8FAFC",
            "border": "#CBD5E1",
            "badge_color": "#DBEAFE",
            "badge_text": "#1E40AF",
            "items": [
                ("SMS Spam Benchmark", True),
                ("  5,572 audited messages", False),
                ("CEAS08 Benchmark", True),
                ("  15,000 email records", False),
                ("MeAJOR Benchmark", True),
                ("  108,684 raw emails", False),
                ("Audit & Deduplication", True),
            ]
        },
        {
            "num": "STAGE 2",
            "title": "REPRESENTATION",
            "subtitle": "Feature Encodings",
            "x": 0.22, "w": 0.165,
            "header_bg": "#581C87", # Rich Purple
            "card_bg": "#FAF5FF",
            "border": "#E9D5FF",
            "badge_color": "#F3E8FF",
            "badge_text": "#6B21A8",
            "items": [
                ("TF-IDF N-grams", True),
                ("  Word (1-2) + Char (3-5)", False),
                ("  Vocabulary: 5,000 terms", False),
                ("RoBERTa Dense", True),
                ("  768D Contextual Embs", False),
                ("L2 Vector Normalization", True),
                ("Controlled Sparsity", False),
            ]
        },
        {
            "num": "STAGE 3",
            "title": "BOTTLENECK",
            "subtitle": "Dimension Projection",
            "x": 0.42, "w": 0.165,
            "header_bg": "#9A3412", # Terracotta / Amber
            "card_bg": "#FFFBEB",
            "border": "#FDE68A",
            "badge_color": "#FEF3C7",
            "badge_text": "#B45309",
            "items": [
                ("PCA Compression", True),
                ("  d in {2, 4, 6, 8, 10, 12, 16}", False),
                ("Qubit Regimes", True),
                ("  Statevector simulation", False),
                ("Spectral Energy Decay", True),
                ("  Variance preservation", False),
                ("Rank-Constrained Space", False),
            ]
        },
        {
            "num": "STAGE 4",
            "title": "MODEL INDUCTION",
            "subtitle": "Kernel Estimation",
            "x": 0.62, "w": 0.165,
            "header_bg": "#064E3B", # Deep Forest
            "card_bg": "#F0FDF4",
            "border": "#A7F3D0",
            "badge_color": "#D1FAE5",
            "badge_text": "#047857",
            "items": [
                ("Linear SVM Baseline", True),
                ("  Primal O(d) complexity", False),
                ("Classical RBF Kernel", True),
                ("  K(x, z) = exp(-g||x-z||^2)", False),
                ("Quantum ZZFeatureMap", True),
                ("  U_Phi(x) |0^d> Statevector", False),
                ("Exact Gram Matrix Eval", False),
            ]
        },
        {
            "num": "STAGE 5",
            "title": "EVALUATION",
            "subtitle": "Multi-Regime Protocol",
            "x": 0.82, "w": 0.160,
            "header_bg": "#881337", # Deep Crimson
            "card_bg": "#FFF1F2",
            "border": "#FECDD3",
            "badge_color": "#FFE4E6",
            "badge_text": "#BE123C",
            "items": [
                ("Frozen IID Benchmark", True),
                ("  Matched 5-fold cross-val", False),
                ("Directional Transfer", True),
                ("  Dir A: CEAS -> MeAJOR", False),
                ("  Dir B: MeAJOR -> CEAS", False),
                ("Statistical Rigor", True),
                ("  10k Bootstrap CIs & Tests", False),
            ]
        }
    ]

    card_y = 0.08
    card_h = 0.74
    header_h = 0.155

    for st in stages:
        x, w = st["x"], st["w"]
        
        # Subtle Drop Shadow
        shadow = FancyBboxPatch((x + 0.003, card_y - 0.006), w, card_h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                ec="none", fc="#0F172A", alpha=0.06, zorder=1)
        ax.add_patch(shadow)

        # Main Card Body
        body_patch = FancyBboxPatch((x, card_y), w, card_h,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    ec=st["border"], fc=st["card_bg"], lw=1.5, zorder=2)
        ax.add_patch(body_patch)

        # Header Box
        header_y = card_y + card_h - header_h
        header_patch = FancyBboxPatch((x, header_y), w, header_h,
                                      boxstyle="round,pad=0.012,rounding_size=0.02",
                                      ec=st["header_bg"], fc=st["header_bg"], lw=1.2, zorder=3)
        ax.add_patch(header_patch)

        # Stage Badge Pill inside header
        badge_w, badge_h = 0.075, 0.032
        badge_x = x + (w - badge_w) / 2
        badge_y = header_y + header_h - 0.046
        badge_patch = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                                     boxstyle="round,pad=0.004,rounding_size=0.01",
                                     ec="none", fc=st["badge_color"], zorder=4)
        ax.add_patch(badge_patch)
        ax.text(x + w/2, badge_y + badge_h/2, st["num"],
                ha="center", va="center", fontsize=7.2, fontweight="bold",
                color=st["badge_text"], zorder=5)

        # Header Title
        ax.text(x + w/2, header_y + 0.042, st["title"],
                ha="center", va="center", fontsize=8.8, fontweight="bold",
                color="#FFFFFF", zorder=5)

        # Subtitle below header box
        ax.text(x + w/2, header_y - 0.028, st["subtitle"].upper(),
                ha="center", va="center", fontsize=6.8, fontweight="bold",
                color=st["header_bg"], zorder=4)

        # Divider line
        ax.plot([x + 0.012, x + w - 0.012], [header_y - 0.046, header_y - 0.046],
                color=st["border"], lw=0.9, zorder=4)

        # Items
        start_item_y = header_y - 0.076
        line_spacing = 0.070
        for idx, (text_val, is_heading) in enumerate(st["items"]):
            item_y = start_item_y - idx * line_spacing
            if is_heading:
                ax.text(x + 0.010, item_y, text_val,
                        ha="left", va="center", fontsize=7.8, fontweight="bold",
                        color="#0F172A", zorder=4)
            else:
                ax.text(x + 0.010, item_y, text_val,
                        ha="left", va="center", fontsize=7.1, fontweight="normal",
                        color="#475569", zorder=4)

    # Connector Arrows with circular pill connector styling
    arrow_y = card_y + card_h * 0.48
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"] + 0.006
        x2 = stages[i+1]["x"] - 0.006
        
        arrow = FancyArrowPatch((x1, arrow_y), (x2, arrow_y),
                                arrowstyle="-|>", mutation_scale=14,
                                color="#475569", lw=2.0, zorder=5)
        ax.add_patch(arrow)

    # Title & Subtitle banner
    ax.text(0.5, 0.945, "Figure 1: Methodological Architecture of the Quantum vs Classical NLP Study",
            ha="center", va="center", fontsize=12.2, fontweight="bold",
            color="#0F172A")
    ax.text(0.5, 0.895, "Comprehensive 5-stage pipeline across frozen IID scaling, cross-domain transfer, and paired non-parametric significance testing",
            ha="center", va="center", fontsize=8.6, fontstyle="italic",
            color="#475569")

    plt.tight_layout(pad=0.4)

    target_dirs = [
        "/Users/pavanaksshay/quantum/paper/submission/figures",
        "/Users/pavanaksshay/quantum/results/exp47/figures",
        "/Users/pavanaksshay/quantum/results/exp39_paper/figures"
    ]

    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        png_path = os.path.join(d, "figure_1_experimental_framework.png")
        pdf_path = os.path.join(d, "figure_1_experimental_framework.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved: {png_path} and {pdf_path}")

    plt.close(fig)

if __name__ == "__main__":
    generate_clean_figure_1()
