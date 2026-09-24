import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

def generate_figure_1():
    plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(figsize=(12.5, 7.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Title
    ax.text(0.5, 0.95, "Figure 1: Methodological Architecture of the Quantum vs Classical NLP Study",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#000000")

    # Box layout coordinates
    # Top Row: 3 Boxes
    y_top = 0.57
    h_top = 0.30

    # Box 1: Text Corpora
    b1_x, b1_w = 0.04, 0.27
    rect1 = Rectangle((b1_x, y_top), b1_w, h_top, facecolor="#E3F2FD", edgecolor="#000000", linewidth=2.0)
    ax.add_patch(rect1)
    ax.text(b1_x + b1_w/2, y_top + h_top * 0.62, "Text Corpora", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#000000")
    ax.text(b1_x + b1_w/2, y_top + h_top * 0.35, "(SMS, CEAS, MeAJOR)", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1E293B")

    # Box 2: Representation Pipelines
    b2_x, b2_w = 0.365, 0.27
    rect2 = Rectangle((b2_x, y_top), b2_w, h_top, facecolor="#F3E5F5", edgecolor="#000000", linewidth=2.0)
    ax.add_patch(rect2)
    ax.text(b2_x + b2_w/2, y_top + h_top * 0.68, "Representation\nPipelines", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#000000", linespacing=1.15)
    ax.text(b2_x + b2_w/2, y_top + h_top * 0.30, "(TF-IDF N-grams /\nRoBERTa Embeddings)", ha="center", va="center",
            fontsize=12.5, fontweight="semibold", color="#1E293B", linespacing=1.2)

    # Box 3: Information Bottleneck
    b3_x, b3_w = 0.69, 0.27
    rect3 = Rectangle((b3_x, y_top), b3_w, h_top, facecolor="#FFF3E0", edgecolor="#000000", linewidth=2.0)
    ax.add_patch(rect3)
    ax.text(b3_x + b3_w/2, y_top + h_top * 0.68, "Information\nBottleneck", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#000000", linespacing=1.15)
    ax.text(b3_x + b3_w/2, y_top + h_top * 0.30, "(PCA: 2D, 4D, 6D, 8D,\n10D, 12D, 16D)", ha="center", va="center",
            fontsize=12.5, fontweight="semibold", color="#1E293B", linespacing=1.2)

    # Bottom Row: 2 Boxes
    y_bot = 0.10
    h_bot = 0.35

    # Box 4: Model Induction
    b4_x, b4_w = 0.06, 0.40
    rect4 = Rectangle((b4_x, y_bot), b4_w, h_bot, facecolor="#E8F5E9", edgecolor="#000000", linewidth=2.0)
    ax.add_patch(rect4)
    ax.text(b4_x + b4_w/2, y_bot + h_bot * 0.83, "Model Induction", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#000000")
    ax.text(b4_x + 0.03, y_bot + h_bot * 0.60, "•  Linear SVM", ha="left", va="center",
            fontsize=13, fontweight="semibold", color="#0F172A")
    ax.text(b4_x + 0.03, y_bot + h_bot * 0.40, "•  Classical RBF SVC", ha="left", va="center",
            fontsize=13, fontweight="semibold", color="#0F172A")
    ax.text(b4_x + 0.03, y_bot + h_bot * 0.20, "•  Quantum ZZFeatureMap SVC", ha="left", va="center",
            fontsize=13, fontweight="semibold", color="#0F172A")

    # Box 5: Rigorous Multi-Regime Evaluation
    b5_x, b5_w = 0.54, 0.40
    rect5 = Rectangle((b5_x, y_bot), b5_w, h_bot, facecolor="#FFEBEE", edgecolor="#000000", linewidth=2.0)
    ax.add_patch(rect5)
    ax.text(b5_x + b5_w/2, y_bot + h_bot * 0.83, "Rigorous Multi-Regime Evaluation", ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="#000000")
    ax.text(b5_x + 0.025, y_bot + h_bot * 0.60, "•  Frozen IID Scaling Benchmark", ha="left", va="center",
            fontsize=12.5, fontweight="semibold", color="#0F172A")
    ax.text(b5_x + 0.025, y_bot + h_bot * 0.40, "•  Directional Source-Holdout (A & B)", ha="left", va="center",
            fontsize=12.5, fontweight="semibold", color="#0F172A")
    ax.text(b5_x + 0.025, y_bot + h_bot * 0.20, "•  10k Bootstrap CIs & Permutation Tests", ha="left", va="center",
            fontsize=12.5, fontweight="semibold", color="#0F172A")

    # ------------------- ARROWS -------------------
    # Arrow 1: Box 1 -> Box 2
    arrow1 = FancyArrowPatch((b1_x + b1_w, y_top + h_top/2), (b2_x, y_top + h_top/2),
                             arrowstyle="-|>", mutation_scale=18, color="#000000", lw=2.2)
    ax.add_patch(arrow1)

    # Arrow 2: Box 2 -> Box 3
    arrow2 = FancyArrowPatch((b2_x + b2_w, y_top + h_top/2), (b3_x, y_top + h_top/2),
                             arrowstyle="-|>", mutation_scale=18, color="#000000", lw=2.2)
    ax.add_patch(arrow2)

    # Arrow 3: Box 3 -> Box 4
    p3_x = b3_x + b3_w/2      # 0.825
    p4_x = b4_x + b4_w/2      # 0.26
    y_corridor = (y_top + y_bot + h_bot) / 2  # 0.51

    # Orthogonal Line Path from Box 3 bottom down to corridor, then left to Box 4 center
    verts = [
        (p3_x, y_top),
        (p3_x, y_corridor),
        (p4_x, y_corridor)
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO]
    path = Path(verts, codes)
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor='#000000', lw=2.2))

    # Downward arrow from corridor into top of Box 4
    arrow3 = FancyArrowPatch((p4_x, y_corridor), (p4_x, y_bot + h_bot),
                             arrowstyle="-|>", mutation_scale=18, color="#000000", lw=2.2)
    ax.add_patch(arrow3)

    # Arrow 4: Box 4 -> Box 5
    arrow4 = FancyArrowPatch((b4_x + b4_w, y_bot + h_bot/2), (b5_x, y_bot + h_bot/2),
                             arrowstyle="-|>", mutation_scale=18, color="#000000", lw=2.2)
    ax.add_patch(arrow4)

    plt.tight_layout(pad=0.3)

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
    generate_figure_1()
