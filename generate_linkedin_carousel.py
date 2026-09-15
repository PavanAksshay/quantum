import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

CANVAS_W = 1200
CANVAS_H = 1200
DPI = 150
FIG_W = CANVAS_W / DPI # 8 inches
FIG_H = CANVAS_H / DPI # 8 inches

# Professional Monochrome / Slate Color Palette (No blue, no rounded pills)
COLOR_BG = '#FFFFFF'
COLOR_TITLE = '#0F172A'       # Deep black / slate
COLOR_SUBTITLE = '#334155'    # Charcoal
COLOR_BODY = '#475569'        # Medium slate
COLOR_MUTED = '#64748B'       # Light slate
COLOR_BORDER = '#E2E8F0'      # Clean subtle divider line
COLOR_ACCENT = '#0F172A'      # High-contrast dark accent

def create_slide_1(output_path):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(COLOR_BG)
    
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.set_facecolor(COLOR_BG)
    ax.axis('off')
    
    # Top Tag (Plain clean text with subtle underline)
    ax.text(0.08, 0.92, "QUANTUM MACHINE LEARNING RESEARCH", 
            color=COLOR_TITLE, fontsize=10, fontweight='bold')
    ax.plot([0.08, 0.92], [0.895, 0.895], color=COLOR_TITLE, lw=1.5)
    
    # Big Headline
    ax.text(0.08, 0.73, "I went looking for a\nQuantum Advantage in\nText Security.", 
            color=COLOR_TITLE, fontsize=24, fontweight='bold', linespacing=1.2)
    
    ax.text(0.08, 0.60, "Here is what the empirical data actually revealed.", 
            color=COLOR_SUBTITLE, fontsize=14.5, fontweight='bold')
    
    # Divider line
    ax.plot([0.08, 0.92], [0.56, 0.56], color=COLOR_BORDER, lw=1.5)
    
    # 4 Structured Pillars with clean straight lines
    pillars = [
        ("01", "CONTROLLED BENCHMARK", "Fidelity kernels vs RBF & Linear across 10 matched seeds & 2D–12D."),
        ("02", "THE 'VANISHING' EFFECT", "Why an initial +1.14 pp MiniLM gain collapsed to -2.23 pp upon replication."),
        ("03", "CIRCUIT DEPTH PARADOX", "Why adding ZZFeatureMap layers degraded F1 from 0.533 down to 0.320."),
        ("04", "GEOMETRIC MECHANISM", "Why classical distance preservation (r = 0.77) governs quantum F1.")
    ]
    
    y_starts = [0.47, 0.36, 0.25, 0.14]
    for i, (num, head, sub) in enumerate(pillars):
        y = y_starts[i]
        # Left border bar
        ax.plot([0.08, 0.08], [y - 0.03, y + 0.03], color=COLOR_TITLE, lw=3)
        # Numbers & text
        ax.text(0.10, y + 0.012, f"{num}  |  {head}", color=COLOR_TITLE, fontsize=10.5, fontweight='bold')
        ax.text(0.10, y - 0.022, sub, color=COLOR_BODY, fontsize=9.5)
        # Bottom light separator (except last)
        if i < 3:
            ax.plot([0.08, 0.92], [y - 0.045, y - 0.045], color=COLOR_BORDER, lw=0.8, ls=':')

    # Footer
    ax.plot([0.08, 0.92], [0.07, 0.07], color=COLOR_BORDER, lw=1.2)
    ax.text(0.08, 0.035, "SWIPE TO EXPLORE THE EVIDENCE  →", color=COLOR_TITLE, fontsize=10, fontweight='bold')
    ax.text(0.92, 0.035, "Slide 1 / 6", color=COLOR_MUTED, fontsize=9.5, fontweight='bold', ha='right')
    
    plt.savefig(output_path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close()

def create_slide_with_image(output_path, slide_num, tag, title, bullets, image_path, takeaway):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(COLOR_BG)
    
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.set_facecolor(COLOR_BG)
    ax.axis('off')
    
    # Top Tag
    ax.text(0.07, 0.935, tag.upper(), color=COLOR_MUTED, fontsize=9.5, fontweight='bold')
    ax.plot([0.07, 0.93], [0.915, 0.915], color=COLOR_BORDER, lw=1.2)
    
    # Title
    ax.text(0.07, 0.865, title, color=COLOR_TITLE, fontsize=17, fontweight='bold')
    
    # Bullet points (clean dark text)
    y = 0.81
    for b in bullets:
        ax.text(0.07, y, f"—  {b}", color=COLOR_SUBTITLE, fontsize=9.8, wrap=True)
        y -= 0.036
    
    # Embed image
    if os.path.exists(image_path):
        img_ax = fig.add_axes([0.07, 0.165, 0.86, 0.51])
        img = mpimg.imread(image_path)
        img_ax.imshow(img)
        img_ax.axis('off')
        
        # Crisp square border
        for spine in img_ax.spines.values():
            spine.set_edgecolor(COLOR_BORDER)
            spine.set_linewidth(1.2)
            spine.set_visible(True)

    # Takeaway Section (Clean left-accent bar instead of yellow/rounded box)
    ax.plot([0.07, 0.07], [0.075, 0.14], color=COLOR_TITLE, lw=3.5)
    ax.text(0.085, 0.118, "KEY TAKEAWAY", color=COLOR_TITLE, fontsize=9, fontweight='bold')
    ax.text(0.085, 0.088, takeaway, color=COLOR_BODY, fontsize=9)
    
    # Footer
    ax.plot([0.07, 0.93], [0.055, 0.055], color=COLOR_BORDER, lw=1.0)
    ax.text(0.07, 0.025, "Pavan Aksshay • Quantum Text Security Research", color=COLOR_MUTED, fontsize=9)
    ax.text(0.93, 0.025, f"Slide {slide_num} / 6", color=COLOR_MUTED, fontsize=9.5, fontweight='bold', ha='right')
    
    plt.savefig(output_path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close()

def main():
    os.makedirs("carousel_slides", exist_ok=True)
    
    print("1. Generating Slide 1 (Cover)...")
    create_slide_1("carousel_slides/slide_1.png")
    
    print("2. Generating Slide 2 (Architecture)...")
    create_slide_with_image(
        "carousel_slides/slide_2.png",
        slide_num=2,
        tag="01 • System Architecture",
        title="Controlled Evaluation Pipeline",
        bullets=[
            "Compared TF-IDF against Sentence Transformers (RoBERTa, MiniLM, MPNet).",
            "Evaluated Quantum Fidelity Kernels (cyclic ZZFeatureMap) against RBF & Linear SVM.",
            "Benchmarked across 10 random seeds, 2D–12D dimensions, and domain shift."
        ],
        image_path="paper/submission/figures/figure_1_experimental_framework.png",
        takeaway="Evaluating across matched seeds and dimensions prevents cherry-picked isolated metrics."
    )
    
    print("3. Generating Slide 3 (MiniLM Seed Effect)...")
    create_slide_with_image(
        "carousel_slides/slide_3.png",
        slide_num=3,
        tag="02 • Statistical Reproducibility",
        title="The 'Vanishing' Advantage: 3 Seeds vs 10 Seeds",
        bullets=[
            "Initial 3-seed run on MiniLM showed an apparent +1.14 pp F1 quantum advantage.",
            "Under rigorous 10-seed paired replication, confirmed difference was -2.23 pp (p > 0.05).",
            "Initial perceived gain was an artifact of small-sample seed variance."
        ],
        image_path="results/exp42/figures/fig6_minilm_seed_dist.png",
        takeaway="Finding a positive result is not the same as finding a statistically reliable result."
    )
    
    print("4. Generating Slide 4 (Circuit Depth Paradox)...")
    create_slide_with_image(
        "carousel_slides/slide_4.png",
        slide_num=4,
        tag="03 • Circuit Expressivity",
        title="Circuit Depth Paradox: Deeper Maps Degrade F1",
        bullets=[
            "Ablated cyclic ZZFeatureMap depth: 1 Layer (0.533 F1) → 2 Layers (0.375) → 3 Layers (0.320).",
            "Deeper quantum circuits did not enhance representation power on sentence embeddings.",
            "Instead, unparameterized entanglement layers caused severe orthogonal state collapse."
        ],
        image_path="results/exp43/figures/fig3_depth_f1.png",
        takeaway="Deeper quantum circuits without trainable parameters can distort metric spaces."
    )
    
    print("5. Generating Slide 5 (Distance Preservation)...")
    create_slide_with_image(
        "carousel_slides/slide_5.png",
        slide_num=5,
        tag="04 • Geometric Mechanism",
        title="Distance Preservation (r = 0.77) Explains Performance",
        bullets=[
            "Quantum F1 correlates strongly with classical distance preservation (r = 0.77).",
            "Kernel diversity alone shows virtually zero correlation with F1 (r = -0.05).",
            "Core question: When is text embedding geometry compatible with a quantum feature map?"
        ],
        image_path="results/exp43/figures/fig6_geom_correlation.png",
        takeaway="Quantum kernels succeed when they preserve meaningful metric structure, not raw diversity."
    )
    
    print("6. Generating Slide 6 (Conclusion & Portal)...")
    create_slide_with_image(
        "carousel_slides/slide_6.png",
        slide_num=6,
        tag="05 • Final Takeaways",
        title="Key Findings & Interactive Dashboard",
        bullets=[
            "Quantum Fidelity Kernels matched classical RBF under IID, but degraded under source shift.",
            "12D simulation incurred a ~64× computational penalty without a net F1 gain.",
            "Explore the live open-source research dashboard: quantum-three-hazel.vercel.app"
        ],
        image_path="paper/submission/figures/figure_2_iid_f1_vs_dimensionality.png",
        takeaway="A rigorous negative result provides more actionable scientific insight than a false positive."
    )
    
    slide_files = [f"carousel_slides/slide_{i}.png" for i in range(1, 7)]
    images = [Image.open(f).convert('RGB') for f in slide_files]
    
    pdf_path = "Quantum_Text_Security_LinkedIn_Carousel.pdf"
    images[0].save(
        pdf_path, 
        save_all=True, 
        append_images=images[1:], 
        resolution=150.0
    )
    print(f"\n✅ Successfully generated clean monochrome/minimalist LinkedIn Carousel PDF: {pdf_path}")

if __name__ == "__main__":
    main()
