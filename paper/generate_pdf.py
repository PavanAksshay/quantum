import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "When Do Quantum Kernels Help for Text Security? — Empirical Benchmark")
            self.drawRightString(612 - 54, 750, "Research Paper")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Footer
        self.setStrokeColor(colors.HexColor("#D1D5DB"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Confidential — Prepared for Academic Review & Evaluation")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, page_text)
        self.restoreState()

def build_pdf(output_path="RESEARCH_PAPER.pdf"):
    figures_dir = os.path.join(os.path.dirname(__file__), "submission", "figures")
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        alignment=1, # Center
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=12
    )

    author_style = ParagraphStyle(
        'DocAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        alignment=1,
        spaceAfter=4
    )

    affiliation_style = ParagraphStyle(
        'DocAffil',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=15
    )

    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#0F172A"),
        alignment=1,
        spaceAfter=6
    )

    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=4, # Justified
        leftIndent=15,
        rightIndent=15,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        alignment=4,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
        alignment=0
    )

    ref_style = ParagraphStyle(
        'BibRef',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        firstLineIndent=-15,
        spaceAfter=4
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("When Do Quantum Kernels Help for Text Security?", title_style))
    story.append(Paragraph("A Multi-Dataset Evaluation of Representation, Geometry, Generalization, and Computational Cost", subtitle_style))
    story.append(Paragraph("Research Manuscript & Comprehensive Experimental Report", author_style))
    story.append(Paragraph("Department of Computer Science / Quantum Computing Lab &bull; Ready for Faculty Review", affiliation_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=0, spaceAfter=10))

    # Abstract Box
    story.append(Paragraph("<b>ABSTRACT</b>", abstract_heading))
    abstract_content = (
        "Quantum kernel methods have attracted substantial interest for natural language processing and cybersecurity, "
        "motivated by theoretical conjectures that mapping classical data into exponentially large Hilbert spaces could yield superior non-linear "
        "decision boundaries or enhanced generalization under distribution shift. However, empirical studies often rely on small sample sizes, "
        "unmatched classical baselines, or unvalidated split partitions. In this work, we present a controlled empirical evaluation comparing "
        "parameter-free quantum fidelity kernels with matched classical radial basis function (RBF) kernels across three benchmark corpora drawing "
        "from more than 150,000 audited text records: the SMS Spam Collection, the CEAS 2008 Email Corpus, and the multi-source MeAJOR archive. "
        "Using frozen, leakage-safe protocols and 10 independent random seeds, we evaluate in-distribution scaling (2–12 qubits), cross-source domain holdouts "
        "(TREC 2007 &rarr; TREC 2005/2006), feature space geometry, and computational overhead.<br/><br/>"
        "Under matched in-distribution (IID) conditions, the quantum kernel is competitive with classical RBF, displaying minor statistically detectable "
        "improvements at intermediate dimensions (+0.46 pp at 8D, p = 0.0016; +0.57 pp at 10D, p = 0.0052) that remain strictly within the predefined "
        "practical-equivalence threshold (&epsilon; = 0.01 F1), converging to complete parity at 12D (+0.14 pp, p = 0.2824). Under cross-source domain "
        "transfer, the quantum kernel exhibits a statistically significant and practically meaningful performance deficit (&Delta;F1 = -0.0233, p = 0.0046, "
        "Benjamini-Hochberg FDR p = 0.0069). Representation ablations demonstrate that upstream feature representation dominates kernel selection by an "
        "order of magnitude: switching from TF-IDF to dense RoBERTa embeddings shifts relative performance by 3.90 percentage points. Geometrically, "
        "quantum and RBF Gram matrices show moderate correlation (r &approx; 0.55–0.65), while single-state entropy is strongly inversely associated with "
        "pairwise kernel diversity (r = -0.78 to -0.83). Computationally, classical statevector simulation of the quantum kernel requires 108.8s per run "
        "at 12D compared to 1.7s for RBF (&approx;64&times; penalty). We conclude that parameter-free quantum fidelity kernels do not produce a consistent "
        "practical advantage for text security classification, and we outline rigorous benchmarking standards for future applied QML evaluations."
    )
    story.append(Paragraph(abstract_content, abstract_text))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=0, spaceAfter=12))

    # Section 1: Introduction
    story.append(Paragraph("1. Introduction & Research Problem", h1_style))
    story.append(Paragraph(
        "Text-based social engineering attacks, including email phishing, SMS scams, and fraudulent communications, represent one of the most pervasive "
        "threat vectors in modern digital infrastructure. Automated defense mechanisms rely heavily on natural language processing (NLP) and machine "
        "learning classifiers to filter malicious content. However, security environments are characterized by persistent distribution shift: adversaries "
        "continuously modify lexical patterns to evade detection filters, attack campaigns differ substantially across organizational sources, and seasonal shifts "
        "alter background communications.", body_style
    ))
    story.append(Paragraph(
        "In recent years, quantum machine learning (QML) has emerged as an alternative paradigm for non-linear pattern recognition. In a quantum support "
        "vector classifier (QSVC), classical input vectors <b>x</b> &isin; &real;<sup>d</sup> are mapped into quantum states |&psi;(<b>x</b>)&rang; in a "
        "2<sup>N<sub>q</sub></sup>-dimensional complex Hilbert space via a parameterized unitary circuit. Theoretical investigations have established that "
        "certain quantum feature maps generate inner products that are classically intractable to estimate efficiently, offering conjectured separations "
        "on engineered data distributions. This mathematical framework has motivated the hypothesis that quantum kernels might uncover non-linear structural "
        "regularities in text representations that remain inaccessible to standard classical kernels.", body_style
    ))
    story.append(Paragraph(
        "However, prior applied studies have suffered from four major methodological flaws: (1) small sample sizes (N &lt; 1,000), (2) subtle data leakage "
        "in vectorizer/SVD fitting, (3) unmatched classical controls, and (4) absence of cross-domain out-of-distribution validation. This study resolves "
        "this critical open question through a strictly controlled multi-dataset empirical benchmark.", body_style
    ))

    # Figure 1
    fig1_path = os.path.join(figures_dir, "figure_1_experimental_framework.png")
    if os.path.exists(fig1_path):
        img1 = RLImage(fig1_path, width=6.8*inch, height=2.3*inch)
        story.append(Spacer(1, 4))
        story.append(img1)
        story.append(Paragraph("<b>Figure 1:</b> Leakage-safe end-to-end experimental framework comparing Classical Gaussian RBF and Quantum Fidelity Kernels.", caption_style))

    # Section 2: Methodology & Experimental Design
    story.append(Paragraph("2. Methodology & Experimental Framework", h1_style))
    story.append(Paragraph(
        "<b>Quantum Fidelity Kernel Formulation:</b> We evaluate the canonical two-layer cyclic <i>ZZFeatureMap</i> on N<sub>q</sub> = d qubits. For scaled input "
        "<b>x</b> &isin; [0, &pi;]<sup>d</sup>, the state preparation unitary is &Ucirc;<sub>&Phi;(<b>x</b>)</sub> = (U<sub>&Phi;(<b>x</b>)</sub> H<sup>&otimes;N<sub>q</sub></sup>)<sup>2</sup>, "
        "where H is the Hadamard gate and U<sub>&Phi;(<b>x</b>)</sub> applies diagonal phase rotations and entangling ZZ interactions. The quantum kernel evaluates pure state fidelity: "
        "<b>K<sub>Q</sub>(x, z) = |&lang;&psi;(x)|&psi;(z)&rang;|<sup>2</sup></b>, producing a strictly positive semi-definite (PSD) Gram matrix with unit diagonal entries.", body_style
    ))
    story.append(Paragraph(
        "<b>Matched Classical Gaussian RBF Baseline:</b> The matched classical baseline evaluates K<sub>RBF</sub>(x, z) = exp(-&gamma; ||x - z||<sub>2</sub><sup>2</sup>), "
        "with dynamic scaling &gamma; = 1 / (d &middot; Var(X)) on the identical d-dimensional representation.", body_style
    ))

    # Table 1: Corpora
    story.append(Spacer(1, 4))
    tab1_data = [
        [Paragraph("<b>Corpus</b>", table_header_style), Paragraph("<b>Raw Size</b>", table_header_style), Paragraph("<b>Usable</b>", table_header_style), Paragraph("<b>Positive %</b>", table_header_style), Paragraph("<b>Train / Val / Test</b>", table_header_style), Paragraph("<b>Source Profile</b>", table_header_style)],
        [Paragraph("SMS Spam", table_cell_left), Paragraph("5,574", table_cell_style), Paragraph("5,572", table_cell_style), Paragraph("13.41%", table_cell_style), Paragraph("3,343 / 1,114 / 1,115", table_cell_style), Paragraph("Mobile SMS messages", table_cell_left)],
        [Paragraph("CEAS 2008", table_cell_left), Paragraph("39,154", table_cell_style), Paragraph("15,000", table_cell_style), Paragraph("18.90%", table_cell_style), Paragraph("10,000 / 2,500 / 2,500", table_cell_style), Paragraph("Phishing & Ham emails", table_cell_left)],
        [Paragraph("MeAJOR Archive", table_cell_left), Paragraph("108,685", table_cell_style), Paragraph("108,684", table_cell_style), Paragraph("19.33%", table_cell_style), Paragraph("10,000 / 2,500 / 2,500", table_cell_style), Paragraph("Multi-source (TREC 5/6/7)", table_cell_left)]
    ]
    t1 = Table(tab1_data, colWidths=[1.1*inch, 0.7*inch, 0.8*inch, 0.8*inch, 1.4*inch, 2.0*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table 1:</b> Benchmark dataset characteristics, sample sizes, and split partitions.", caption_style))

    # Section 3: Empirical Results
    story.append(Paragraph("3. Empirical Findings & Analysis", h1_style))
    story.append(Paragraph(
        "<b>3.1 In-Distribution Performance (IID Parity):</b> Under canonical in-distribution evaluation on MeAJOR at 8 dimensions (N=10 seeds), "
        "the quantum fidelity kernel achieves test F1 = 0.8754 &plusmn; 0.0029, while the matched classical RBF kernel achieves F1 = 0.8709 &plusmn; 0.0030. "
        "The paired difference is &Delta;F1 = +0.0046 &plusmn; 0.0027 (+0.46 pp, p = 0.0016). Crucially, the entire 95% bootstrap confidence interval "
        "[+0.0030, +0.0061] lies strictly below the predefined practical-equivalence threshold (&epsilon; = 0.01), demonstrating practical equivalence.", body_style
    ))

    # Table 2: Dimensionality scaling
    story.append(Spacer(1, 4))
    tab2_data = [
        [Paragraph("<b>Dim (d)</b>", table_header_style), Paragraph("<b>Quantum F1 (x̄ ± s)</b>", table_header_style), Paragraph("<b>Classical RBF F1</b>", table_header_style), Paragraph("<b>Paired &Delta;F1</b>", table_header_style), Paragraph("<b>95% Bootstrap CI</b>", table_header_style), Paragraph("<b>Permutation p</b>", table_header_style), Paragraph("<b>Equivalence Status</b>", table_header_style)],
        [Paragraph("2D", table_cell_style), Paragraph("0.6447 ± 0.0038", table_cell_style), Paragraph("0.6735 ± 0.0031", table_cell_style), Paragraph("-0.0288", table_cell_style), Paragraph("[-0.0312, -0.0264]", table_cell_style), Paragraph("< 0.001", table_cell_style), Paragraph("Classical Advantage", table_cell_style)],
        [Paragraph("4D", table_cell_style), Paragraph("0.7876 ± 0.0030", table_cell_style), Paragraph("0.7918 ± 0.0025", table_cell_style), Paragraph("-0.0042", table_cell_style), Paragraph("[-0.0065, -0.0019]", table_cell_style), Paragraph("0.0180", table_cell_style), Paragraph("Equivalent (|&Delta;| < 0.01)", table_cell_style)],
        [Paragraph("6D", table_cell_style), Paragraph("0.8253 ± 0.0026", table_cell_style), Paragraph("0.8254 ± 0.0022", table_cell_style), Paragraph("-0.0001", table_cell_style), Paragraph("[-0.0021, +0.0018]", table_cell_style), Paragraph("0.9410", table_cell_style), Paragraph("Equivalent (|&Delta;| < 0.01)", table_cell_style)],
        [Paragraph("8D", table_cell_style), Paragraph("0.8754 ± 0.0029", table_cell_style), Paragraph("0.8709 ± 0.0030", table_cell_style), Paragraph("+0.0046", table_cell_style), Paragraph("[+0.0030, +0.0061]", table_cell_style), Paragraph("0.0016", table_cell_style), Paragraph("Equivalent (|&Delta;| < 0.01)", table_cell_style)],
        [Paragraph("10D", table_cell_style), Paragraph("0.9023 ± 0.0034", table_cell_style), Paragraph("0.8967 ± 0.0049", table_cell_style), Paragraph("+0.0057", table_cell_style), Paragraph("[+0.0025, +0.0089]", table_cell_style), Paragraph("0.0052", table_cell_style), Paragraph("Equivalent (|&Delta;| < 0.01)", table_cell_style)],
        [Paragraph("12D", table_cell_style), Paragraph("0.9137 ± 0.0046", table_cell_style), Paragraph("0.9123 ± 0.0023", table_cell_style), Paragraph("+0.0014", table_cell_style), Paragraph("[-0.0010, +0.0037]", table_cell_style), Paragraph("0.2824", table_cell_style), Paragraph("Strict Statistical Parity", table_cell_style)]
    ]
    t2 = Table(tab2_data, colWidths=[0.6*inch, 1.2*inch, 1.1*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.1*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table 2:</b> Dimensionality scaling profile across 10 computational seeds on MeAJOR IID partition.", caption_style))

    # Figures 2 & 3 side by side
    fig2_path = os.path.join(figures_dir, "figure_2_iid_f1_vs_dimensionality.png")
    fig3_path = os.path.join(figures_dir, "figure_3_quantum_minus_rbf_vs_dimensionality.png")
    if os.path.exists(fig2_path) and os.path.exists(fig3_path):
        f2_img = RLImage(fig2_path, width=3.3*inch, height=2.2*inch)
        f3_img = RLImage(fig3_path, width=3.3*inch, height=2.2*inch)
        img_table = Table([[f2_img, f3_img]], colWidths=[3.4*inch, 3.4*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table)
        story.append(Paragraph("<b>Figures 2 & 3:</b> (Left) In-distribution test F1 scaling vs dimensionality d &isin; {2..12}. (Right) Paired &Delta;F1 (Quantum minus RBF) with 95% bootstrap CI against the &plusmn;0.01 practical equivalence boundary.", caption_style))

    # Result 4: Cross-Source Domain Transfer
    story.append(Paragraph("3.2 Cross-Source Domain Shift & Out-of-Distribution Degradation", h2_style))
    story.append(Paragraph(
        "Under cross-source domain transfer (Direction B: trained strictly on TREC 2007 and evaluated on unseen TREC 2005/2006 emails across 10 seeds):<br/>"
        "&bull; <b>Classical RBF (8D):</b> Holdout F1 = 0.6913 &plusmn; 0.0161 (20.6% relative drop from IID).<br/>"
        "&bull; <b>Quantum Kernel (8D):</b> Holdout F1 = 0.6680 &plusmn; 0.0094 (23.7% relative drop from IID).<br/>"
        "&bull; <b>Paired Degradation Difference:</b> &Delta;F1 = -0.0233 &plusmn; 0.0200, 95% Bootstrap CI [-0.0353, -0.0117], paired permutation p = 0.0046, "
        "Benjamini-Hochberg FDR p = 0.0069. The quantum kernel degrades significantly more under domain transfer than matched classical RBF.", body_style
    ))

    # Figure 4 & Figure 6 side by side
    fig4_path = os.path.join(figures_dir, "figure_4_iid_vs_source_holdout.png")
    fig6_path = os.path.join(figures_dir, "figure_6_representation_interaction.png")
    if os.path.exists(fig4_path) and os.path.exists(fig6_path):
        f4_img = RLImage(fig4_path, width=3.3*inch, height=2.2*inch)
        f6_img = RLImage(fig6_path, width=3.3*inch, height=2.2*inch)
        img_table2 = Table([[f4_img, f6_img]], colWidths=[3.4*inch, 3.4*inch])
        img_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table2)
        story.append(Paragraph("<b>Figures 4 & 6:</b> (Left) In-distribution vs Domain Holdout degradation. (Right) Upstream text representation interaction (TF-IDF vs RoBERTa) causing ranking reversal.", caption_style))

    # Result 5: Representation Primacy
    story.append(Paragraph("3.3 Representation Primacy & Ranking Inversion", h2_style))
    story.append(Paragraph(
        "Evaluating the interaction between kernel function and text embedding representation on CEAS 2008 reveals that feature representation dominates "
        "kernel choice by an order of magnitude: on 8D TF-IDF + TruncatedSVD, Quantum achieves F1 = 0.9736 vs RBF F1 = 0.9641 (+0.95 pp quantum advantage), "
        "whereas on 8D Dense RoBERTa embeddings, Quantum achieves F1 = 0.9601 vs RBF F1 = 0.9896 (-2.95 pp classical advantage). The net shift is 3.90 pp. "
        "This ranking reversal occurs because classical RBF directly leverages the continuous metric clustering of transformer embeddings, while cyclic phase "
        "encodings disrupt Euclidean semantic neighborhoods.", body_style
    ))

    # Figures 7 & 8: Geometry & Entropy
    fig7_path = os.path.join(figures_dir, "figure_7_geometry_correlation_vs_dimensionality.png")
    fig8_path = os.path.join(figures_dir, "figure_8_entropy_vs_kernel_diversity.png")
    if os.path.exists(fig7_path) and os.path.exists(fig8_path):
        f7_img = RLImage(fig7_path, width=3.3*inch, height=2.1*inch)
        f8_img = RLImage(fig8_path, width=3.3*inch, height=2.1*inch)
        img_table3 = Table([[f7_img, f8_img]], colWidths=[3.4*inch, 3.4*inch])
        img_table3.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(img_table3)
        story.append(Paragraph("<b>Figures 7 & 8:</b> (Left) Gram matrix Pearson correlation between Quantum and RBF. (Right) Inverse relationship between statevector entropy and pairwise Gram diversity.", caption_style))

    # Computational Cost Table & Figure 5
    story.append(Paragraph("3.4 Computational Execution Cost Scaling", h2_style))
    story.append(Paragraph(
        "Exact classical statevector simulation of the quantum kernel exhibits steep superlinear time and memory scaling. "
        "At 12 dimensions, the quantum kernel requires 108.8s per run (6.4 GB RAM) versus 1.7s for classical RBF (64&times; execution overhead). "
        "At 16 dimensions, quantum statevector simulation exceeds 10.5 GB RAM and becomes infeasible on standard workstations.", body_style
    ))

    tab5_data = [
        [Paragraph("<b>Qubits / Dim (d)</b>", table_header_style), Paragraph("<b>Quantum Sim Time (s)</b>", table_header_style), Paragraph("<b>Classical RBF Time (s)</b>", table_header_style), Paragraph("<b>Overhead Ratio</b>", table_header_style), Paragraph("<b>Peak Memory (Quantum)</b>", table_header_style)],
        [Paragraph("2D", table_cell_style), Paragraph("3.4 s", table_cell_style), Paragraph("1.3 s", table_cell_style), Paragraph("2.6&times;", table_cell_style), Paragraph("142 MB", table_cell_style)],
        [Paragraph("4D", table_cell_style), Paragraph("5.8 s", table_cell_style), Paragraph("1.3 s", table_cell_style), Paragraph("4.5&times;", table_cell_style), Paragraph("185 MB", table_cell_style)],
        [Paragraph("8D", table_cell_style), Paragraph("17.2 s", table_cell_style), Paragraph("1.4 s", table_cell_style), Paragraph("12.3&times;", table_cell_style), Paragraph("680 MB", table_cell_style)],
        [Paragraph("12D", table_cell_style), Paragraph("108.8 s", table_cell_style), Paragraph("1.7 s", table_cell_style), Paragraph("64.0&times;", table_cell_style), Paragraph("6.4 GB", table_cell_style)],
        [Paragraph("16D", table_cell_style), Paragraph("OOM (>10.5 GB)", table_cell_style), Paragraph("2.1 s", table_cell_style), Paragraph("&infin;", table_cell_style), Paragraph(">10.5 GB", table_cell_style)]
    ]
    t5 = Table(tab5_data, colWidths=[1.2*inch, 1.4*inch, 1.4*inch, 1.1*inch, 1.7*inch])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t5)
    story.append(Paragraph("<b>Table 3:</b> Computational execution runtime and memory footprint benchmark on 10,000 samples.", caption_style))

    # Section 4: Discussion & Conclusions
    story.append(Paragraph("4. Discussion & Scientific Recommendations", h1_style))
    story.append(Paragraph(
        "<b>Key Takeaways for Applied QML:</b><br/>"
        "1. <b>No Consistent Advantage:</b> Parameter-free quantum fidelity kernels do not provide a practical advantage over matched Gaussian RBF for text security.<br/>"
        "2. <b>Primacy of Feature Engineering:</b> Upstream text representations (TF-IDF vs dense Transformer embeddings) matter far more than kernel geometry.<br/>"
        "3. <b>Domain Robustness:</b> Quantum Hilbert space mappings do not confer natural immunity to adversarial domain shift; out-of-distribution evaluation is essential.<br/>"
        "4. <b>Benchmarking Standard:</b> Future QML research must report matched baseline controls, multi-seed confidence intervals, and computational runtimes.", body_style
    ))

    # References
    story.append(Spacer(1, 6))
    story.append(Paragraph("Selected References", h1_style))
    refs = [
        "[1] V. Havlicek, et al., 'Supervised learning with quantum-enhanced feature spaces,' Nature, vol. 567, no. 7747, pp. 209-213, 2019.",
        "[2] M. Schuld and N. Killoran, 'Quantum machine learning in feature Hilbert spaces,' Phys. Rev. Lett., vol. 122, no. 4, p. 040504, 2019.",
        "[3] H.-Y. Huang, et al., 'Power of data in quantum machine learning,' Nature Communications, vol. 12, no. 1, p. 2631, 2021.",
        "[4] S. Thanasilp, et al., 'Exponential concentration and expressivity bottlenecks in quantum kernel methods,' PRX Quantum, 2024.",
        "[5] J. Kübler, et al., 'The inductive bias of quantum kernels,' Advances in Neural Information Processing Systems (NeurIPS), 2021.",
        "[6] A. Hridi, et al., 'Batch-processed quantum machine learning for email phishing detection,' IEEE Access, 2026.",
        "[7] C. Cortes and V. Vapnik, 'Support-vector networks,' Machine Learning, vol. 20, no. 3, pp. 273-297, 1995."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "RESEARCH_PAPER.pdf")
    build_pdf(out)
