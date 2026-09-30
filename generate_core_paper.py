#!/usr/bin/env python3
"""
Generate the IEEE 2-Column Research Paper containing content strictly from Abstract to Conclusion
with the updated humanized text.
Excludes:
- Paper Title & Author Header Block
- Limitations and Scope Section
- References Section
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

PROJECT_ROOT = "/Users/pavanaksshay/quantum"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=30, bottom=30, left=35, right=35):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8.5)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.0)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_body_p(doc, text, bold_prefix=None, space_after=3.2, first_line_indent=0.14):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.04
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_line_indent > 0:
        p.paragraph_format.first_line_indent = Inches(first_line_indent)
    else:
        p.paragraph_format.first_line_indent = Inches(0)
        
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(9.0)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(0, 0, 0)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(20, 20, 20)
    return p

def add_bullet_p(doc, text, bold_prefix=None, space_after=1.6):
    """Add a bullet / list paragraph where each item is its own paragraph to prevent artificial space stretching."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.04
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.first_line_indent = Inches(-0.12)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(9.0)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(0, 0, 0)
        
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(20, 20, 20)
    return p

def add_display_equation(doc, eq_text, eq_num):
    """Add a beautifully formatted mathematical display equation using IEEE standard tab stops."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(3.0)
    p.paragraph_format.space_after = Pt(3.0)
    p.paragraph_format.line_spacing = 1.05
    
    # Center tab at 1.65 in, Right tab at 3.40 in
    p.paragraph_format.tab_stops.add_tab_stop(Inches(1.65), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.40), WD_TAB_ALIGNMENT.RIGHT)
    
    p.add_run("\t")
    r_eq = p.add_run(eq_text)
    r_eq.font.name = "Times New Roman"
    r_eq.font.size = Pt(8.5)
    r_eq.font.italic = True
    r_eq.font.color.rgb = RGBColor(0, 0, 0)
    
    p.add_run("\t")
    r_num = p.add_run(f"({eq_num})")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(8.5)
    r_num.font.color.rgb = RGBColor(0, 0, 0)
    return p

def add_table_ieee(doc, table_num, caption, headers, data, col_widths, font_size=7.2):
    """Add a structured IEEE-style table constrained within a single column (~3.45 in)."""
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(5)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    r_t = p_cap.add_run(f"TABLE {table_num}\n")
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(7.8)
    r_t.font.bold = True
    
    r_c = p_cap.add_run(caption.upper())
    r_c.font.name = "Times New Roman"
    r_c.font.size = Pt(7.0)
    r_c.font.bold = False
    
    tbl = doc.add_table(rows=len(data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    total_dxa = sum(int(w * 1440) for w in col_widths)
    tblPr = tbl._element.xpath('w:tblPr')
    if tblPr:
        tblW = parse_xml(f'<w:tblW {nsdecls("w")} w:w="{total_dxa}" w:type="dxa"/>')
        tblPr[0].append(tblW)
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:insideH w:val="none"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)
        
    hdr_cells = tbl.rows[0].cells
    for col_idx, text in enumerate(headers):
        hdr_cells[col_idx].width = Inches(col_widths[col_idx])
        hdr_cells[col_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_background(hdr_cells[col_idx], "F1F5F9")
        set_cell_margins(hdr_cells[col_idx], top=22, bottom=22, left=20, right=20)
        
        tcPr = hdr_cells[col_idx]._element.get_or_add_tcPr()
        b_bottom = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:bottom w:val="single" w:sz="5" w:space="0" w:color="000000"/></w:tcBorders>')
        tcPr.append(b_bottom)
        
        p = hdr_cells[col_idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 0, 0)
        
    for row_idx, row_data in enumerate(data):
        row_cells = tbl.rows[row_idx + 1].cells
        bg_hex = "FFFFFF" if row_idx % 2 == 0 else "F8FAFC"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].width = Inches(col_widths[col_idx])
            row_cells[col_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(row_cells[col_idx], bg_hex)
            set_cell_margins(row_cells[col_idx], top=18, bottom=18, left=20, right=20)
            
            p = row_cells[col_idx].paragraphs[0]
            p.text = str(cell_value)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(font_size)
                run.font.color.rgb = RGBColor(30, 30, 30)
                
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(2.5)
    return tbl

def add_figure_ieee(doc, fig_path, fig_num, caption, width_in=3.35):
    """Add a centered figure with bottom caption matching IEEE style."""
    if os.path.exists(fig_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(1.5)
        p_img.paragraph_format.first_line_indent = Inches(0)
        p_img.add_run().add_picture(fig_path, width=Inches(width_in))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_cap.paragraph_format.space_after = Pt(3.5)
        p_cap.paragraph_format.first_line_indent = Inches(0)
        p_cap.paragraph_format.line_spacing = 1.0
        p_cap.paragraph_format.keep_with_next = True
        
        r_pre = p_cap.add_run(f"Fig. {fig_num}.  ")
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(7.5)
        r_pre.font.bold = True
        
        r_txt = p_cap.add_run(caption)
        r_txt.font.name = "Times New Roman"
        r_txt.font.size = Pt(7.5)
        r_txt.font.color.rgb = RGBColor(40, 40, 40)

def build_paper():
    print("Building IEEE Research Paper (Strictly Abstract to Conclusion, Custom Humanized Text)...")
    doc = Document()
    
    # 2-Column Section throughout the document (No Title / Author header section)
    sec1 = doc.sections[0]
    sec1.top_margin = Inches(0.75)
    sec1.bottom_margin = Inches(0.75)
    sec1.left_margin = Inches(0.65)
    sec1.right_margin = Inches(0.65)
    
    sectPr = sec1._sectPr
    cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="360"/>')
    sectPr.append(cols)
        
    # -------------------------------------------------------------
    # ABSTRACT & KEYWORDS
    # -------------------------------------------------------------
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.paragraph_format.line_spacing = 1.02
    p_abs.paragraph_format.first_line_indent = Inches(0)
    
    r_absh = p_abs.add_run("Abstract—")
    r_absh.font.name = "Times New Roman"
    r_absh.font.size = Pt(8.5)
    r_absh.font.bold = True
    r_absh.font.italic = True
    
    r_abst = p_abs.add_run(
        "Quantum machine learning is often pitched as a way to pick up subtle patterns by mapping data into enormous mathematical spaces. Whether that promise holds up on real language and cybersecurity problems is still an open question. "
        "We compare a parameter-free quantum fidelity kernel against carefully matched Gaussian RBF kernels on spam, phishing, and malicious-text detection. The benchmark draws on 153,410 records from three datasets: SMS Spam, CEAS 2008, and MeAJOR. Both kinds of model get the same data representations, dimensions, splits, and evaluation procedure, so any gap between them can't be blamed on setup. "
        "When training and test data come from the same distribution, the quantum kernel lands close to the RBF baseline. At eight dimensions it scores an F1 of 0.8754 against 0.8709 for the matched RBF, and an equivalence test says the gap is too small to matter. By twelve dimensions the two are nearly impossible to tell apart. "
        "Things look different once the data source changes. Trained on TREC 2007 and tested on TREC 2005/2006, the quantum model reaches an F1 of 0.6680 and the RBF model reaches 0.6913, so the quantum kernel gives up more when it meets an unfamiliar source. "
        "Two other findings stand out. How the text is represented matters far more than which kernel sits on top; swapping representations moved results by as much as 52.88 percentage points. And simulating the quantum kernel exactly on a classical machine is expensive: 64 times slower than RBF at twelve dimensions, and out of memory at sixteen. Taken together, the results give no support for a general practical advantage from parameter-free quantum kernels in text-security classification. They do make a case for strong matched baselines, strict leakage controls, representation analysis, domain-shift testing, and honest reporting of computational cost."
    )
    r_abst.font.name = "Times New Roman"
    r_abst.font.size = Pt(8.5)
    r_abst.font.italic = True
    
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_before = Pt(0)
    p_kw.paragraph_format.space_after = Pt(5)
    p_kw.paragraph_format.line_spacing = 1.02
    p_kw.paragraph_format.first_line_indent = Inches(0)
    
    r_kwh = p_kw.add_run("Keywords—")
    r_kwh.font.name = "Times New Roman"
    r_kwh.font.size = Pt(8.5)
    r_kwh.font.bold = True
    r_kwh.font.italic = True
    
    r_kwt = p_kw.add_run("Quantum machine learning; quantum kernels; text classification; phishing detection; SMS spam; Hilbert-space geometry; domain adaptation; practical equivalence testing.")
    r_kwt.font.name = "Times New Roman"
    r_kwt.font.size = Pt(8.5)
    r_kwt.font.italic = True

    # -------------------------------------------------------------
    # 1. INTRODUCTION
    # -------------------------------------------------------------
    add_heading_1(doc, "1.  INTRODUCTION")
    
    add_heading_2(doc, "1.1 Text security and domain shift")
    add_body_p(
        doc,
        "Phishing emails, fraudulent text messages, and other deceptive online messages are still one of the easiest ways into an organization and a steady source of financial fraud. Automated text classifiers have become a standard part of the defense."
    )
    add_body_p(
        doc,
        "Deployment is where these classifiers struggle, because language keeps moving. Attackers rewrite their wording to slip past keyword filters, different organizations write in different styles, and everyday language drifts over time. A model that looks nearly perfect on familiar test data can fall apart when it sees messages from a new source or a later period."
    )
    
    add_heading_2(doc, "1.2 Why test quantum machine learning here?")
    add_body_p(
        doc,
        "Interest in quantum machine learning has grown partly because of its potential for non-linear pattern recognition. In a quantum support vector classifier, a classical vector x is encoded as a quantum state |ψ(x)⟩ in a very large Hilbert space. Pairs of states are compared through their fidelity, and the results fill a kernel matrix."
    )
    add_body_p(
        doc,
        "Theoretical separations between quantum and classical methods do exist, but they were shown on carefully built mathematical distributions. Applied QML benchmarks, by contrast, tend to share a handful of weaknesses: very small datasets, preprocessing fitted on test data, weak linear baselines instead of tuned non-linear ones, and no check on whether the model survives a change in distribution."
    )
    add_body_p(
        doc,
        "Our question is narrow. If representation, dimensionality, splits, preprocessing, and optimization are matched as closely as we can manage, does a parameter-free quantum fidelity kernel give a meaningful practical advantage over a Gaussian RBF kernel for text-security classification?"
    )

    # Figure 1: Flowchart / Experimental Framework
    fig1_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_1_experimental_framework.png")
    add_figure_ieee(doc, fig1_path, 1, "End-to-end controlled experimental architecture and evaluation protocol (Multi-corpus ingestion -> Leakage-safe feature extraction -> Dimensionality projection -> Tripartite classifier induction -> Multi-regime inferential evaluation & diagnostics).")

    add_heading_2(doc, "1.3 Hypotheses")
    add_bullet_p(doc, "On in-distribution data, the quantum fidelity kernel and a matched Gaussian RBF kernel are practically equivalent, within ±0.01 F1.", bold_prefix="H1 (in-distribution equivalence). ")
    add_bullet_p(doc, "Weak results at low dimensions come mostly from information lost in linear compression, not from any failure of quantum Hilbert-space geometry.", bold_prefix="H2 (dimensionality bottleneck). ")
    add_bullet_p(doc, "The upstream text representation changes results by roughly an order of magnitude more than the choice of kernel does.", bold_prefix="H3 (representation primacy). ")
    add_bullet_p(doc, "Quantum kernels have no built-in robustness to cross-source domain transfer.", bold_prefix="H4 (domain-shift fragility). ")
    add_bullet_p(doc, "High basis-state dispersion, measured by von Neumann entropy, does not automatically improve pairwise kernel discrimination.", bold_prefix="H5 (state-dispersion decoupling). ")
    add_bullet_p(doc, "Exact statevector simulation scales as O(N² · 2^(N_q)) and runs past typical workstation memory beyond twelve qubits.", bold_prefix="H6 (simulation scalability). ")

    # -------------------------------------------------------------
    # 2. RELATED WORK
    # -------------------------------------------------------------
    add_heading_1(doc, "2.  RELATED WORK")
    add_body_p(
        doc,
        "Our work touches four areas: quantum kernel theory and expressivity bounds, quantum natural language processing, applied QML for security tasks, and the classical baselines that text-security researchers already rely on."
    )
    add_body_p(
        doc,
        "Kernel theory asks how quantum circuits embed data into Hilbert spaces and how expressive those embeddings can be. Most of that analysis has been run on synthetic or purpose-built distributions, not on operational text-security data."
    )
    add_body_p(
        doc,
        "Quantum NLP splits into two camps. The structural camp encodes syntax with categorical grammar and circuit diagrams, usually on small parsed datasets. The statistical camp takes conventional representations such as TF-IDF, Word2Vec, or transformer embeddings and feeds them to a quantum classifier."
    )
    add_body_p(
        doc,
        "Applied QML work has looked at phishing, spam, and intrusion detection, but the early studies often relied on small samples, a single train-test split, weak or mismatched classical baselines, and little or no domain-shift testing. We try to close those gaps with strict pipeline parity across several corpora totaling 153,410 usable records."
    )

    # Table 1: Literature Taxonomy
    t1_headers = ["Research area", "Core idea", "Typical representation"]
    t1_data = [
        ["Quantum kernel theory", "Prepare quantum states in Hilbert space", "Synthetic or group-structured distributions"],
        ["Expressivity and bounds", "Study fidelity concentration and theoretical limits", "Global unparameterized states"],
        ["Structural QNLP", "Use DisCoCat and ZX-calculus for sentence structure", "Syntactic parse trees, often N < 500"],
        ["Statistical QNLP", "Feed conventional text embeddings into QSVC", "TF-IDF, Word2Vec, transformer embeddings"],
        ["Applied QML security", "Use quantum classifiers for threat detection", "Tabular features or small data splits"],
        ["Multi-dataset benchmarking", "Control representation and baseline parity", "Sparse TF-IDF plus dense embeddings"]
    ]
    add_table_ieee(doc, "I", "Literature Positioning and Methodological Taxonomy", t1_headers, t1_data, col_widths=[1.00, 1.25, 1.20], font_size=6.2)

    # -------------------------------------------------------------
    # 3. PROBLEM DEFINITION
    # -------------------------------------------------------------
    add_heading_1(doc, "3.  PROBLEM DEFINITION")
    add_body_p(
        doc,
        "We treat text security as binary classification. Each message m is either legitimate (ham) or malicious (spam or phishing), and a feature extractor turns it into a d-dimensional vector x = f(m)."
    )
    add_body_p(
        doc,
        "A supervised model learns from examples drawn from P_train(X, Y) and is then evaluated on P_test(X, Y). In security work these two distributions frequently differ, because language changes over time and because each organization or message source has its own style."
    )
    add_body_p(
        doc,
        "A kernel classifier implicitly maps x into a Hilbert space through a non-linear feature map φ and measures similarity as inner product:"
    )
    
    # Equation 1
    add_display_equation(doc, "K(x, z) = ⟨φ(x), φ(z)⟩_ℋ", "1")
    
    add_body_p(
        doc,
        "What we want to know is whether the quantum map φ_Q(x) = |ψ(x)⟩ gives a more useful representation than the classical RBF map φ_RBF(x), both in-distribution and after a change of source."
    )

    # -------------------------------------------------------------
    # 4. METHODOLOGY
    # -------------------------------------------------------------
    add_heading_1(doc, "4.  METHODOLOGY")
    
    add_heading_2(doc, "4.1 Quantum fidelity kernel")
    add_body_p(
        doc,
        "The quantum model uses the standard two-layer cyclic ZZFeatureMap with one qubit per feature, so N_q = d. Inputs are scaled to [0, π]^d before entering the circuit, which keeps the phase values in a sensible range."
    )
    add_body_p(
        doc,
        "The circuit alternates Walsh-Hadamard layers with diagonal phase operations. Cyclic nearest-neighbor interactions entangle each qubit with the next one, and the last qubit links back to the first. Starting from the all-zero state, the circuit prepares a data-dependent state |ψ(x)⟩:"
    )
    
    # Equation 2 & 3
    add_display_equation(doc, "U_Φ(x) = exp( i ∑_(j=1)^d x_j Z_j + i ∑_(j=1)^d (π - x_j)(π - x_j') Z_j Z_j' )", "2")
    
    add_body_p(
        doc,
        "The kernel is the squared overlap of two prepared states, evaluating pure-state fidelity:"
    )
    
    # Equation 4
    add_display_equation(doc, "K_Q(x, z) = |⟨ψ(x)|ψ(z)⟩|²", "3")
    
    add_body_p(
        doc,
        "The resulting Gram matrix is positive semidefinite and has 1.0 on the diagonal."
    )

    add_heading_2(doc, "4.2 Matched classical RBF kernel")
    add_body_p(
        doc,
        "The main classical comparator is a Gaussian RBF kernel applied to exactly the same d-dimensional vectors:"
    )
    
    # Equation 5
    add_display_equation(doc, "K_RBF(x, z) = exp( -γ ‖x - z‖_2² ),   γ = 1 / ( d · Var(X) )", "4")
    
    add_body_p(
        doc,
        "We also include a tuned RBF baseline. It searches C over 0.1, 1, 10, and 100 and γ over 0.001, 0.01, 0.1, 1.0, and scale, using five-fold cross-validation on the training data only."
    )

    add_heading_2(doc, "4.3 Geometric diagnostics")
    add_body_p(
        doc,
        "Accuracy on its own doesn't explain why two kernels behave differently, so we also look at three structural properties of their Gram matrices."
    )
    add_body_p(
        doc,
        "Centered kernel-target alignment (CKA) measures how closely a kernel matrix matches the ideal similarity structure implied by the labels. Spectral effective rank estimates how many meaningful eigen-directions the matrix actually uses. Single-state basis-dispersion entropy measures how widely one quantum state's amplitude is spread over the computational-basis states:"
    )
    
    # Equations 6, 7, 8
    add_display_equation(doc, "CKA(K, Y) = ⟨H K H, H Y H⟩_F / ( ‖H K H‖_F · ‖H Y H‖_F )", "5")
    add_display_equation(doc, "R_eff(K) = exp( - ∑_(i=1)^N λ̃_i ln λ̃_i ),   λ̃_i = λ_i / ∑_j λ_j", "6")
    add_display_equation(doc, "S(|ψ(x)⟩) = - ∑_(k=1)^(2^d) |c_k(x)|² ln |c_k(x)|²", "7")

    add_heading_2(doc, "4.4 Data hygiene controls")
    add_body_p(
        doc,
        "Every experiment follows eight safeguards meant to prevent leakage and keep the comparison fair."
    )

    # Table 2: Data Hygiene
    t2_headers = ["Audit item", "Protocol", "Status"]
    t2_data = [
        ["Sample parity", "Every model gets the same train, validation, and test arrays for a given random seed.", "Pass"],
        ["Label parity", "Ground-truth class labels are identical across models.", "Pass"],
        ["Preprocessing isolation", "Tokenizers, TF-IDF vectorizers, and scalers are fit on training data only.", "Pass"],
        ["Projection parity", "TruncatedSVD is fit on training data only, and RBF and QSVC receive the same projected vectors.", "Pass"],
        ["Classical optimization", "RBF gamma is set exactly as 1 / (d · Var(X_train)).", "Pass"],
        ["Parameter matching", "The primary comparison gives neither method a tuning advantage.", "Pass"],
        ["State normalization", "Features are scaled to [0, π]^d before quantum encoding.", "Pass"],
        ["Seed standardization", "The same ten seeds, 42 to 51, govern resampling and other stochastic steps.", "Pass"]
    ]
    add_table_ieee(doc, "II", "Eight-Point Data Hygiene and Causality Verification Protocol", t2_headers, t2_data, col_widths=[0.85, 2.10, 0.50], font_size=6.2)

    add_heading_2(doc, "4.5 Datasets and representations")
    add_body_p(
        doc,
        "The benchmark uses three cybersecurity corpora with 153,410 raw records in total. Text is represented in two ways: a 50,000-term TF-IDF space with unigrams and bigrams, and 768-dimensional dense embeddings from RoBERTa-base, all-mpnet-base-v2, and all-MiniLM-L6-v2."
    )

    # Table 3: Datasets
    t3_headers = ["Corpus", "Usable records", "Ham / spam split", "Vocabulary", "Task"]
    t3_data = [
        ["SMS Spam", "5,572", "4,825 / 747 (13.4% spam)", "8,713", "Mobile fraud detection"],
        ["CEAS 2008", "39,154", "19,058 / 20,096 (51.3% spam)", "48,291", "Email phishing and spam"],
        ["MeAJOR", "108,684", "49,948 / 58,736 (54.0% spam)", "142,805", "Multi-source domain-shift benchmark"]
    ]
    add_table_ieee(doc, "III", "Benchmark Corpus Characteristics and Partitions", t3_headers, t3_data, col_widths=[0.65, 0.60, 0.85, 0.50, 0.85], font_size=6.2)

    # -------------------------------------------------------------
    # 5. RESULTS AND DISCUSSION
    # -------------------------------------------------------------
    add_heading_1(doc, "5.  RESULTS AND DISCUSSION")

    add_heading_2(doc, "5.1 Classical baseline audit")
    add_body_p(
        doc,
        "Before bringing in any quantum kernel, we benchmarked five classical models: Linear SVM, Logistic Regression, the matched Gaussian RBF SVM, Random Forest, and an MLP. Each was run on the full 50,000-feature TF-IDF representation and on an 8-dimensional TruncatedSVD version."
    )
    add_body_p(
        doc,
        "On the full representation the Linear SVM is close to perfect, with F1 of 0.9832 on SMS Spam, 0.9926 on CEAS 2008, and 0.9754 on MeAJOR. Squeeze the inputs down to eight dimensions and every classical architecture loses roughly 10 to 13 F1 points. So at low dimensions the ceiling is set by what compression throws away, not by any particular classifier falling short."
    )

    # Table 4: Classical Baseline Audit
    t4_headers = ["Corpus", "Model", "Full-text F1", "8D SVD F1", "Training time"]
    t4_data = [
        ["SMS Spam", "Linear SVM", "0.9832 ± 0.003", "0.8402 ± 0.005", "0.04 s"],
        ["SMS Spam", "Logistic Regression", "0.9612 ± 0.004", "0.8350 ± 0.006", "0.03 s"],
        ["SMS Spam", "Matched RBF", "0.9785 ± 0.003", "0.8276 ± 0.005", "0.82 s"],
        ["SMS Spam", "Random Forest", "0.9710 ± 0.004", "0.8524 ± 0.004", "0.45 s"],
        ["SMS Spam", "MLP", "0.9801 ± 0.003", "0.8465 ± 0.005", "1.20 s"],
        ["CEAS 2008", "Linear SVM", "0.9926 ± 0.001", "0.9602 ± 0.002", "0.32 s"],
        ["CEAS 2008", "Logistic Regression", "0.9875 ± 0.001", "0.9548 ± 0.002", "0.28 s"],
        ["CEAS 2008", "Matched RBF", "0.9910 ± 0.001", "0.9641 ± 0.002", "12.45 s"],
        ["CEAS 2008", "Random Forest", "0.9880 ± 0.002", "0.9620 ± 0.003", "1.85 s"],
        ["CEAS 2008", "MLP", "0.9918 ± 0.001", "0.9635 ± 0.002", "18.60 s"],
        ["MeAJOR", "Linear SVM", "0.9754 ± 0.001", "0.8445 ± 0.002", "1.45 s"],
        ["MeAJOR", "Logistic Regression", "0.9680 ± 0.002", "0.8398 ± 0.003", "1.10 s"],
        ["MeAJOR", "Matched RBF", "0.9700 ± 0.000", "0.8709 ± 0.003", "71.84 s"],
        ["MeAJOR", "Random Forest", "0.9516 ± 0.002", "0.9014 ± 0.006", "2.16 s"],
        ["MeAJOR", "MLP", "0.9727 ± 0.003", "0.8795 ± 0.002", "42.12 s"]
    ]
    add_table_ieee(doc, "IV", "Classical Baseline Audit Across Five Architectures (Mean ± SD across 10 Seeds)", t4_headers, t4_data, col_widths=[0.65, 0.95, 0.65, 0.65, 0.55], font_size=5.8)

    # Figure 2: Classical Baseline F1
    fig2_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "fig1_classical_f1_across_datasets.png")
    add_figure_ieee(doc, fig2_path, 2, "Empirical performance of five classical machine learning algorithms across datasets (Full 50k TF-IDF vs Matched 8D SVD). Linear SVM dominates full text, while linear compression imposes an identical 10–13 percentage point penalty across all models.")

    add_heading_2(doc, "5.2 In-distribution comparison")
    add_body_p(
        doc,
        "On MeAJOR with matched 8-dimensional SVD features and ten shared seeds, the quantum fidelity kernel reaches an F1 of 0.8754 ± 0.0029. The matched RBF reaches 0.8709 ± 0.0030 and the tuned RBF 0.8742 ± 0.0028. Both non-linear methods beat the 8D Linear SVM, which scores 0.8445 ± 0.0022."
    )
    add_body_p(
        doc,
        "On average the quantum kernel is 0.46 percentage points ahead of the matched RBF. The paired p-value is 0.0016, so the difference is statistically detectable, but a TOST equivalence test with a ±0.01 F1 margin shows it is small enough to count as practically equivalent. Against the tuned RBF the gap shrinks to 0.12 points and is not statistically significant."
    )

    # Table 5: In-Distribution Comparison
    t5_headers = ["Model", "Test F1", "PR-AUC", "ROC-AUC", "Interpretation"]
    t5_data = [
        ["Linear SVM (8D)", "0.8445 ± 0.0022", "0.9315", "0.9404", "Linear baseline"],
        ["Matched RBF (8D)", "0.8709 ± 0.0030", "0.9423", "0.9535", "Matched comparator"],
        ["Tuned RBF (8D)", "0.8742 ± 0.0028", "0.9450", "0.9560", "Tuned comparator"],
        ["Quantum kernel (8D)", "0.8754 ± 0.0029", "0.9372", "0.9515", "Practically equivalent"]
    ]
    add_table_ieee(doc, "V", "Canonical In-Distribution Model Comparison on MeAJOR (Matched 8D SVD, N=10 Seeds)", t5_headers, t5_data, col_widths=[1.05, 0.75, 0.40, 0.40, 0.85], font_size=6.2)

    add_heading_2(doc, "5.3 Dimensionality scaling")
    add_body_p(
        doc,
        "We evaluated 2, 4, 6, 8, 10, and 12 dimensions. Quantum F1 climbs steadily from 0.6447 at 2D to 0.9137 at 12D, a relative gain of 41.7%. The classical RBF follows almost the same path, going from 0.6735 to 0.9123."
    )
    add_body_p(
        doc,
        "At 2D the classical kernel is clearly better. From 4D through 10D the differences stay within our predefined equivalence margin. At 12D the quantum kernel and the matched RBF are separated by just +0.14 percentage points, the bootstrap interval crosses zero, and the result meets our strict parity criteria."
    )

    # Table 6: Dimensionality Scaling
    t6_headers = ["Dimensions", "Quantum F1", "Matched RBF", "Tuned RBF", "Quantum − RBF", "Status"]
    t6_data = [
        ["2D", "0.6447 ± 0.0038", "0.6735 ± 0.0031", "0.6780 ± 0.0029", "−2.88 pp", "Classical superior"],
        ["4D", "0.7876 ± 0.0030", "0.7918 ± 0.0025", "0.7954 ± 0.0024", "−0.42 pp", "Equivalent"],
        ["6D", "0.8253 ± 0.0026", "0.8254 ± 0.0022", "0.8291 ± 0.0021", "−0.01 pp", "Equivalent"],
        ["8D", "0.8754 ± 0.0029", "0.8709 ± 0.0030", "0.8742 ± 0.0028", "+0.46 pp", "Equivalent"],
        ["10D", "0.9023 ± 0.0034", "0.8967 ± 0.0049", "0.9015 ± 0.0038", "+0.57 pp", "Equivalent"],
        ["12D", "0.9137 ± 0.0046", "0.9123 ± 0.0023", "0.9148 ± 0.0020", "+0.14 pp", "Strict parity"]
    ]
    add_table_ieee(doc, "VI", "Dimensionality Scaling Profile on MeAJOR IID (N=10 Seeds)", t6_headers, t6_data, col_widths=[0.55, 0.65, 0.65, 0.65, 0.50, 0.45], font_size=5.8)

    fig3_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_2_iid_f1_vs_dimensionality.png")
    fig4_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_3_quantum_minus_rbf_vs_dimensionality.png")
    add_figure_ieee(doc, fig3_path, 3, "In-distribution dimensionality scaling trajectory (2D to 12D) on MeAJOR. Mean test F1 across 10 seeds with 95% bootstrap CI bands.")
    add_figure_ieee(doc, fig4_path, 4, "Paired Quantum minus Classical RBF difference (ΔF1) vs dimensionality with pre-registered practical equivalence zone (ε = ±0.01).")

    add_heading_2(doc, "5.4 Representation effects and ranking reversals")
    add_body_p(
        doc,
        "How a kernel interacts with its upstream representation matters much more than the average gap between quantum and classical. On CEAS 2008 with TF-IDF plus SVD, the quantum kernel is slightly ahead, 0.9736 to 0.9641. Switch to dense RoBERTa embeddings and the order flips: RBF reaches 0.9896 and quantum only 0.9601."
    )
    add_body_p(
        doc,
        "SMS Spam shows the effect even more starkly. With sparse TF-IDF, RBF beats quantum by 19.52 percentage points. With dense MPNet embeddings, RBF scores 0.9045 while quantum drops to 0.3756, a gap of 52.88 points. Any statement about how a model performs is therefore inseparable from the representation it was given."
    )

    # Table 7: Representation Screening
    t7_headers = ["Corpus", "Representation", "Quantum F1", "RBF F1", "Difference", "Interpretation"]
    t7_data = [
        ["CEAS 2008", "Sparse TF-IDF + SVD", "0.9736", "0.9641", "+0.95 pp", "Small quantum edge"],
        ["CEAS 2008", "Dense RoBERTa-base", "0.9601", "0.9896", "−2.95 pp", "RBF uses Euclidean clustering better"],
        ["SMS Spam", "Sparse TF-IDF + SVD", "0.6324", "0.8276", "−19.52 pp", "RBF is clearly stronger"],
        ["SMS Spam", "Dense MiniLM-L6", "0.7707", "0.7930", "−2.23 pp", "Gap narrows"],
        ["SMS Spam", "Dense MPNet-base", "0.3756", "0.9045", "−52.88 pp", "Severe quantum degradation"]
    ]
    add_table_ieee(doc, "VII", "Upstream Representation Screening and Ranking Reversals", t7_headers, t7_data, col_widths=[0.60, 0.85, 0.45, 0.45, 0.55, 0.55], font_size=5.8)

    fig5_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_6_representation_interaction.png")
    add_figure_ieee(doc, fig5_path, 5, "Upstream representation interaction and ranking inversion on CEAS 2008 and SMS Spam. Switching from sparse TF-IDF to dense contextual transformers inverts quantum-vs-classical rankings by up to 52.88 pp.")

    add_heading_2(doc, "5.5 Cross-source domain shift")
    add_body_p(
        doc,
        "For the domain-transfer test, models were trained on TREC 2007 and evaluated on a balanced mix of TREC 2005 and TREC 2006. The quantum kernel falls to an F1 of 0.6680 ± 0.0094, a 23.7% relative drop from its in-distribution score. The matched RBF reaches 0.6913 ± 0.0161, a smaller 20.6% relative drop."
    )
    add_body_p(
        doc,
        "Relative to RBF, the quantum kernel is 2.33 percentage points lower. The bootstrap confidence interval sits entirely below zero, the permutation p-value is 0.0046, and the FDR-adjusted p-value is 0.0069. This result is not practically equivalent; under the source shift we tested, the matched RBF is significantly more reliable."
    )

    # Table 8: Domain Holdout
    t8_headers = ["Model", "IID F1", "Holdout F1", "Absolute drop", "Relative drop", "PR-AUC", "ROC-AUC"]
    t8_data = [
        ["Linear SVM", "0.8445 ± 0.002", "0.6622 ± 0.014", "−0.1823", "−21.6%", "0.7812", "0.7950"],
        ["Matched RBF", "0.8709 ± 0.003", "0.6913 ± 0.016", "−0.1796", "−20.6%", "0.8115", "0.8240"],
        ["Quantum kernel", "0.8754 ± 0.003", "0.6680 ± 0.009", "−0.2074", "−23.7%", "0.7890", "0.8010"]
    ]
    add_table_ieee(doc, "VIII", "Cross-Source Domain Holdout Performance (MeAJOR Direction B, N=10 Seeds)", t8_headers, t8_data, col_widths=[0.65, 0.50, 0.50, 0.40, 0.40, 0.40, 0.40], font_size=5.8)

    fig6_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_4_iid_vs_source_holdout.png")
    add_figure_ieee(doc, fig6_path, 6, "In-distribution vs cross-source domain holdout (Direction B: TREC 2007 -> TREC 2005/2006). Parameter-free quantum fidelity kernels suffer greater degradation under source shift than classical RBF.")

    add_heading_2(doc, "5.6 Kernel-geometry diagnostics")
    add_body_p(
        doc,
        "Off-diagonal entries of the quantum and RBF Gram matrices correlate moderately, at roughly 0.55 to 0.65 through 12 dimensions, dropping to 0.4566 at 16D. The quantum kernel partly follows the RBF similarity pattern but also departs from it in structure."
    )
    add_body_p(
        doc,
        "The quantum kernel's centered kernel-target alignment is 50 to 60% lower across datasets, about 0.022 to 0.040 versus about 0.060 to 0.077 for RBF. That lower alignment fits with the performance limits we observed."
    )
    add_body_p(
        doc,
        "Single-state von Neumann entropy has a strong negative relationship with how varied the pairwise Gram-matrix entries are: r = −0.8257 on SMS, −0.8170 on CEAS, and −0.7822 on MeAJOR. In plain terms, spreading each quantum state over more basis states doesn't help discrimination here. It tends to make the pairwise similarities more uniform."
    )
    add_body_p(
        doc,
        "At 8D the quantum effective rank is about 14.2, compared with 18.6 for RBF, which points to a modest concentration of the quantum Gram matrix's spectrum."
    )

    fig7_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_7_geometry_correlation_vs_dimensionality.png")
    fig8_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_8_entropy_vs_kernel_diversity.png")
    add_figure_ieee(doc, fig7_path, 7, "Quantum vs classical RBF off-diagonal Gram matrix correlation across dimensionality (2D to 16D). Correlation remains moderate (r ≈ 0.55–0.65).")
    add_figure_ieee(doc, fig8_path, 8, "Single-state basis dispersion entropy vs pairwise Gram matrix diversity. Strong negative correlation (r = -0.78 to -0.83) confirms amplitude dispersion collapses pairwise variance.")

    add_heading_2(doc, "5.7 Simulation cost and memory limits")
    add_body_p(
        doc,
        "Exact statevector simulation gets expensive quickly as dimensionality grows. At 8D, computing the quantum kernel takes 17.2 seconds against 1.4 seconds for RBF, a 12.3× slowdown. At 12D it takes 108.8 seconds against 1.7, a 64× slowdown."
    )
    add_body_p(
        doc,
        "At 16D, simulating 10,000 samples means 65,536 complex amplitudes per sample and more than 10.5 GB of memory, which our workstation couldn't handle. RBF finished in 2.1 seconds using under 1 MB."
    )

    # Table 9: Simulation Cost
    t9_headers = ["Dimension", "Quantum time", "RBF time", "Slowdown", "Peak RAM", "Status"]
    t9_data = [
        ["2 qubits", "3.4 s", "1.3 s", "2.6×", "142 MB", "Feasible"],
        ["4 qubits", "5.8 s", "1.3 s", "4.5×", "185 MB", "Feasible"],
        ["8 qubits", "17.2 s", "1.4 s", "12.3×", "680 MB", "Feasible"],
        ["12 qubits", "108.8 s", "1.7 s", "64.0×", "6.4 GB", "Heavy simulation"],
        ["16 qubits", "Infeasible", "2.1 s", "—", ">10.5 GB", "Out-of-memory limit"]
    ]
    add_table_ieee(doc, "IX", "Computational Complexity and Memory Scaling on 10,000 Samples", t9_headers, t9_data, col_widths=[0.45, 0.60, 0.50, 0.35, 0.50, 0.65], font_size=5.8)

    fig9_path = os.path.join(PROJECT_ROOT, "paper", "submission", "figures", "figure_5_runtime_vs_dimensionality.png")
    add_figure_ieee(doc, fig9_path, 9, "Wall-clock execution time and peak RAM on 10,000 samples. Simulating 12 qubits incurs a 64x execution penalty over RBF, with 16D exceeding workstation RAM limits.")

    # -------------------------------------------------------------
    # 6. SYNTHESIS
    # -------------------------------------------------------------
    add_heading_1(doc, "6.  SYNTHESIS")
    add_body_p(
        doc,
        "Four results stand out. First, in controlled in-distribution settings, a parameter-free quantum fidelity kernel is practically equivalent to a matched classical RBF kernel, and by 12D the difference is only +0.14 percentage points."
    )
    add_body_p(
        doc,
        "Second, the choice of representation can swing performance by as much as 52.88 percentage points and can even reverse which kernel comes out ahead. Third, under the cross-source shift we measured, the quantum kernel is more fragile, trailing RBF by 2.33 points. Fourth, exact classical simulation of the quantum kernel carries large runtime and memory costs, and we saw no predictive benefit in return."
    )
    add_body_p(
        doc,
        "It follows that a tiny accuracy edge in one controlled experiment shouldn't be called quantum advantage. That label would need replication, equivalence analysis, robust baselines, domain-shift evaluation, and cost accounting behind it. For the text-security settings we evaluated, linear models on full TF-IDF and tuned classical RBF kernels remain the stronger engineering choices."
    )

    # -------------------------------------------------------------
    # 7. CONCLUSION
    # -------------------------------------------------------------
    add_heading_1(doc, "7.  CONCLUSION")
    add_body_p(
        doc,
        "This paper benchmarks parameter-free quantum fidelity kernels against matched classical baselines on three cybersecurity datasets containing 153,410 records. In-distribution, the quantum and RBF kernels are practically equivalent, and the choice of representation matters far more than the choice of kernel."
    )
    add_body_p(
        doc,
        "Under cross-source domain shift the quantum kernel loses more performance than the classical RBF. Exact statevector simulation also scales badly, growing much slower with each added dimension and eventually running out of memory."
    )
    add_body_p(
        doc,
        "Future claims of quantum advantage in text processing should control for representation and dimensionality, match classical baselines carefully, test under meaningful domain shift, replicate results statistically, and report computational cost next to predictive metrics. A small numerical edge on its own isn't enough to show a useful real-world advantage."
    )

    # Save documents to root directory
    out_docx = os.path.join(PROJECT_ROOT, "RESEARCH_PAPER_ABSTRACT_TO_CONCLUSION.docx")
    doc.save(out_docx)
    print(f"Successfully generated DOCX paper (Abstract to Conclusion) at: {out_docx}")

if __name__ == "__main__":
    build_paper()
