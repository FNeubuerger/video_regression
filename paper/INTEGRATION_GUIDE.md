# Integration Guide: Introduction Draft → main.tex

## Section-by-Section Mapping

### What You Have Now (main.tex)
```tex
\section{Introduction and Motivation}
  - HIFU context (brief)
  - Privacy constraint
  - Edge device requirement
  - UQ requirement
  - 3-bullet list of contributions

\section{Related Work}
  \subsection{Non-Invasive Temperature Monitoring}
    - MRT mention
    - Ultrasound-based thermometry
  \subsection{Deep Learning in Medical Imaging}
    - ResNet and medical imaging
    - Computational intensity issue
  \subsection{Physics-Informed Machine Learning}
    - PINN definition
    - Heat diffusion example

\section{Research Questions}
  - RQ1-RQ4
```

### What INTRODUCTION_DRAFT provides
```tex
\section{Introduction and Motivation}
  \subsection{Clinical Context} ← NEW: Clinical background
    - MWA clinical applications
    - Advantages over RFA (with specs)
    - Temperature standards
    - Why monitoring matters
  
  \subsection{Monitoring Modalities} ← NEW: Comprehensive comparison
    - MRT pros/cons
    - CT pros/cons
    - Ultrasound pros/cons ← YOUR FOCUS
    - Invasive sensors pros/cons
    - Gap identified
  
  \subsection{Deep Learning for US} ← NEW: Recent advances
    - Zhang et al. CNN work
    - Current limitations
    - Research gap
  
  \subsection{Physics-Informed ML} ← EXPANDED: More technical depth
    - PINNs intro
    - Bioheat equation (with math)
    - Medical application
  
  \subsection{Edge Computing} ← NEW: Systematic requirements
    - Data privacy
    - Real-time (>30 FPS threshold)
    - Hardware constraints (Raspberry Pi, Jetson Nano)
    - Interpretability
    - Tension: accuracy vs efficiency
  
  \subsection{Research Gaps} ← NEW: Explicit gap analysis
    - 5 specific research gaps
  
  \subsection{Contributions} ← EXPANDED: Detailed list
    - 6 contributions with technical detail

Related Work section (optional separate draft available)

\section{Research Questions}
  - RQ1-RQ4 (unchanged, can keep as-is)
```

---

## How to Integrate: Three Approaches

### APPROACH 1: Surgical Replacement (Keep everything else)
```tex
% In main.tex, FIND:
\section{Introduction and Motivation}
Estimating physical parameters, such as temperature, from medical imaging data...
% TO:
\section{Research Questions}

% REPLACE WITH: Content from INTRODUCTION_DRAFT.tex
% Lines: Start at \section{Introduction and Motivation}
%        End at: (just before \section{Related Work})
```

**Pros:** 
- Clean integration
- Preserves all your existing structure elsewhere
- Easy to revert

**Cons:**
- May lose some of your original phrasing

---

### APPROACH 2: Selective Enrichment (Keep your structure, add depth)

Mix your current intro with DRAFT by:

1. **Keep your clinical hook:** First 2 paragraphs from your current intro
2. **Add:** Clinical Context subsection from DRAFT (1.1)
3. **Replace:** Your current Related Work → Monitoring Modalities subsection from DRAFT (1.2)
4. **Keep:** Your current DeepLearning and Physics subsections BUT cite Zhang et al. from DRAFT (1.3-1.4)
5. **Add:** Edge Computing requirements subsection from DRAFT (1.5)
6. **Add:** Research Gaps subsection from DRAFT (1.6)
7. **Keep:** Your contributions list but label as (1.7)

**Pros:**
- Preserves your original voice
- Systematic integration
- Progressive detail increase

**Cons:**
- More manual work
- Need to ensure smooth transitions

---

### APPROACH 3: Reference Only (Keep your intro, use draft for citations)

Keep your `main.tex` introduction exactly as-is, but:
1. Add missing citations from DRAFT where appropriate
2. Use DRAFT's section numbering/structure as template for "Related Work"
3. Reference DRAFT when writing Methods section

**Pros:**
- Minimal disruption
- Preserves your current flow
- Good for quick Polish

**Cons:**
- Doesn't leverage depth of DRAFT

---

## Citation Mappings Needed

### From INTRODUCTION_DRAFT, these citations appear:
```
\cite{vogl2017}           → MWA basics, advantages over RFA
\cite{shibata2000}        → MWA vs RFA comparison
\cite{ahmed2014}          → Technical success criteria, ablative margin
\cite{lubner2010}         → MWA vs RFA advantages
\cite{geoghegan2022}      → Monitoring modalities review (MOST CITED)
\cite{zhang2020}          → CNN for thermal lesion detection
\cite{seip2002}           → Ultrasound thermometry limitations
\cite{raissi2019physics}  → PINNs definition
```

### Map to your references.bib:
Create BibTeX entries with these keys. Example format:

```bibtex
@article{zhang2020,
  title={Detection and Monitoring of Thermal Lesions Induced by Microwave Ablation Using Ultrasound Imaging and Convolutional Neural Networks},
  author={Zhang, Siyuan and Wu, Shan and Shang, Shaoqiang and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={24},
  number={4},
  pages={965--976},
  year={2020},
}

@article{geoghegan2022,
  title={Methods of monitoring thermal ablation of soft tissue tumors: A comprehensive review},
  author={Geoghegan, Joseph P. and others},
  journal={Medical Physics},
  year={2022},
}
```

---

## Recommended Integration Workflow

### Step 1: Extract Citation Data
```bash
cd /home/helena/Dokumente/video_regression/paper
# Copy key BibTeX entries from your PDFs into references.bib
# Use the reading_notes_UPDATED.csv as guide
```

### Step 2: Choose Integration Approach
- For major overhaul → APPROACH 1 (full replacement)
- For quality improvement → APPROACH 2 (selective enrichment)
- For final polish → APPROACH 3 (reference only)

### Step 3: Copy Content
- Open both `main.tex` and `INTRODUCTION_DRAFT.tex` side-by-side
- Copy/paste according to chosen approach

### Step 4: Replace Citation Keys
- Use Find & Replace to map `\cite{vogl2017}` → `\cite{yourkey}` once BibTeX added

### Step 5: Test
```bash
cd /home/helena/Dokumente/video_regression/paper
pdflatex main.tex  # Check for citation errors
```

---

## Quality Checks Before Committing

- [ ] All `\cite{}` commands resolve (no undefined references)
- [ ] Section numbering is sequential and makes sense
- [ ] Spacing between subsections is consistent
- [ ] Grammar/tone matches rest of paper
- [ ] All research questions (RQ1-RQ4) are covered
- [ ] Clinical motivation is clear to non-ML audience
- [ ] Technical depth is appropriate for IEEE conference
- [ ] Contributes explicit transition to Methods section

---

## Estimated Time to Integration

| Approach | Time | Difficulty |
|----------|------|------------|
| Approach 1 (Full) | 30-45 min | Easy |
| Approach 2 (Selective) | 60-90 min | Medium |
| Approach 3 (Reference) | 15-30 min | Easy |

---

**Ready to proceed?** Start with Approach 2 (selective enrichment) for best balance of quality and speed.

