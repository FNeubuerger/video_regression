# Plan: Write Introduction & Experimental-Setup Writing (literature-only workflow)

Purpose
- Focus only on the writing and literature-research parts for the paper.
- This file is a living checklist + set of templates you can copy into your repo, fill with citations, and use while writing the Introduction and Experimental Setup (without running lab work).

How to use this file
- Work top → bottom. Complete small checkboxes one at a time.
- Save PDFs and BibTeX in a citation manager (Zotero/Mendeley) or `paper/references.bib`.
- Put short paper summaries in the "Papers & notes" table below.
- Copy the ready-to-use paragraph templates into your LaTeX file and substitute citation placeholders when you have BibTeX keys.

Quick checklist (writing & literature only)
- Phase 1 — Setup
  - [x] Create folder `literature` and a bibliography file `paper/references.bib`
  - [x] Create a `literature/reading_notes.csv` (columns suggested below)
  - [ ] Install/prepare Zotero or Mendeley (optional)
- Phase 2 — Search & collect
  - [ ] Run targeted searches (queries below) and save candidate PDFs (20–40)
  - [ ] Select top 12–16 papers and add BibTeX to `paper/references.bib`
  - [ ] For each selected paper, write a 2–3 sentence summary into the CSV
- Phase 3 — Drafting
  - [ ] Draft Introduction (use templates below)
  - [ ] Draft Experimental-Setup Methods text using existing notes (placeholders for missing numbers)
  - [ ] Add figure placeholders and caption drafts
- Phase 4 — Revise & polish
  - [ ] Replace citation placeholders with actual `\cite{}` keys in LaTeX
  - [ ] Proofread; ask a colleague to check reproducibility of written methods
  - [ ] Finalize reference formatting

Search queries (copy-paste)
- Clinical context & reviews
  - "microwave ablation clinical indications outcomes review"
  - "microwave ablation liver lung kidney review 2.45 GHz"
- Physics & antenna design
  - "microwave ablation antenna design 2.45 GHz active tip geometry heating pattern"
  - "microwave heating tissue dielectric properties 2.45 GHz 915 MHz"
- Thermometry & monitoring modalities
  - "MR thermometry review microwave ablation PRF thermometry"
  - "fiber optic thermometer microwave ablation calibration comparison thermocouple"
- Ultrasound-based monitoring & video methods
  - "ultrasound visualization thermal lesion coagulation echogenicity"
  - "optical flow ultrasound temperature estimation thermal ablation"
  - "video-based thermometry ultrasound speckle tracking review"
- Phantoms & protein-doped agar
  - "egg white agar phantom coagulation ultrasound"
  - "agar phantom acoustic thermal properties review albumen"
- Safety & dosimetry
  - "CEM43 thermal dose coagulation review"
  - "microwave ablation safety water cooling emergency stop guidelines"

Papers & notes (suggested CSV columns; create `paper/lit/reading_notes.csv`)
- Columns: bibkey,title,authors,year,category(one-word),one_line_summary,where_to_cite,notes,filepath
- Example row:
  - `menikou2017,Acoustic and thermal characterization...,Menikou & Damianou,2017,phantom,"Methods to measure speed of sound & thermal diffusivity in agar phantoms","Methods/Phantom properties","measured attenuation at 3 MHz","paper/lit/menikou2017.pdf"`

Priority paper categories to collect first
- 1–2 Reviews on MWA clinical practice and outcomes
- 1–2 Papers comparing MWA vs RFA (physics/clinical)
- 2–3 Papers on thermometry methods (MRI, fiber-optic)
- 3–5 Papers on ultrasound visualization of thermal lesions and phantoms (egg-white)
- 1–2 Papers on computational/video-based monitoring, optical flow in ultrasound
- 1 Paper on physics-informed ML or PINNs for thermal problems

Reading & extraction protocol (fast method)
- Read abstract → conclusion → methods → figures
- Fill one-line summary and "where_to_cite" in CSV immediately
- Save BibTeX and PDF with consistent filenames: `paper/lit/<bibkey>.pdf`

Ready-to-use paragraph templates
- Copy these into your LaTeX `paper/main.tex` and replace [CITATION_X] or insert corresponding `\cite{bibkey}`.

Clinical context (paste into paper)
Microwave ablation (MWA) is a minimally invasive thermal therapy used to treat a range of solid tumors, notably in the liver, lung and kidney. Compared to radiofrequency ablation, MWA typically produces faster heating, higher intratumoral temperatures and larger ablation zones with reduced sensitivity to vascular heat-sink effects, which improves efficacy for lesions adjacent to large vessels [CITATION_MWA_REVIEW, CITATION_MWA_VS_RFA]. Despite growing clinical adoption, intra‑procedural monitoring of temperature and lesion extent remains critical to ensure complete tumor coverage while limiting collateral damage.

Monitoring modalities and limitations
Accurate temperature mapping is essential for safe and effective ablation. Magnetic resonance thermometry (MRT) provides quantitative temperature maps but is resource-intensive and not universally available in operating rooms [CITATION_MRT]. Invasive sensors such as thermocouples and fiber‑optic probes provide high-temporal-resolution point measurements but sample only discrete locations and increase procedural complexity [CITATION_THERMOCOUPLE]. Ultrasound is widely available and portable, but conventional B‑mode ultrasound does not provide direct, quantitative temperature maps; instead, it reveals indirect acoustic changes (echogenicity, bubble formation, speckle decorrelation) that can be exploited by computational methods [CITATION_ULTRASOUND_COAG].

Ultrasound video-based approaches & computational cues
Thermal coagulation causes changes in acoustic properties and speckle that can be observed in ultrasound videos. Video-based computational approaches—speckle-tracking, elastography, and optical-flow analysis—have been proposed to extract spatiotemporal signatures of heating [CITATION_VIDEO_METHODS]. These approaches are attractive for real‑time, edge-capable monitoring but are challenged by device variability, motion artifacts, and the indirect mapping between acoustic change and absolute temperature.

Role of phantoms and study motivation
Phantoms that mimic acoustic and thermal tissue properties, especially protein-doped agar phantoms, are widely used to develop and validate ultrasound-based monitoring techniques because they reproduce visible coagulation effects under heating [CITATION_PHANTOMS, CITATION_EGGWHITE]. This work focuses on whether lightweight deep-learning architectures, augmented with physical priors and uncertainty quantification, can estimate temperature dynamics from ultrasound video in real time on CPU-class hardware, enabling a complementary intra‑procedural monitoring modality.

Gap statement & aims
Although prior work demonstrates ultrasound-based indicators of heating, there is limited evidence for real-time, device-agnostic temperature regression from ultrasound video that is lightweight enough for on-device deployment and that returns principled uncertainty estimates. Our aims are to (1) design and compare lightweight CNN–LSTM and physics-informed architectures for video regression of temperature, (2) quantify uncertainty using ensembles and Bayesian methods, and (3) benchmark inference latency on CPU targets to assess feasibility of on‑device deployment.

Figure & table checklist (for Introduction/Methods)
- Fig 1: schematic of experimental setup (annotated)
- Fig 2: representative ultrasound frames showing coagulation
- Table 1: monitoring modalities comparison (MRI / ultrasound / thermocouple / fiber-optic)
- Table 2: prioritized literature (key references and short rationale)

Writing schedule (example, adjust to your pace)
- Day 1: Collect 10 candidate papers and add BibTeX (2–3 h)
- Day 2: Select top 12, write one-line summaries (2–3 h)
- Day 3: Draft Introduction using templates above (2–3 h)
- Day 4: Insert citations, revise and prepare figures/tables (2–4 h)


Notes & placeholders
- Where numbers, brands, or settings are missing in your Methods, place a clear TODO like: `TODO: confirm antenna active tip length (mm) and model`.
- Keep a short changelog at top of the file when you revise (date, what you changed).

