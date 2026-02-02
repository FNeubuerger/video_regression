# PDF Literature Search Summary
**Date: January 19, 2026**
**Source: 16 PDFs from `research/literature/unread/` directory**

---

## PDFs Found & Ready to Use

### Medical Imaging & Monitoring (5 papers)
1. **Detection_and_Monitoring_of_Thermal_Lesions_Induced_by_Microwave_Ablation_Using_Ultrasound_Imaging_and_Convolutional_Neural_Networks.pdf**
   - CNN for thermal lesion segmentation from US RF data
   - Compares B-mode vs CNN approaches
   - **Use for:** Deep learning methods, ultrasound monitoring, CNN architecture

2. **Medical Physics - 2022 - Geoghegan - Methods of monitoring thermal ablation of soft tissue tumors A comprehensive review.pdf**
   - Review of ALL monitoring modalities (MRI, CT, US, thermocouple)
   - **Use for:** Comprehensive monitoring section, state-of-the-art

3. **Measurement_and_Analysis_of_Tissue_Temperature_During_Microwave_Liver_Ablation.pdf**
   - Tissue temperature measurement during MWA
   - **Use for:** Temperature measurement methods, thermal dynamics

4. **radiol.2383050262.pdf** & **nihms-894127.pdf**
   - MWA clinical applications and imaging guidance
   - **Use for:** Clinical context, MWA advantages

### MWA Clinical & Technical (6 papers)
5. **Simulation_of_Image-Guided_Microwave_Ablation_Therapy_Using_a_Digital_Twin_Computational_Model.pdf**
   - Computational modeling of MWA
   - **Use for:** Physics-informed approaches, temperature simulation

6. **sonographically-guided-microwave-coagulation-treatment-of-liver-cancer-an-experimental-and-clinical-study.pdf**
   - Experimental & clinical MWA procedures
   - **Use for:** Clinical methods, experimental setup validation

7. **s12885-021-08099-7.pdf**, **s12876-025-04081-w.pdf**, **WJG-19-5430.pdf**, **cancers-17-00409.pdf**
   - MWA outcomes, liver cancer, efficacy studies
   - **Use for:** Clinical context, ablation zone characteristics

### Physics & Tissue Properties (3 papers)
8. **1-s2.0-S0301562901005191-main.pdf** & **1-s2.0-S0720048X09001776-main.pdf**
   - Tissue properties, dielectric properties, thermal conductivity
   - **Use for:** Physics background, tissue-specific parameters

9. **srep41246.pdf**, **13645700500470025.pdf**
   - Ablation mechanisms, thermal effects
   - **Use for:** Tissue damage mechanisms, temperature thresholds

### Monitoring & Sensors (2 papers)
10. **sensors-19-00977-v2.pdf** (duplicate: sensors-19-00977-v2-1.pdf)
    - Sensor-based monitoring and real-time feedback
    - **Use for:** Edge deployment, real-time monitoring

11. **cancers-16-01700.pdf**
    - Cancer treatment modalities, ablation techniques
    - **Use for:** Clinical overview

---

## Key Topics for Introduction (Extracted from PDFs)

### ✅ 1. MWA Clinical Applications & Advantages
**Found in:** Zhang et al. (CNN paper), clinical papers, reviews

**Key Points:**
- MWA advantages over RFA:
  - Faster heating
  - Higher intratumoral temperatures
  - Larger ablation zones with shorter ablation times
  - Better performance in high-perfusion areas
  - Less heat-sink effect
  - More predictable ablation zone
  
- Clinical applications: Liver, lung, kidney, bone, breast, uterine fibroids
- Standard temperature thresholds:
  - 50-60°C for coagulation (target for ablation)
  - >100°C for vaporization
  - Successful ablation: 50-60°C maintained ≥5 minutes with safety margin 5-10mm

### ✅ 2. Monitoring Modalities Comparison

**Found in:** Geoghegan review (2022), clinical papers

**Monitoring approaches:**
1. **MRI Thermometry**
   - Advantage: High spatial/temporal resolution, near real-time temperature maps
   - Disadvantage: Expensive, complex, incompatible with pacemakers, limited availability

2. **CT Imaging**
   - Advantage: Shows ablation zone post-procedure
   - Disadvantage: No real-time feedback, ionizing radiation

3. **Ultrasound (B-mode)**
   - Advantage: Real-time, portable, low-cost, non-invasive, widely available
   - Disadvantage: Low intrinsic contrast, difficulty distinguishing thermal lesion from normal tissue
   - Enhancement: CNN-based segmentation improves detection vs. manual B-mode

4. **Invasive Thermometry** (Thermocouple, Fiber-optic)
   - Advantage: High-precision point measurements
   - Disadvantage: Only measures discrete locations, adds procedural complexity

5. **Indirect US Markers:**
   - Speckle tracking
   - Echo shift/decorrelation
   - Hyperechogenic microbubbles (overestimates ablation zone)
   - Backscattered energy changes

### ✅ 3. Ultrasound-Based Monitoring & Video Analysis

**Found in:** Zhang et al., clinical monitoring papers

**Visual indicators of thermal lesion:**
- Hyperechogenic focus (microbubbles)
- Changes in backscattered RF energy
- Speckle decorrelation
- Coagulation necrosis visualization

**Challenges:**
- Low contrast between lesion and normal tissue in B-mode
- Motion artifacts
- Device variability
- Indirect mapping to absolute temperature

### ✅ 4. Deep Learning in Medical Imaging

**Found in:** Zhang et al., sensor papers

**CNN architectures for ablation monitoring:**
- Modified ResNet-18 for feature extraction
- CNN with segmentation heads (SIresNet, SIm-CNN)
- LSTM for temporal dynamics
- Performance: AUC 0.88-0.89 (CNN) vs 0.69 (B-mode)

**Applications:**
- Thermal lesion detection & segmentation
- Real-time ablation zone monitoring
- Operator support for treatment guidance

### ✅ 5. Physics-Informed Approaches

**Found in:** Simulation paper, computational studies

**Relevant physics:**
- Bioheat equation (Pennes equation)
- Heat diffusion: $\frac{\partial T}{\partial t} = \beta \nabla^2 T$
- Perfusion cooling: $\alpha(T - T_{arterial})$
- External heating: $Q_{ext}$
- Tissue properties: thermal conductivity, perfusion rate

**Computational modeling:**
- Digital twin approaches for temperature simulation
- Temperature-dependent tissue properties
- Ablation zone prediction

### ✅ 6. Edge Deployment & Real-Time Processing

**Found in:** Sensor papers, deployment discussions

**Edge device targets:**
- Raspberry Pi 4
- NVIDIA Jetson Nano
- Real-time requirement: ≥30 FPS for HIFU feedback
- CPU-based inference (no GPU in OR)

---

## Summary of What's Available

| Topic | Paper(s) | Quality | Use For |
|-------|----------|---------|---------|
| **MWA clinical context** | 6 papers | ⭐⭐⭐ | Introduction, clinical motivation |
| **Monitoring modalities review** | Geoghegan 2022 | ⭐⭐⭐⭐ | Comprehensive comparison, benchmarking |
| **CNN for thermal lesion detection** | Zhang et al. | ⭐⭐⭐ | Methods, deep learning approaches |
| **Temperature measurement** | 2-3 papers | ⭐⭐⭐ | Experimental setup validation |
| **Physics/bioheat equation** | Simulation paper | ⭐⭐⭐ | Physics-informed methods |
| **Ultrasound monitoring techniques** | 3-4 papers | ⭐⭐⭐ | Monitoring modality section |
| **Edge deployment** | Sensor papers | ⭐⭐ | Real-time requirements |

---

## Next Steps for Writing Introduction

### Section 1: Clinical Context (use MWA clinical papers + review)
- [ ] MWA definition and clinical applications
- [ ] MWA advantages over RFA
- [ ] Clinical indications and contraindications
- [ ] Temperature thresholds for effective ablation

### Section 2: Monitoring Challenge (use Geoghegan review + clinical papers)
- [ ] Why monitoring is critical
- [ ] Limitations of current approaches
- [ ] Comparison table: MRI vs CT vs US vs invasive sensors
- [ ] Why ultrasound-based approaches are attractive but challenging

### Section 3: Related Work on US Video Analysis (use Zhang CNN paper + monitoring papers)
- [ ] Ultrasound visualization of thermal lesions
- [ ] CNN applications to US imaging
- [ ] Speckle tracking and RF analysis methods
- [ ] Current limitations and gap

### Section 4: Physics-Informed ML & Edge Computing (use simulation paper + sensor papers)
- [ ] Brief mention of physics constraints in ML
- [ ] Real-time edge deployment requirements
- [ ] Research gap: video regression + uncertainty + edge deployment

### Section 5: Our Contributions (from main.tex)
- [ ] Propose CNN-LSTM architecture optimized for edge
- [ ] Compare against PINNs and pretrained models
- [ ] Integrate uncertainty quantification
- [ ] Benchmark on Raspberry Pi / Jetson Nano

---

## File Organization Recommendation

```
research/
├── literature/
│   ├── reading_notes.csv            [update with summaries]
│   ├── monitoring/                  [high-priority papers]
│   │   ├── Zhang_2020_CNN.pdf       [rename & move]
│   │   ├── Geoghegan_2022_Review.pdf
│   │   └── Measurement_Temperature.pdf
│   ├── unread/                      [other papers]
│   └── PDF_EXTRACTION_SUMMARY.md    [this file - your reference]
```

---

