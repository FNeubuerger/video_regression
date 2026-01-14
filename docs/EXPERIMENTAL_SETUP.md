**Phantom composition**

- **Bulk composition:** 600 mL deionized water, 30 g agar (~5% w/v).
  
- **Protein additive:** egg white added at 30 wt% (percentage relative to total phantom mass).

- **Changes in recipe over time:** Limitation of the amount of agar to 20g/25g; Series of measurements with different phantom compositions still pending
  
**Phantom preparation (proposed protocol)**

- **Hydration:** Mix agar powder with 100 mL cold water and allow to hydrate for 10 min.
- **Heat source:** Bring the remaining water to a rolling boil.
- **Dissolution:** On a hotplate with stirring, combine the boiling water and the hydrated agar in a beaker; continue heating until the mixture reaches 95 °C.
- **Hold time:** Maintain 95 °C for 5 min to ensure complete dissolution.
- **Transfer & cool:** Pour the hot mixture into a receiving container and allow the temperature to decrease.
- **Protein addition:** When the mixture reaches approximately 55 °C, add egg white corresponding to 30 wt% through a fine sieve to remove bubbles and particulates.
- **Mixing:** Gently homogenize while avoiding air entrainment.
- **Solidification:** Allow the phantom to cool to ambient temperature until fully set.

**Antenna and generator**

- **Generator:** SABERWAVE ECO-200G (ECO Medical Technology, Nanjing, China). Described as a CE-marked 2.45 GHz water-cooled microwave ablation system.
- **Operating note:** Original notes list 30 W for 10 min, also experiments with higher power (increase by 5 W each new measurement cycle) for 5 min exist; antenna geometry/specification not recorded — see verification tasks.

**Experimental setup & instrumentation**

- **Mounting:** Cooled phantom placed in a 3D‑printed holder designed with insertion ports for the antenna and sensors.
- **Probe placement:** Insert MWA antenna and four temperature sensors through dedicated openings; sensors numbered 1–4 starting at the upper left and arranged symmetrically around the antenna active zone.
- **Thermometry:** Fiber‑optic thermometer (COMEM Optocon GmbH, Dresden, Germany) recording at 250 ms intervals.
- **Ultrasound imaging:** Handheld ultrasound (VScan Air, GE Healthcare) records B‑mode video at 30 fps.
- **Video capture workflow:** Handheld connects by Bluetooth to a tablet; continuous recordings are captured via screen recording because the device itself limits internal recordings to ~10 s.

**Data acquisition & synchronization**

- **Sampling rates:** Temperature logged at 250 ms intervals; ultrasound at 30 fps. Implement timestamping or trigger signals to enable temporal alignment between thermometry and imaging.
- **Recommendation:** Log system times from the ultrasound tablet and the thermometry acquisition computer; perform a synchronization test (e.g., audible/visual trigger) before experiments.

**Safety & handling**

- **Hazards:** Hot liquids and heated agar present burn risk; microwave ablation systems require trained operation and a functioning water‑cooling loop.
- **Controls:** Use appropriate PPE, verify emergency shutoff for the microwave generator, and ensure containment for any spilled hot material.

**Actionable to‑dos & lab follow‑ups**

- [ ] Verify SABERWAVE ECO‑200G technical documentation (datasheet, antenna model(s), active tip geometry, recommended power/time settings, water-cooling requirements, and CE documentation).
- [ ] Confirm interpretation of “30 wt% egg white” (percentage relative to total phantom mass or other basis) and justification for its use
- [ ] search phantom literature for similar protein-doped agar phantoms.
- [ ] Reproduce the phantom at small scale and record exact masses, volumes, temperatures, and timings; document variability across replicates.
- [ ] Test VScan Air → tablet → screen-capture workflow: verify continuous recording capability, frame integrity, Bluetooth reliability, and temporal alignment with temperature logs.
- [ ] Measure or obtain acoustic and thermal properties of the phantom (speed of sound, attenuation, thermal diffusivity) from literature or laboratory measurements.
- [ ] Verify the 3D‑printed holder material compatibility with heating and microwave exposure; document mounting reproducibility.

---

Last edited: 9 January 2026

**References & notes**

- Iizuka, N. et al., "Optical phantom materials for near infrared laser photocoagulation studies," Lasers in Surgery and Medicine (1999). Relevant for protein-doped phantoms and coagulation contrast. PubMed: https://pubmed.ncbi.nlm.nih.gov/10455223/
- Menikou, E. & Damianou, C., "Acoustic and thermal characterization of agar based phantoms used for evaluating focused ultrasound exposures," Journal of Therapeutic Ultrasound (2017). Methods for measuring acoustic/thermal properties and use of protein additives: https://link.springer.com/article/10.1186/s40349-017-0093-z
- Dabbagh, A. et al., "Tissue‑Mimicking Gel Phantoms for Thermal Therapy Studies," Ultrasonic Imaging (2014). Review of phantom recipes, measurement techniques, and use of albumen/BSA: https://doi.org/10.1177/0161734614526372
- Egg‑white phantom methods (recipe & characterization example): Academia preprint PDF (reports 30% egg white effectiveness for ultrasound lesion visualization): https://www.academia.edu/download/98095705/JAKO201518050733492.pdf

**Placement of references in this file**

- Phantom composition & preparation: see Iizuka 1999; Menikou & Damianou 2017; egg‑white PDF for recipe justification and concentration ranges.
- Measurement of acoustic/thermal properties: see Menikou & Damianou 2017 and Dabbagh et al. 2014.
- Antenna/generator specifications: attempt to obtain the official SABERWAVE ECO‑200G datasheet from the vendor or distributor (none found yet).

