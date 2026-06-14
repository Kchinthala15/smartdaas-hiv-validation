# SmartDaaS v2 — Known Limitations and Future Validation Plan

This document provides an honest account of the current limitations of the SmartDaaS v2 architecture prototype, and the planned steps to address each limitation through the APIN and AMPATH pilot studies.

---

## 1. Synthetic data limitations

All v2 architecture modules (survival model, transformer encoder, multi-task learning, drift detection, facility embeddings, causal/uplift modeling) were developed and tested on **synthetically generated data only**.

The synthetic data generator produces realistic HIV patient trajectories calibrated to published distributions from the Nigerian Quality of Care dataset, but it is **architecture-grade synthetic data**, not epidemiologically validated simulation. Specifically:

- Event timing distributions are simplified exponential approximations
- Interruption patterns do not capture seasonal or contextual variation
- Facility heterogeneity is simulated with random draws, not real programme structure
- Adherence trajectories do not capture within-patient correlation over time
- Viral suppression dynamics are approximated, not mechanistically modelled

**All performance metrics (AUC, C-index, IBS, ITE correlation) from v2 modules reflect synthetic testing environments and must not be used as evidence of clinical performance.**

---

## 2. PHIA proxy validation limitations

The SmartDaaS v1 external validation uses nine Population-based HIV Impact Assessment (PHIA) surveys spanning six countries as a proxy for multi-country external validation.

PHIA surveys are **cross-sectional population surveys**, not longitudinal programme EMR data. Key limitations:

- Outcome definition (missed ARV doses as proxy for poor adherence) differs from the training outcome (ArvAdherenceLatestLevel = Poor in programme EMR)
- Variable definitions were harmonised across surveys but not perfectly aligned with training data
- Temporal label equivalence cannot be guaranteed across survey rounds
- Surveys capture population-level HIV-positive adults, not necessarily patients currently enrolled in ART programmes

This proxy validation provides evidence of cross-country generalisability of the underlying risk signal but is not equivalent to full external validation on longitudinal programme EMR data with identical outcome definitions.

---

## 3. Causal / uplift modeling limitations

The causal uplift module estimates individual treatment effects (CATE) using observational data simulation. Key limitations:

- Treatment assignment in the simulation is non-random (high-risk patients more likely treated), creating confounding that cannot be fully removed
- The Causal Forest DML model performed poorly (negative ITE correlation) on synthetic data with small sample sizes — performance will improve substantially with real programme data
- Benefit is defined as -CATE (sign corrected), assuming treatment reduces interruption risk; this assumption requires validation in real deployment
- Uplift thresholds for tier assignment (URGENT / MONITOR / OPPORTUNITY / STANDARD CARE) are operational defaults pending clinical calibration with APIN field teams

---

## 4. Facility embedding generalisation limitations

The facility embedding module learns a 16-dimensional representation per facility from patient outcome patterns. Key limitations:

- With 40 synthetic facilities, embeddings may partially memorise facility identity rather than learning generalisable structural patterns
- Validation on entirely unseen facilities (leave-one-facility-out) has not yet been performed — required before any deployment claim
- Facility embedding overfitting risk increases with small facility counts; APIN pilot should include held-out facility validation
- Transfer learning between facilities in different countries is theoretically motivated but empirically unvalidated

---

## 5. Transformer attention masking limitations

The current transformer encoder derives padding masks from all-zero rows in the input sequence. This approach works correctly for synthetic data but carries risk for real EMR data:

- Genuine clinical visits with zero-valued measurements (e.g., zero CD4 count recorded as missing/zero) may be incorrectly masked as padding
- True missingness and padding are not explicitly separated in the current implementation
- For real APIN data, explicit attention masks should be passed based on actual sequence lengths, not derived from feature values

This is the most important technical fix to implement before processing real pilot data.

---

## 6. Drift detection threshold limitations

The drift detection thresholds (PSI > 0.2, KS p < 0.05, JS divergence > 0.1, AUC drop > 5pp) are standard operational defaults from industry practice (PSI from banking/credit risk monitoring, KS from statistical process control). They have not been:

- Clinically calibrated for HIV retention prediction specifically
- Empirically validated against known distribution shifts in sub-Saharan African HIV programmes
- Tested against seasonal, programmatic, or guideline-change-driven shifts in real data

These thresholds should be treated as starting points and recalibrated during the APIN pilot based on observed programme dynamics.

---

## 7. Multi-task learning target leakage risk

The four prediction targets (treatment interruption, viral failure, poor adherence, high missed visit rate) are derived from the same patient records and are correlated by design. Specific concerns:

- Adherence and interruption targets may share information from overlapping time windows
- Viral failure and interruption are causally linked — predicting one may implicitly predict the other
- Strict temporal outcome windows (predicting outcome at time T+N from features at time T) have not been implemented in the current prototype

Temporal target definitions should be formalised and pre-registered before the APIN pilot.

---

## 8. Future validation plan

| Limitation | Planned resolution | Timeline |
|---|---|---|
| Synthetic data | Replace with APIN/AMPATH longitudinal EMR data | Pilot start |
| PHIA proxy validation | Full external validation on programme EMR with identical outcome definitions | Pilot Phase 1 |
| Transformer masking | Explicit attention_mask from sequence lengths | Before pilot data processing |
| Facility generalisation | Leave-one-facility-out validation | Pilot Phase 1 |
| Causal validity | Pre-registered uplift analysis with real treatment records | Pilot Phase 2 |
| Drift thresholds | Empirical recalibration against pilot data | Pilot Phase 1 |
| Multi-task windows | Formalised temporal outcome definitions | Before pilot |

---

## 9. What SmartDaaS v1 has established (separate from v2 prototype)

The v1 model (Random Forest, 15 features, snapshot prediction) has:

- Been trained on 27,288 real-world Nigerian HIV programme records
- Achieved AUC 0.9753 on cross-validation and AUC 0.772 on temporal holdout validation
- Been externally validated against nine PHIA surveys across six countries
- Completed PROBAST risk-of-bias assessment (predominantly low risk across all four domains)
- Been reported in full per the TRIPOD prediction model reporting checklist
- Two manuscripts currently under peer review (Scientific Reports, BMJ Global Health)

The v2 architecture prototype extends v1 with longitudinal, survival-aware, drift-monitored, facility-contextual, and uplift-ready modules — all currently at prototype stage pending real pilot data.

---

*Prepared by: Lakshmi Kalyani Chinthala | SmartDaaS LLC | lkchinthala@smartdaas.org*  
*GitHub: github.com/Kchinthala15/smartdaas-hiv-validation*
