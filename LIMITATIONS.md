# SmartDaaS — Known Limitations and Future Validation Plan

*Last corrected: 2026-07-16.*

This document provides an honest account of the current limitations of both the SmartDaaS v1 production model (§2, §9) and the v2 architecture prototype (§1, §3–§7), and the planned steps to address each.

**§2 and §9 were substantially rewritten in July 2026** after a data leakage error was found in the v1 training pipeline and the PHIA "external validation" claim was found to be unsupportable. The superseded text is retained in §9 rather than deleted.

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

## 2. The PHIA work is not validation, proxy or otherwise

Earlier versions of this document described nine Population-based HIV Impact Assessment (PHIA) surveys across six countries as external — or "proxy external" — validation of SmartDaaS v1, reporting AUC 0.769. **That claim is withdrawn.**

**PHIA cannot supply the model's inputs.** `constants.py` records it directly: *"PHIA supports 8/15 features only."* The mapping document is more specific — PHIA has no WHO clinical stage, no weight, no height, no BMI, and being cross-sectional, only one CD4 measurement, so `CD4_improvement`, `weight_change` and `stage_worsened` cannot exist. **A 15-feature model cannot be applied to data that lacks seven of its fifteen inputs.** Whatever produced 0.769, it was not this model being tested.

The outcome differs too: self-reported missed ARV doses, not `ArvAdherenceLatestLevel = Poor` recorded by a clinician.

The same applies to DHS, and more so — the DHS mapping document's own §5 lists CD4 count, WHO staging, days-to-ART, interruption history and side effects as unavailable in **any** DHS survey. DHS supports roughly 4 of 15 features and has no adherence outcome at all; the outcome nominated in that document (`hiv03`) is HIV serostatus, a different question entirely.

**What the PHIA and DHS work actually is:** the design for a *separate* model — population-level HIV cascade risk from survey data, using social determinants (food insecurity, stigma, mobility, distance to facility, wealth quintile) that clinical records do not capture. That is a legitimate and novel study, and it addresses the sociocultural gap Kwarah et al. explicitly identified and that zero of 12 reviewed models included. It is a sibling study, not a validation of this model, and it will be reported as one.

**SmartDaaS v1 has no external validation.** That is the honest position, and it is what CNICS and IeDEA West Africa exist to change.

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

**Status caveat:** the APIN and AMPATH pilots referenced throughout this table are prospective. No signed data use agreement is in place for either at the time of writing, and no timeline should be inferred. The routes currently in motion are CNICS (feasibility request submitted July 2026) and IeDEA West Africa (>50,000 adult ART initiators across eight countries including Nigeria; concept sheet process).

| Limitation | Planned resolution | Timeline |
|---|---|---|
| Synthetic data | Replace with APIN/AMPATH longitudinal EMR data | Pilot start |
| No external validation (PHIA cannot serve) | External validation on programme EMR with an equivalent outcome definition — via CNICS, IeDEA West Africa, or a partner DUA | Not yet scheduled |
| Transformer masking | Explicit attention_mask from sequence lengths | Before pilot data processing |
| Facility generalisation | Leave-one-facility-out validation | Pilot Phase 1 |
| Causal validity | Pre-registered uplift analysis with real treatment records | Pilot Phase 2 |
| Drift thresholds | Empirical recalibration against pilot data | Pilot Phase 1 |
| Multi-task windows | Formalised temporal outcome definitions | Before pilot |

---

## 9. What SmartDaaS v1 has actually established

**Corrected 2026-07-16.** The previous version of this section is reproduced at the end, because the corrections are the point.

The v1 model (Random Forest, 15 features, same-encounter prediction) has:

- Been trained on **23,144 adults** from a public HIV programme dataset of **undocumented provenance**. Internal evidence indicates Nigerian records; the deposit itself states no country, custodian, sampling frame or methodology. Of 27,288 raw records, 1,297 with a blank adherence outcome were previously coded as *adherent* and are now excluded; 2,270 patients under 18 are out of scope.
- Achieved **AUC 0.801 (10-fold CV, natural 3.67% prevalence)** and **AUC 0.806 (95% CI 0.774–0.837) on temporal hold-out** — trained on ART initiations up to September 2016, tested on 6,942 later initiators. The two agree (+0.006), which is the evidence that the signal generalises.
- **No external validation.** None. See §2.
- **No valid PROBAST assessment.** `SmartDaaS_PROBAST_Positioning.pdf` is withdrawn. Its items 2.1 and 3.2 state that no predictor is drawn from the same encounter as the outcome — seven of fifteen are. Item 2.4 states SMOTE was applied to training folds only; it was applied to the full dataset before splitting. Item 4.1 describes a hold-out at natural prevalence (3.4%); that hold-out was 50% positive. It also uses a custom item numbering that does not correspond to PROBAST 2019, and an overall rating ("predominantly low risk") that is not a valid PROBAST judgement.
- **No manuscripts under review.** The *Scientific Reports* submission was withdrawn in July 2026 after the leakage error was identified. The *BMJ Open* submission (not BMJ Global Health, as previously stated here) was rejected in July 2026 because the data's provenance, sampling and ownership were unclear — an objection now accepted as correct.

### The error, plainly

`01_data_preprocessing.py` applied SMOTE to the entire dataset before any train/test split. `02_model_training_cv.py` then cross-validated and held out from the resampled data. The reported "20% hold-out of 10,540 patients at natural class distribution" was 20% of 52,696 SMOTE-balanced rows at 50% prevalence, containing synthetic minority cases interpolated from the training set. Every metric downstream of that — CV AUC 0.9753, hold-out 0.973, sensitivity 87.3%, specificity 95.7%, Brier 0.079, ablation 0.963 — is invalid.

The "expected degradation" from CV to temporal AUC (0.203) was the leak. It is now +0.006.

Removing SMOTE entirely, excluding blank outcomes and paediatric patients, and correcting mixed CD4 units **improved** the model: 0.772 → 0.806.

### Superseded text (July 2026)

> - Been trained on 27,288 real-world Nigerian HIV programme records
> - Achieved AUC 0.9753 on cross-validation and AUC 0.772 on temporal holdout validation
> - Been externally validated against nine PHIA surveys across six countries
> - Completed PROBAST risk-of-bias assessment (predominantly low risk across all four domains)
> - Two manuscripts currently under peer review (Scientific Reports, BMJ Global Health)

---

The v2 architecture prototype extends v1 with longitudinal, survival-aware, drift-monitored, facility-contextual, and uplift-ready modules — all at prototype stage on synthetic data, pending real pilot data. See §1.

---

*Prepared by: Lakshmi Kalyani Chinthala | SmartDaaS LLC | lkchinthala@smartdaas.org*  
*GitHub: github.com/Kchinthala15/smartdaas-hiv-validation*
