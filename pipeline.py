"""
SmartDaaS v1.0 — Data Pipeline
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

All data ingestion, column mapping, PHIA pre-processing,
validation metadata, tier detection, data quality screening,
IeDEA benchmarks, and feature engineering functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

from constants import (
    FEATURES, COLUMN_ALIASES, FEAT_LABELS, FEAT_DESCRIPTIONS, FEAT_RANGES,
    TIER_CORE_REQUIRED, TIER_STANDARD_QUALIFYING, TIER_STANDARD_MIN,
    TIER_ENHANCED_QUALIFYING, TIER_ENHANCED_MIN, ART_INFERENCE_TRIGGERS_LOWER,
    FEATURE_VALID_RANGES, HIGH_IMPORTANCE_FEATURES,
)


def normalize_columns(df):
    """
    Attempt to map uploaded column names to expected feature names.
    Returns (df_mapped, list_of_missing_cols, list_of_mappings_applied).
    Case-insensitive matching + alias lookup.
    """
    # Step 1: exact match first
    col_map = {}
    for col in df.columns:
        if col in FEATURES:
            col_map[col] = col

    # Step 2: case-insensitive + alias lookup for remaining
    needed = [f for f in FEATURES if f not in col_map.values()]
    for feat in needed:
        for col in df.columns:
            col_lower = col.lower().strip().replace(' ', '_')
            # Direct case-insensitive
            if col_lower == feat.lower():
                col_map[col] = feat
                break
            # Alias lookup
            if col_lower in COLUMN_ALIASES and COLUMN_ALIASES[col_lower] == feat:
                col_map[col] = feat
                break

    # Build renamed df
    rename_dict = {k: v for k, v in col_map.items() if k != v}
    df_mapped = df.rename(columns=rename_dict)

    mappings_applied = [(k, v) for k, v in rename_dict.items()]

    # Sex recode: handle M/F, Male/Female text strings -> binary 0/1
    # Partners commonly export sex as text - auto-recode so model gets numeric
    if 'sex_female' in df_mapped.columns:
        col = df_mapped['sex_female']
        numeric_attempt = pd.to_numeric(col, errors='coerce')
        # If conversion fails for most values, it's a text column
        if numeric_attempt.isna().sum() > len(col) * 0.5:
            col_str = col.astype(str).str.strip().str.lower()
            df_mapped['sex_female'] = col_str.map({
                'f': 1.0, 'female': 1.0, 'woman': 1.0, 'w': 1.0, '1': 1.0,
                'm': 0.0, 'male': 0.0, 'man': 0.0, '0': 0.0,
            })
            mappings_applied.append(('sex (M/F text)', 'sex_female (0=Male, 1=Female)'))

    missing = [f for f in FEATURES if f not in df_mapped.columns]

    return df_mapped, missing, mappings_applied


# ─────────────────────────────────────────────────────────────
# PHIA / POPULATION SURVEY PRE-PROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_phia_compatible(df):
    """
    Pre-processing pipeline for PHIA and compatible population-survey datasets.
    Must be called BEFORE normalize_columns().

    Handles transformations that require logic beyond simple column renaming:
      1. sex_female recode from PHIA gender coding (1=Male, 2=Female)
      2. had_interruption proxy from arvsmissdays (missed doses > 0)
      3. opp_infection composite from TB + STI proxy variables
      4. days_to_ART approximation from month/year fields
      5. Explicit flagging of all derived variables

    Returns (df_processed, derivation_log) where derivation_log is a list
    of strings describing every transformation applied, for transparency.

    VALIDATION CAVEATS preserved in derivation_log:
      - cd4count = most-recent survey CD4, not ART-start baseline
      - tbdiagn = self-reported TB, not lab-confirmed
      - days_to_ART = low-confidence derived feature (~±15 day error)
      - arvsmissdays proxy ≠ clinical treatment interruption
      - PHIA supports core signal validation only (8/15 SmartDaaS features)
    """
    import numpy as np
    df = df.copy()
    log = []
    cols = {c.lower().strip(): c for c in df.columns}

    # ── 1. sex_female recode ──────────────────────────────────────────────────
    # PHIA gender: 1=Male, 2=Female → SmartDaaS sex_female: 0=Male, 1=Female
    if 'gender' in cols and 'sex_female' not in cols:
        gc = cols['gender']
        gender_num = pd.to_numeric(df[gc], errors='coerce')
        df['sex_female'] = (gender_num == 2).astype(float)
        df.loc[gender_num.isna(), 'sex_female'] = np.nan
        log.append("sex_female: recoded from gender (PHIA: 1=Male→0, 2=Female→1)")

    # ── 2. had_interruption proxy from arvsmissdays ───────────────────────────
    # Use only if arvinterr (direct label) is absent or all-null.
    # arvinterr is handled via COLUMN_ALIASES rename; check post-rename name.
    has_direct = 'arvinterr' in cols
    has_proxy  = 'arvsmissdays' in cols
    if has_proxy and not has_direct and 'had_interruption' not in cols:
        mc = cols['arvsmissdays']
        miss = pd.to_numeric(df[mc], errors='coerce')
        df['had_interruption'] = (miss > 0).astype(float)
        df.loc[miss.isna(), 'had_interruption'] = np.nan
        df['_derived_interruption_proxy'] = 1  # flag column
        log.append(
            "had_interruption: PROXY derived from arvsmissdays>0. "
            "CAVEAT: self-reported missed doses, top-coded at 4. "
            "Not equivalent to clinical treatment interruption. "
            "Direct label (arvinterr) available in PHIA 2020+ waves only."
        )
    elif has_direct:
        log.append(
            "had_interruption: direct label (arvinterr) present — "
            "alias mapping will handle rename. No proxy needed."
        )

    # ── 3. opp_infection composite ────────────────────────────────────────────
    # Composite from TB diagnosis + active syphilis + STI diagnosis.
    # None of these individually equals clinical OI — composite is a proxy.
    if 'opp_infection' not in cols:
        sources_used = []
        oi = pd.Series(False, index=df.index, dtype=bool)
        any_source = False

        for src_key, src_label in [
            ('tbdiagn', 'tbdiagn==1'),
            ('activesyphilis', 'activesyphilis==1'),
            ('stddiag', 'stddiag==1'),
        ]:
            if src_key in cols:
                val = pd.to_numeric(df[cols[src_key]], errors='coerce')
                oi = oi | (val == 1).fillna(False)
                sources_used.append(src_label)
                any_source = True

        if any_source:
            df['opp_infection'] = oi.astype(float)
            df['_derived_opp_infection_composite'] = 1  # flag column
            log.append(
                f"opp_infection: COMPOSITE derived from [{', '.join(sources_used)}]. "
                "CAVEAT: narrower than clinical OI definition. "
                "tbdiagn = self-reported TB (not lab-confirmed). "
                "Composite proxy only."
            )

    # ── 4. days_to_ART approximation ─────────────────────────────────────────
    # Derived from first ART month/year and first HIV+ test month/year.
    # Month-year resolution only → ±15 day approximation error.
    # LOW-CONFIDENCE DERIVED FEATURE — flag explicitly.
    art_m = 'arvftm' in cols
    art_y = 'arvfty' in cols
    dx_m  = 'hivtfposm' in cols
    dx_y  = 'hivtfposy' in cols

    if art_m and art_y and dx_m and dx_y and 'days_to_ART' not in cols:
        arv_months = (
            pd.to_numeric(df[cols['arvfty']], errors='coerce') * 12 +
            pd.to_numeric(df[cols['arvftm']], errors='coerce')
        )
        dx_months = (
            pd.to_numeric(df[cols['hivtfposy']], errors='coerce') * 12 +
            pd.to_numeric(df[cols['hivtfposm']], errors='coerce')
        )
        days = ((arv_months - dx_months) * 30.44).round(0)
        df['days_to_ART'] = days.clip(0, 3650)
        df['_derived_days_to_art'] = 1  # flag column
        log.append(
            "days_to_ART: LOW-CONFIDENCE DERIVED from arvftm/fty and hivtfposm/y. "
            "Month-year resolution only — mid-month (day 15) assumed → ~±15 day error. "
            "Do not treat as equivalent to EMR-recorded ART initiation date. "
            "Rows flagged with _derived_days_to_art==1."
        )

    # ── 5. Persistent validation caveats ─────────────────────────────────────
    # These are appended to the log regardless of what was derived,
    # so callers always see the full caveat set.
    log.append(
        "PHIA VALIDATION SCOPE: This dataset supports validation of the core "
        "SmartDaaS predictive signal (up to 8 of 15 features). "
        "7 features are structurally absent from PHIA (CD4_improvement, "
        "stage_start_num, WeightAtStart, weight_change, BMI_start, "
        "side_effects, stage_worsened). "
        "Full 15-feature model validation requires longitudinal programme data."
    )
    if 'cd4count' in cols:
        log.append(
            "CD4 CAVEAT: cd4count in PHIA = most-recent survey-measured CD4, "
            "NOT CD4 at ART initiation. Maps to MostRecentCd4Count only. "
            "Standardised difference vs training cohort baseline CD4 is ~0.72 SD."
        )

    return df, log


# ─────────────────────────────────────────────────────────────
# VALIDATION METADATA LAYER
# ─────────────────────────────────────────────────────────────

def build_validation_metadata(df_raw, df_mapped, mappings_applied,
                               missing_features, derivation_log=None,
                               dq_results=None, tier=None):
    """
    Build a structured validation metadata object that tracks every
    inference, proxy, imputation, and derivation applied during processing.

    Returns a dict with the following structure:
    {
        "source_type": str,          # 'original' | 'phia_compatible' | 'mixed'
        "features_original": int,    # features present directly in upload
        "features_derived": int,     # features derived/proxied during processing
        "features_imputed": int,     # features filled with imputed defaults
        "features_missing": int,     # features structurally absent
        "derived_fields": [          # detail for each derived/proxied field
            {
                "feature": str,
                "method": str,       # 'alias_rename' | 'proxy' | 'composite' |
                                     # 'derived_approximate' | 'imputed_default'
                "source_fields": [],
                "confidence": str,   # 'high' | 'moderate' | 'low'
                "caveat": str
            }
        ],
        "warnings": [],              # list of caveat strings for UI display
        "phia_scope_note": str,      # canonical validation scope statement
        "audit_trail": []            # ordered log of all transformations
    }

    Designed for:
      - UI display in the platform (Data Quality / Model Transparency pages)
      - Pilot partner due diligence review
      - Grant/funder audit trail
      - Journal supplement material
    """
    import numpy as np

    meta = {
        "source_type": "original",
        "features_original": 0,
        "features_derived": 0,
        "features_imputed": 0,
        "features_missing": len(missing_features) if missing_features else 0,
        "derived_fields": [],
        "warnings": [],
        "phia_scope_note": "",
        "audit_trail": derivation_log or [],
    }

    cols_raw   = {c.lower().strip() for c in df_raw.columns}
    cols_mapped = set(df_mapped.columns)

    # ── Classify each model feature ──────────────────────────────────────────
    phia_proxy_flags = {
        '_derived_interruption_proxy',
        '_derived_opp_infection_composite',
        '_derived_days_to_art',
    }
    is_phia = any(f in df_mapped.columns for f in phia_proxy_flags)

    # Rename map: original col → model feature
    rename_map = {orig: mapped for orig, mapped in (mappings_applied or [])}

    for feat in FEATURES:
        if feat not in cols_mapped:
            # Structurally missing — already counted above
            continue

        # Check if it arrived via alias rename
        arrived_via_alias = feat in rename_map.values()

        # Check proxy flags
        if feat == 'had_interruption' and '_derived_interruption_proxy' in cols_mapped:
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "had_interruption",
                "method": "proxy",
                "source_fields": ["arvsmissdays"],
                "confidence": "moderate",
                "caveat": (
                    "Proxy derived from arvsmissdays>0. Self-reported missed doses "
                    "(top-coded at 4). Not equivalent to clinically documented "
                    "treatment interruption. Direct label (arvinterr) available "
                    "in PHIA 2020+ waves only."
                )
            })
            meta["warnings"].append(
                "had_interruption: PROXY (arvsmissdays>0) — not equivalent to "
                "clinical interruption record."
            )

        elif feat == 'opp_infection' and '_derived_opp_infection_composite' in cols_mapped:
            sources = []
            for src in ['tbdiagn', 'activesyphilis', 'stddiag']:
                if src in cols_raw:
                    sources.append(src)
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "opp_infection",
                "method": "composite",
                "source_fields": sources,
                "confidence": "moderate",
                "caveat": (
                    f"Composite from [{', '.join(sources)}]. Narrower than clinical OI "
                    "definition. tbdiagn = self-reported TB (not lab-confirmed)."
                )
            })
            meta["warnings"].append(
                "opp_infection: COMPOSITE proxy (TB + STI variables). "
                "Narrower than clinical OI. tbdiagn is self-reported."
            )

        elif feat == 'days_to_ART' and '_derived_days_to_art' in cols_mapped:
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "days_to_ART",
                "method": "derived_approximate",
                "source_fields": ["arvftm", "arvfty", "hivtfposm", "hivtfposy"],
                "confidence": "low",
                "caveat": (
                    "LOW-CONFIDENCE DERIVED. Month-year resolution only — "
                    "mid-month (day 15) assumed, introducing ~±15 day error. "
                    "Do not treat as equivalent to EMR-recorded ART initiation date."
                )
            })
            meta["warnings"].append(
                "days_to_ART: LOW-CONFIDENCE DERIVED from month/year fields "
                "(±15 day approximation). Not equivalent to EMR ART initiation date."
            )

        elif feat == 'tb_positive' and arrived_via_alias:
            # Arrived via tbdiagn → tb_positive alias
            src = next((o for o, m in rename_map.items() if m == 'tb_positive'), 'tbdiagn')
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "tb_positive",
                "method": "alias_rename",
                "source_fields": [src],
                "confidence": "moderate",
                "caveat": (
                    f"Mapped from '{src}'. "
                    "CAVEAT: self-reported TB diagnosis ('ever told by health worker "
                    "you had TB'). Not lab-confirmed. Different sensitivity/specificity "
                    "from clinical TB documentation."
                )
            })
            meta["warnings"].append(
                f"tb_positive: mapped from '{src}' — SELF-REPORTED TB diagnosis, "
                "not lab-confirmed."
            )

        elif feat == 'MostRecentCd4Count' and arrived_via_alias:
            src = next((o for o, m in rename_map.items() if m == 'MostRecentCd4Count'), None)
            if src == 'cd4count':
                meta["features_derived"] += 1
                meta["derived_fields"].append({
                    "feature": "MostRecentCd4Count",
                    "method": "alias_rename",
                    "source_fields": ["cd4count"],
                    "confidence": "high",
                    "caveat": (
                        "Mapped from cd4count (PHIA biomarker). "
                        "CAVEAT: PHIA cd4count = most-recent survey-measured CD4, "
                        "NOT CD4 at ART initiation. Standardised difference vs "
                        "training cohort baseline CD4 is ~0.72 SD. "
                        "Do not populate Cd4AtStart with this value."
                    )
                })
                meta["warnings"].append(
                    "MostRecentCd4Count: from PHIA cd4count — "
                    "most-recent survey CD4, NOT ART-start baseline. "
                    "Std diff vs training cohort: ~0.72 SD."
                )
            elif src == 'cd4cat':
                meta["features_derived"] += 1
                meta["derived_fields"].append({
                    "feature": "MostRecentCd4Count",
                    "method": "derived_approximate",
                    "source_fields": ["cd4cat"],
                    "confidence": "low",
                    "caveat": (
                        "Mapped from cd4cat (ordinal category) using midpoint "
                        "approximation: <200→100, 200-349→275, 350-499→425, 500+→650. "
                        "FALLBACK only. Treat as low-confidence CD4 estimate."
                    )
                })
                meta["warnings"].append(
                    "MostRecentCd4Count: from cd4cat ordinal category — "
                    "FALLBACK midpoint approximation. Low-confidence CD4 estimate."
                )
            else:
                meta["features_original"] += 1

        elif arrived_via_alias:
            # Standard alias rename — original data, just renamed
            meta["features_original"] += 1
            meta["derived_fields"].append({
                "feature": feat,
                "method": "alias_rename",
                "source_fields": [k for k, v in rename_map.items() if v == feat],
                "confidence": "high",
                "caveat": ""
            })

        elif dq_results and feat in dq_results.get('missing', {}):
            # Feature was present but partially/fully imputed by DQ step
            imp_info = dq_results['missing'][feat]
            meta["features_imputed"] += 1
            meta["derived_fields"].append({
                "feature": feat,
                "method": "imputed_default",
                "source_fields": [],
                "confidence": "low",
                "caveat": (
                    f"Imputed with {imp_info.get('impute_method','column median')} "
                    f"(value: {imp_info.get('impute_val','N/A')}) for "
                    f"{imp_info.get('pct_missing','?')}% missing values."
                )
            })

        else:
            meta["features_original"] += 1

    # ── Source type classification ────────────────────────────────────────────
    n_derived = meta["features_derived"]
    n_orig    = meta["features_original"]
    if n_derived == 0:
        meta["source_type"] = "original"
    elif n_orig == 0:
        meta["source_type"] = "phia_compatible"
    else:
        meta["source_type"] = "mixed"

    # ── PHIA scope note ───────────────────────────────────────────────────────
    n_structurally_absent = sum(
        1 for f in ['CD4_improvement','stage_start_num','WeightAtStart',
                    'weight_change','BMI_start','side_effects','stage_worsened']
        if f in (missing_features or [])
    )
    if n_structurally_absent >= 4 or is_phia:
        meta["phia_scope_note"] = (
            "SmartDaaS demonstrated external consistency of its core predictive "
            "signal across independent PHIA populations, while full model validation "
            "will require longitudinal programme datasets with complete feature "
            "availability and target outcomes. "
            f"{15 - meta['features_missing']} of 15 SmartDaaS features are available "
            f"in this upload ({meta['features_missing']} structurally absent)."
        )

    return meta


def render_validation_metadata(meta):
    """
    Render the validation metadata dict as a structured Streamlit UI component.
    Call after build_validation_metadata(), before or after DQ report.
    Designed for the Data Quality / Patient Risk pages.
    """
    if not meta:
        return

    has_derived   = meta["features_derived"] > 0
    has_imputed   = meta["features_imputed"] > 0
    has_warnings  = len(meta["warnings"]) > 0
    has_scope     = bool(meta.get("phia_scope_note"))

    # Only render if there's something noteworthy to show
    if not (has_derived or has_imputed or has_warnings or has_scope):
        return

    with st.expander(
        f"\U0001f50d Validation Metadata — "
        f"{meta['features_original']} original · "
        f"{meta['features_derived']} derived/proxied · "
        f"{meta['features_imputed']} imputed · "
        f"{meta['features_missing']} absent",
        expanded=has_warnings
    ):
        # Summary row
        cols = st.columns(4)
        cols[0].metric("Original features",  meta["features_original"])
        cols[1].metric("Derived / proxied",   meta["features_derived"],
                       delta=None if meta["features_derived"]==0 else "review caveats",
                       delta_color="off")
        cols[2].metric("Imputed (DQ fill)",   meta["features_imputed"],
                       delta=None if meta["features_imputed"]==0 else "low confidence",
                       delta_color="off")
        cols[3].metric("Structurally absent", meta["features_missing"],
                       delta=None if meta["features_missing"]==0 else "null-filled",
                       delta_color="off")

        st.markdown("---")

        # Scope note
        if has_scope:
            st.info(f"\U0001f4cb **Validation scope:** {meta['phia_scope_note']}")

        # Derived fields table
        if meta["derived_fields"]:
            st.markdown("**Feature provenance detail:**")
            conf_icons = {"high": "\U00002705", "moderate": "\U000026a0", "low": "\U0001f7e1"}
            for item in meta["derived_fields"]:
                if not item.get("caveat"):
                    continue  # skip clean alias renames with no caveat
                icon = conf_icons.get(item["confidence"], "\u2139\ufe0f")
                src  = ", ".join(f"`{s}`" for s in item["source_fields"]) if item["source_fields"] else "—"
                st.markdown(
                    f"{icon} **`{item['feature']}`** "
                    f"— method: *{item['method']}* "
                    f"— source: {src}  \n"
                    f"<small style='color:#8b949e'>{item['caveat']}</small>",
                    unsafe_allow_html=True
                )

        # Warnings
        if has_warnings:
            st.markdown("---")
            st.markdown("**Caveats for clinical interpretation:**")
            for w in meta["warnings"]:
                st.warning(w)

        # Audit trail
        if meta.get("audit_trail"):
            with st.expander("\U0001f4cb Full derivation audit trail", expanded=False):
                for i, entry in enumerate(meta["audit_trail"], 1):
                    st.markdown(f"**{i}.** {entry}")


def detect_art_status(df_original):
    cols_lower = {c.lower().strip().replace(' ', '_') for c in df_original.columns}
    explicit_names = {
        'art_status', 'on_art', 'receiving_art', 'art',
        'waspatientreceivingarv', 'was_patient_receiving_arv', 'arv_status'
    }
    if cols_lower & explicit_names:
        return True, False, ""
    matches = cols_lower & ART_INFERENCE_TRIGGERS_LOWER
    if matches:
        shown = sorted(matches)[:3]
        ellipsis = '...' if len(matches) > 3 else ''
        note = (
            f"ART status was inferred from ART-related clinical variables "
            f"({', '.join(shown)}{ellipsis}) because no explicit ART status "
            f"column was detected."
        )
        return False, True, note
    return False, False, (
        "No ART status column or ART-related variables detected. "
        "This upload cannot be confirmed as an ART patient cohort. "
        "Please add an 'art_status' column (1=on ART, 0=not on ART) or include "
        "ART clinical variables such as 'art_start_date', 'cd4_at_start', or 'days_to_art'."
    )


def detect_tier(df_mapped, art_confirmed, art_inferred):
    available = set(df_mapped.columns)
    if not art_confirmed and not art_inferred:
        return 'INSUFFICIENT', [], list(TIER_CORE_REQUIRED), [], [], [
            "Upload does not contain confirmed ART patient data. Risk scoring is not possible."
        ]
    missing_core = [f for f in TIER_CORE_REQUIRED if f not in available]
    if missing_core:
        return 'INSUFFICIENT', [], missing_core, [], [], [
            f"Missing required variables: {', '.join(missing_core)}. "
            "Age and sex are required for any analysis."
        ]
    standard_present = [f for f in TIER_STANDARD_QUALIFYING if f in available]
    enhanced_present = [f for f in TIER_ENHANCED_QUALIFYING if f in available]
    all_present = (
        [f for f in TIER_CORE_REQUIRED if f in available]
        + standard_present + enhanced_present
    )
    if len(standard_present) >= TIER_STANDARD_MIN and len(enhanced_present) >= TIER_ENHANCED_MIN:
        tier = 'ENHANCED'
    elif len(standard_present) >= TIER_STANDARD_MIN:
        tier = 'STANDARD'
    else:
        tier = 'CORE'
    return tier, all_present, [], standard_present, enhanced_present, []


def check_pediatric_patients(df_mapped):
    if 'Age' not in df_mapped.columns:
        return []
    try:
        ages = pd.to_numeric(df_mapped['Age'], errors='coerce')
        return list(df_mapped[ages < 15].index)
    except Exception:
        return []


def render_tier_report(tier, present, missing_core, standard_present,
                       enhanced_present, art_confirmed, art_inferred,
                       art_note, pediatric_indices, df_mapped):
    TIER_COLOURS = {
        'ENHANCED': '#21d4fd',
        'STANDARD': '#f0a500',
        'CORE': '#8b949e',
        'INSUFFICIENT': '#f85149'
    }
    TIER_LABELS = {
        'ENHANCED': 'Enhanced Tier — Full Analysis Available',
        'STANDARD': 'Standard Tier — Partial Feature Availability',
        'CORE': 'Core Tier — Cohort Characterisation Only',
        'INSUFFICIENT': 'Insufficient Data — Cannot Proceed'
    }
    TIER_CAPABILITIES = {
        'ENHANCED': [
            "Full 15-feature patient risk scores",
            "SHAP explainability per patient",
            "Full cohort intelligence dashboard",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Full executive report",
            "Intervention recommendations",
        ],
        'STANDARD': [
            "Risk estimates generated using partial feature availability — prediction confidence and stability may vary depending on which clinical variables are present",
            "Partial SHAP explainability",
            "Cohort intelligence dashboard",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Standard executive report",
            "Intervention recommendations (reduced specificity)",
        ],
        'CORE': [
            "Patient risk scores — NOT available (insufficient clinical variables)",
            "SHAP explainability — NOT available",
            "Basic cohort characterisation (age, sex distribution)",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Limited executive report (population summary only)",
            "To unlock risk scoring: add CD4, WHO stage, TB status, and days to ART",
        ],
        'INSUFFICIENT': [
            "No analysis available",
            "Please review missing variables and re-upload",
        ]
    }
    colour = TIER_COLOURS.get(tier, '#8b949e')
    label = TIER_LABELS.get(tier, tier)
    caps = TIER_CAPABILITIES.get(tier, [])
    tier_icons = {
        'ENHANCED': 'success', 'STANDARD': 'warning',
        'CORE': 'info', 'INSUFFICIENT': 'error'
    }
    getattr(st, tier_icons.get(tier, 'info'))(
        f"**Data Tier Detected: {label}**"
    )
    if art_inferred and art_note:
        st.warning(f"ℹ️ {art_note}")
    elif not art_confirmed and not art_inferred and art_note:
        st.error(f"❌ {art_note}")
    st.markdown("**What this upload enables:**")
    icon_map = {'ENHANCED': '✅', 'STANDARD': '⚡', 'CORE': '📊', 'INSUFFICIENT': '❌'}
    icon = icon_map.get(tier, '•')
    for cap in caps:
        if 'NOT available' in cap or 'not available' in cap or 'No analysis' in cap:
            st.markdown(f"❌ {cap}")
        elif 'unlock' in cap.lower() or 'To ' in cap:
            st.markdown(f"💡 {cap}")
        else:
            st.markdown(f"{icon} {cap}")
    if missing_core:
        st.error(f"**Missing required variables:** {', '.join(missing_core)}")
    if present:
        with st.expander("Variables detected in your upload", expanded=False):
            cols = st.columns(2)
            half = len(present) // 2
            with cols[0]:
                for f in present[:half]:
                    st.markdown(f"✅ {FEAT_LABELS.get(f, f)}")
            with cols[1]:
                for f in present[half:]:
                    st.markdown(f"✅ {FEAT_LABELS.get(f, f)}")
    missing_standard = [f for f in TIER_STANDARD_QUALIFYING if f not in standard_present]
    missing_enhanced = [f for f in TIER_ENHANCED_QUALIFYING if f not in enhanced_present]
    if tier in ('CORE', 'STANDARD') and (missing_standard or missing_enhanced):
        with st.expander("Variables that would upgrade your tier", expanded=False):
            if missing_standard:
                st.markdown("**To reach Standard tier, add:**")
                for f in missing_standard:
                    st.markdown(f"- {FEAT_LABELS.get(f, f)} (`{f}`)")
            if tier == 'STANDARD' and missing_enhanced:
                st.markdown("**To reach Enhanced tier, also add:**")
                for f in missing_enhanced:
                    st.markdown(f"- {FEAT_LABELS.get(f, f)} (`{f}`)")
    if pediatric_indices:
        n_ped = len(pediatric_indices)
        st.warning(
            f"**{n_ped} pediatric patient{'s' if n_ped > 1 else ''} detected** "
            f"(age < 15). This model was trained on patients aged 15 and above. "
            f"Risk scores for these patients are not validated for pediatric HIV care "
            f"and will be flagged individually. Clinical interpretation by a qualified "
            f"paediatric HIV clinician is required."
        )
    return tier != 'INSUFFICIENT'


# ─────────────────────────────────────────────────────────────
# DATA QUALITY SCREENING — Group 5
# ─────────────────────────────────────────────────────────────

# Valid ranges per feature — used for out-of-range detection
# Clinically important features — missing these hurts quality grade more
def run_data_quality_screening(df_mapped, available_features):
    """
    Run all data quality checks on the uploaded dataframe.
    Returns a structured results dict.
    """
    results = {
        'missing': {},
        'out_of_range': {},
        'duplicates': None,
        'constant_columns': [],
        'grade': None,
        'grade_reasons': [],
        'deductions': 0,
    }

    n_rows = len(df_mapped)
    if n_rows == 0:
        results['grade'] = 'D'
        results['grade_reasons'] = ['Upload contains no rows.']
        return results

    # ── 1. Missing value analysis ─────────────────────────
    total_missing_pct = 0
    high_importance_missing_pct = 0
    n_high_importance_checked = 0

    for feat in available_features:
        if feat not in df_mapped.columns:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce')
        n_missing = col.isnull().sum()
        pct_missing = n_missing / n_rows * 100
        if n_missing > 0:
            impute_val = col.median() if feat not in {
                'sex_female', 'had_interruption', 'opp_infection',
                'side_effects', 'tb_positive', 'stage_worsened'
            } else col.mode().iloc[0] if len(col.mode()) > 0 else 0
            impute_method = 'column median' if feat not in {
                'sex_female', 'had_interruption', 'opp_infection',
                'side_effects', 'tb_positive', 'stage_worsened'
            } else 'column mode'
            results['missing'][feat] = {
                'n_missing': int(n_missing),
                'pct_missing': round(pct_missing, 1),
                'impute_val': round(float(impute_val), 1) if pd.notna(impute_val) else 0,
                'impute_method': impute_method,
                'high_importance': feat in HIGH_IMPORTANCE_FEATURES,
            }
            total_missing_pct += pct_missing
            if feat in HIGH_IMPORTANCE_FEATURES:
                high_importance_missing_pct += pct_missing
                n_high_importance_checked += 1

    avg_missing = total_missing_pct / len(available_features) if available_features else 0
    avg_hi_missing = (high_importance_missing_pct / n_high_importance_checked
                      if n_high_importance_checked > 0 else 0)

    # ── 2. Out-of-range detection ─────────────────────────
    n_severe_range = 0
    for feat in available_features:
        if feat not in df_mapped.columns or feat not in FEATURE_VALID_RANGES:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce').dropna()
        lo, hi = FEATURE_VALID_RANGES[feat]
        out_mask = (col < lo) | (col > hi)
        n_out = int(out_mask.sum())
        if n_out > 0:
            pct_out = round(n_out / n_rows * 100, 1)
            out_vals = col[out_mask]
            results['out_of_range'][feat] = {
                'n_out': n_out,
                'pct_out': pct_out,
                'valid_range': f'{lo}–{hi}',
                'min_observed': round(float(out_vals.min()), 1),
                'max_observed': round(float(out_vals.max()), 1),
                'severity': 'severe' if pct_out > 5 else 'minor',
            }
            if pct_out > 5:
                n_severe_range += 1

    # ── 3. Duplicate patient ID detection ────────────────
    if 'patient_id' in df_mapped.columns:
        dup_count = int(df_mapped['patient_id'].duplicated().sum())
        if dup_count > 0:
            results['duplicates'] = {
                'n_duplicates': dup_count,
                'pct_duplicates': round(dup_count / n_rows * 100, 1),
            }

    # ── 4. Near-constant column detection ────────────────
    for feat in available_features:
        if feat not in df_mapped.columns:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce').dropna()
        if len(col) < 2:
            continue
        top_val_pct = col.value_counts(normalize=True).iloc[0] * 100
        if top_val_pct >= 95:
            results['constant_columns'].append({
                'feature': feat,
                'dominant_value': round(float(col.value_counts().index[0]), 1),
                'pct_dominant': round(top_val_pct, 1),
            })

    # ── 5. Grade calculation ──────────────────────────────
    deductions = 0
    reasons = []

    # Missing data deductions
    if avg_missing == 0:
        reasons.append('no missing values')
    elif avg_missing <= 5:
        deductions += 5
        reasons.append(f'{avg_missing:.1f}% average missingness (minor)')
    elif avg_missing <= 15:
        deductions += 15
        reasons.append(f'{avg_missing:.1f}% average missingness (moderate)')
    elif avg_missing <= 30:
        deductions += 25
        reasons.append(f'{avg_missing:.1f}% average missingness (high)')
    else:
        deductions += 40
        reasons.append(f'{avg_missing:.1f}% average missingness (severe)')

    # High-importance feature missingness
    if avg_hi_missing > 20:
        deductions += 15
        reasons.append(f'{avg_hi_missing:.1f}% missing in high-importance features')

    # Out-of-range deductions
    if n_severe_range == 0 and len(results['out_of_range']) == 0:
        reasons.append('no out-of-range values detected')
    elif n_severe_range == 0:
        deductions += 5
        reasons.append(f"{len(results['out_of_range'])} feature(s) with minor range violations")
    else:
        deductions += 15
        reasons.append(f'{n_severe_range} feature(s) with severe range violations (>5% of rows)')

    # Duplicate deductions
    if results['duplicates']:
        n_dup = results['duplicates']['n_duplicates']
        pct_dup = results['duplicates']['pct_duplicates']
        if pct_dup < 1:
            deductions += 5
            reasons.append(f'{n_dup} duplicate patient ID(s) detected (minor)')
        elif pct_dup < 5:
            deductions += 15
            reasons.append(f'{n_dup} duplicate patient IDs ({pct_dup}% of cohort)')
        else:
            deductions += 25
            reasons.append(f'{n_dup} duplicate patient IDs ({pct_dup}% of cohort — high)')
    else:
        reasons.append('no duplicate patient IDs')

    # Constant column deductions
    if results['constant_columns']:
        n_const = len(results['constant_columns'])
        deductions += n_const * 5
        feat_names = [FEAT_LABELS.get(c['feature'], c['feature'])
                      for c in results['constant_columns']]
        reasons.append(
            f"{n_const} near-constant column(s): {', '.join(feat_names)} "
            f"(possibly miscoded — check export)"
        )
    else:
        reasons.append('no near-constant columns detected')

    # Assign grade
    score = 100 - deductions
    if score >= 90:
        grade = 'A'
    elif score >= 75:
        grade = 'B'
    elif score >= 55:
        grade = 'C'
    else:
        grade = 'D'

    results['grade'] = grade
    results['grade_reasons'] = reasons
    results['deductions'] = deductions
    results['score'] = score

    return results


def render_data_quality_report(dq, n_rows):
    """
    Display the data quality screening report.
    """
    grade = dq['grade']
    score = dq.get('score', 100 - dq['deductions'])
    reasons = dq['grade_reasons']

    grade_colours = {'A': '#3fb950', 'B': '#21d4fd', 'C': '#f0a500', 'D': '#f85149'}
    grade_labels = {
        'A': 'Excellent — data appears well-formed',
        'B': 'Good — minor issues detected, review recommended',
        'C': 'Fair — moderate issues detected, review before use',
        'D': 'Poor — significant issues detected, review required',
    }
    colour = grade_colours.get(grade, '#8b949e')
    label = grade_labels.get(grade, grade)

    with st.expander(
        f"📋 Data Quality Screening — Grade {grade}: {label}",
        expanded=(grade in ('C', 'D'))
    ):
        # Grade header
        c_grade, c_detail = st.columns([1, 4])
        with c_grade:
            st.markdown(
                f'<div style="text-align:center;padding:12px;border:2px solid {colour};'
                f'border-radius:8px;">'
                f'<div style="color:{colour};font-size:2.5rem;font-weight:900">{grade}</div>'
                f'<div style="color:#8b949e;font-size:0.7rem">Screening Score: {score}/100</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c_detail:
            st.markdown(
                f'**Data quality screening score — not a clinical validity score.**\n\n'
                f'Grade {grade}: ' + '; '.join(reasons) + '.'
            )
            if dq['deductions'] > 0:
                st.caption(
                    f"Total deductions: {dq['deductions']} points across "
                    f"{len([r for r in reasons if 'no ' not in r])} issue(s)."
                )

        st.markdown("---")

        # Missing values table
        if dq['missing']:
            st.markdown("**Missing Values — imputation applied:**")
            miss_rows = []
            for feat, info in dq['missing'].items():
                miss_rows.append({
                    'Variable': FEAT_LABELS.get(feat, feat),
                    'Missing': f"{info['n_missing']:,} ({info['pct_missing']}%)",
                    'Imputed with': f"{info['impute_method']} ({info['impute_val']})",
                    'High importance': '⚠️ Yes' if info['high_importance'] else 'No',
                })
            st.dataframe(pd.DataFrame(miss_rows), use_container_width=True)
        else:
            st.success("✅ No missing values detected.")

        # Out-of-range table
        if dq['out_of_range']:
            st.markdown("**Out-of-Range Values — rows retained, flagged for review:**")
            range_rows = []
            for feat, info in dq['out_of_range'].items():
                range_rows.append({
                    'Variable': FEAT_LABELS.get(feat, feat),
                    'Valid Range': info['valid_range'],
                    'Rows Affected': f"{info['n_out']:,} ({info['pct_out']}%)",
                    'Observed Min/Max': f"{info['min_observed']} / {info['max_observed']}",
                    'Severity': '🔴 Severe' if info['severity'] == 'severe' else '🟡 Minor',
                })
            st.dataframe(pd.DataFrame(range_rows), use_container_width=True)
            st.caption(
                "Rows with out-of-range values are retained. "
                "Review these values with your data manager before operational use."
            )
        else:
            st.success("✅ No out-of-range values detected.")

        # Duplicates
        if dq['duplicates']:
            n_dup = dq['duplicates']['n_duplicates']
            pct_dup = dq['duplicates']['pct_duplicates']
            st.warning(
                f"⚠️ **{n_dup:,} duplicate patient ID(s)** detected ({pct_dup}% of cohort). "
                f"Duplicates have not been removed. Review with your data manager — "
                f"duplicates may indicate repeated records, data export errors, or "
                f"patients attending multiple facilities."
            )
        else:
            st.success("✅ No duplicate patient IDs detected.")

        # Near-constant columns
        if dq['constant_columns']:
            st.markdown("**Near-Constant Columns — possible miscoding:**")
            for c in dq['constant_columns']:
                fname = FEAT_LABELS.get(c['feature'], c['feature'])
                st.warning(
                    f"⚠️ **{fname}**: {c['pct_dominant']}% of values are "
                    f"{c['dominant_value']}. This column may have been miscoded "
                    f"or exported incorrectly. Verify in your EMR system."
                )
        else:
            st.success("✅ No near-constant columns detected.")

        st.caption(
            f"Data quality screening applied to {n_rows:,} patients across "
            f"{len(dq['missing']) + len([f for f in FEATURE_VALID_RANGES if f not in dq['out_of_range']])} "
            f"features. This screening identifies structural data issues only and does not "
            f"assess clinical validity or programme representativeness."
        )



# ─────────────────────────────────────────────────────────────
# IeDEA MUD REGIONAL AGGREGATE CONTEXTUAL BENCHMARKS
# Source: IeDEA Multi-Use Dataset (MUD) v1.0, 2025
# Data through 2022. CC BY-NC-SA 4.0
# These are aggregate contextual benchmarks — NOT patient-level
# external validation.
# ─────────────────────────────────────────────────────────────

IEDEA_MUD_SUMMARY = {
    'WA': {
        'name': 'West Africa',
        'countries': 'Benin, Burkina Faso, Côte d\'Ivoire, Ghana, Mali, Nigeria, Senegal, Togo',
        'artstart_n': 42369,
        'cd4_art_median': 181.0,
        'cd4_art_pct_below200': 43.6,
        'vl_supp_6mo_perc': 84.6,
        'vl_supp_12mo_perc': 84.5,
    },
    'EA': {
        'name': 'East Africa',
        'countries': 'Kenya, Uganda, Tanzania',
        'artstart_n': 229002,
        'cd4_art_median': 197.0,
        'cd4_art_pct_below200': 30.2,
        'vl_supp_6mo_perc': 89.7,
        'vl_supp_12mo_perc': 87.6,
    },
    'SA': {
        'name': 'Southern Africa',
        'countries': 'South Africa, Zambia, Malawi, Lesotho, Mozambique, Zimbabwe',
        'artstart_n': 921922,
        'cd4_art_median': 218.0,
        'cd4_art_pct_below200': 20.5,
        'vl_supp_6mo_perc': 91.0,
        'vl_supp_12mo_perc': 89.7,
    },
    'CA': {
        'name': 'Central Africa',
        'countries': 'Burundi, Cameroon, DRC, Rwanda',
        'artstart_n': 42459,
        'cd4_art_median': 241.0,
        'cd4_art_pct_below200': 24.4,
        'vl_supp_6mo_perc': 91.9,
        'vl_supp_12mo_perc': 88.0,
    },
}

IEDEA_MUD_SEX = {
    'WA': {'Male': 85.0, 'Female': 84.2},
    'EA': {'Male': 86.3, 'Female': 88.3},
    'SA': {'Male': 88.0, 'Female': 90.7},
    'CA': {'Male': 87.9, 'Female': 88.1},
}

IEDEA_MUD_TREND = {
    'WA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [78.3,78.8,82.4,81.8,80.6,84.5,83.2,84.3,86.3,
                         89.9,88.7,88.8,90.0,88.1,90.1,93.2,90.1],
        'cd4_median': [121,125,157,163,171,181.5,207,210.5,207,256,
                       257,309,256,253,257,277.5,293.5],
    },
    'EA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [81.4,76.8,77.7,72.8,62.0,50.2,73.2,84.1,86.8,
                         84.6,84.6,85.8,91.2,93.8,95.7,94.6,95.4],
        'cd4_median': [100,117,140,146.62,169,198,215,262,321,290,
                       326,329,329,324,312,292,289],
    },
    'SA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [88.0,88.6,88.1,87.7,86.4,87.4,87.8,89.0,90.8,
                         90.7,90.3,89.6,89.1,90.8,92.7,93.6,93.0],
        'cd4_median': [127,140,148,154,167,195,217,239,269,284,
                       301,334,325,325,331,319,307],
    },
    'CA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [90.5,90.6,82.9,91.3,85.6,88.8,90.2,90.8,91.1,
                         85.9,84.2,84.5,85.6,91.5,94.1,93.8,96.8],
        'cd4_median': [144,175,201,211,223,249.5,247,250,286,273,
                       322,272,307,287,326,332,336],
    },
}

IEDEA_MUD_CITATION = (
    "IeDEA (2025). Version 1.0. IeDEA Multi-Use Dataset (MUD). "
    "Retrieved from iedea.org. License: CC BY-NC-SA 4.0."
)

IEDEA_MUD_NOTE = (
    "IeDEA MUD data represent aggregate indicators from IeDEA-participating "
    "clinical sites. These are regional contextual benchmarks — not nationally "
    "representative estimates and not patient-level external validation. "
    "Data through 2022. Site composition varies by region and year."
)


def render_iedea_benchmarks(df_upload=None, selected_region=None):
    """
    Render IeDEA MUD regional aggregate contextual benchmarks.
    If df_upload provided, compares cohort metrics against selected region.
    """
    st.markdown('<p class="section-hdr">IeDEA MUD Regional Aggregate Contextual Benchmarks</p>',
                unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box">
    <strong>What this shows:</strong> Regional aggregate indicators from the IeDEA
    Multi-Use Dataset (MUD) v1.0 — covering {sum(v['artstart_n'] for v in IEDEA_MUD_SUMMARY.values()):,}
    patients across West, East, Southern, and Central Africa (data through 2022).
    These are <strong>contextual benchmarks only</strong> — not patient-level external validation
    of the SmartDaaS model. Data reflect IeDEA-participating sites and are not
    nationally representative.<br><br>
    <em>{IEDEA_MUD_CITATION}</em>
    </div>""", unsafe_allow_html=True)

    # ── Region selector ───────────────────────────────────
    region_options = {v['name']: k for k, v in IEDEA_MUD_SUMMARY.items()}
    region_options['All African Regions'] = 'ALL'

    default_region = 'All African Regions'
    if selected_region and selected_region in region_options.values():
        default_idx = list(region_options.values()).index(selected_region)
    else:
        default_idx = list(region_options.keys()).index('All African Regions')

    sel_name = st.selectbox(
        "Select region for comparison:",
        list(region_options.keys()),
        index=default_idx,
        key="iedea_region_sel"
    )
    sel_code = region_options[sel_name]

    # ── Regional summary cards ────────────────────────────
    if sel_code == 'ALL':
        regions_to_show = list(IEDEA_MUD_SUMMARY.keys())
    else:
        regions_to_show = [sel_code]

    cols = st.columns(len(regions_to_show))
    for i, reg in enumerate(regions_to_show):
        d = IEDEA_MUD_SUMMARY[reg]
        with cols[i]:
            st.markdown(f"""<div class="metric-box" style="text-align:center">
                <div style="color:#21d4fd;font-weight:700;font-size:0.95rem;margin-bottom:6px">
                    {d['name']}
                </div>
                <div style="color:#8b949e;font-size:0.7rem;margin-bottom:8px">
                    {d['countries'][:40]}{'...' if len(d['countries'])>40 else ''}
                </div>
                <div style="color:#e6edf3;font-size:1.1rem;font-weight:700">
                    {d['artstart_n']:,}
                </div>
                <div style="color:#8b949e;font-size:0.7rem">patients on ART</div>
                <hr style="border-color:#30363d;margin:8px 0">
                <div style="color:#3fb950;font-size:1rem;font-weight:700">
                    {d['vl_supp_12mo_perc']}%
                </div>
                <div style="color:#8b949e;font-size:0.7rem">VL suppression at 12mo</div>
                <div style="color:#e6edf3;font-size:1rem;margin-top:4px">
                    {d['cd4_art_median']:.0f}
                </div>
                <div style="color:#8b949e;font-size:0.7rem">Median CD4 at ART start</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison charts ─────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "VL Suppression Trend",
        "Sex-Disaggregated Outcomes",
        "CD4 at ART Start Trend"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        colours = {'WA': '#21d4fd', 'EA': '#3fb950', 'SA': '#f0a500', 'CA': '#f85149'}

        for reg in (regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())):
            td = IEDEA_MUD_TREND[reg]
            valid = [(y, v) for y, v in zip(td['years'], td['vl_supp_12mo']) if v is not None]
            if valid:
                ys, vs = zip(*valid)
                ax.plot(ys, vs, color=colours[reg], linewidth=2,
                        marker='o', markersize=3,
                        label=f"{IEDEA_MUD_SUMMARY[reg]['name']}")

        # If upload provided, overlay cohort VL suppression if available
        if df_upload is not None and 'vl_suppressed' in df_upload.columns:
            cohort_vl = pd.to_numeric(
                df_upload['vl_suppressed'], errors='coerce').mean() * 100
            ax.axhline(cohort_vl, color='#ffffff', linewidth=1.5,
                       linestyle='--',
                       label=f'Your cohort ({cohort_vl:.1f}%) — contextual reference only')

        ax.set_xlabel('Year of ART Initiation', color='#8b949e', fontsize=9)
        ax.set_ylabel('VL Suppression at 12 months (%)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: Viral Load Suppression at 12 Months After ART Start\n'
            '(Regional aggregate contextual benchmark — not external validation)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_ylim(50, 100)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    with tab2:
        regs = regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())
        male_vals = [IEDEA_MUD_SEX[r]['Male'] for r in regs]
        female_vals = [IEDEA_MUD_SEX[r]['Female'] for r in regs]
        reg_labels = [IEDEA_MUD_SUMMARY[r]['name'] for r in regs]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        x = range(len(regs))
        w = 0.35
        bars_m = ax.bar([i - w/2 for i in x], male_vals, width=w,
                        color='#21d4fd', label='Male', edgecolor='#0d1117')
        bars_f = ax.bar([i + w/2 for i in x], female_vals, width=w,
                        color='#f0a500', label='Female', edgecolor='#0d1117')

        for bar in bars_m:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}%', ha='center',
                    fontsize=8, color='#e6edf3')
        for bar in bars_f:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}%', ha='center',
                    fontsize=8, color='#e6edf3')

        ax.set_xticks(list(x))
        ax.set_xticklabels(reg_labels, fontsize=8, color='#cdd9e5')
        ax.set_ylabel('VL Suppression at 12 months (%)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: VL Suppression by Sex\n'
            '(Regional aggregate — all years)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_ylim(75, 96)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            "_Note: In the Nigerian discovery cohort, female sex was associated with "
            "lower odds of poor outcome. IeDEA MUD shows female advantage in East and "
            "Southern Africa; West Africa shows a smaller and reversed pattern. "
            "These aggregate patterns provide contextual plausibility only._"
        )
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        colours = {'WA': '#21d4fd', 'EA': '#3fb950', 'SA': '#f0a500', 'CA': '#f85149'}

        for reg in (regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())):
            td = IEDEA_MUD_TREND[reg]
            valid = [(y, c) for y, c in zip(td['years'], td['cd4_median']) if c is not None]
            if valid:
                ys, cs = zip(*valid)
                ax.plot(ys, cs, color=colours[reg], linewidth=2,
                        marker='o', markersize=3,
                        label=f"{IEDEA_MUD_SUMMARY[reg]['name']}")

        ax.set_xlabel('Year of ART Initiation', color='#8b949e', fontsize=9)
        ax.set_ylabel('Median CD4 at ART Start (cells/µL)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: Median CD4 Count at ART Initiation Over Time\n'
            '(Regional aggregate contextual benchmark)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    # ── Disclaimer ───────────────────────────────────────
    st.markdown(f"""<div class="warn-box" style="margin-top:12px">
    ⚠️ <strong>Important:</strong> {IEDEA_MUD_NOTE}
    </div>""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# LOCAL RECALIBRATION ENGINE — Stage 2
# Platt scaling calibration with full validation checks
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                              brier_score_loss, confusion_matrix)
from sklearn.utils import resample

# ── Constants ────────────────────────────────────────────────
# ── 1. Outcome column detection ───────────────────────────────

def detect_outcome_columns(df):
    """
    Scan dataframe for likely outcome columns.
    Returns list of (col_name, detection_reason, value_counts) tuples.
    Detects both numeric 0/1 columns and text-valued programme columns
    (e.g. LTFU/Active/Retained/Dead) that can be normalised to binary.
    """
    # Text values that indicate a programme outcome column
    OUTCOME_TEXT_SIGNALS = {
        'ltfu', 'lost to follow-up', 'lost to follow up', 'lost',
        'dead', 'death', 'died', 'retained', 'active', 'defaulted',
        'interrupted', 'failure', 'transferred out', 'stopped',
        'suppressed', 'on treatment', 'enrolled', 'dropped',
    }

    candidates = []
    for col in df.columns:
        col_lower = col.lower().strip().replace(' ', '_')
        reason = None

        # Name-based detection
        if col_lower in OUTCOME_NAME_HINTS:
            reason = 'column name matches known outcome identifier'

        # Value-pattern detection — binary 0/1
        if reason is None:
            try:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                unique = set(vals.unique())
                if unique <= {0, 1, 0.0, 1.0} and len(vals) > 0:
                    pct_pos = (vals == 1).mean()
                    if 0.01 <= pct_pos <= 0.60:
                        reason = (f'binary 0/1 column with '
                                  f'{pct_pos*100:.1f}% positive rate')
            except Exception:
                pass

        # Text-value detection — programme status strings
        if reason is None:
            try:
                text_vals = df[col].dropna().astype(str).str.strip().str.lower()
                unique_text = set(text_vals.unique())
                matches = unique_text & OUTCOME_TEXT_SIGNALS
                if matches and len(unique_text) <= 10:
                    reason = (f'programme status column — '
                              f'recognised values: '
                              f'{", ".join(sorted(matches)[:4])}')
            except Exception:
                pass

        if reason:
            try:
                vc = df[col].value_counts().to_dict()
            except Exception:
                vc = {}
            candidates.append((col, reason, vc))

    # Exclude columns that are already model features
    candidates = [(c, r, v) for c, r, v in candidates if c not in FEATURES]
    return candidates


# ── 2. Outcome column validation ──────────────────────────────

def normalize_outcome_column(df, outcome_col):
    """
    Automatically normalise a programme outcome column to binary 0/1
    before validation and recalibration.

    Handles common real-world HIV programme export values:
      Poor outcome  → 1:  LTFU, Dead, Interrupted, Failure, Defaulted,
                          Transferred Out, Stopped, Discontinued, Yes, True, Y, 1
      Good outcome  → 0:  Retained, Active, Suppressed, Alive, Enrolled,
                          On Treatment, No, False, N, 0

    Returns (df_out, mapping_applied, ambiguous_values) where:
      - df_out          : copy of df with outcome_col recoded to 0/1 float
      - mapping_applied : dict of {original_value: mapped_value} for audit trail
      - ambiguous_values: list of values that could not be auto-mapped
    """
    # Canonical mappings — case-insensitive, stripped
    POOR_OUTCOME = {
        'ltfu', 'lost to follow-up', 'lost to follow up', 'lost',
        'dead', 'death', 'died',
        'interrupted', 'interruption', 'treatment interrupted',
        'failure', 'failed', 'treatment failure',
        'defaulted', 'default',
        'transferred out', 'transfer out', 'to',
        'stopped', 'stop', 'discontinued', 'discontinue',
        'drop', 'dropped', 'dropped out',
        'non-adherent', 'nonadherent',
        '1', 'yes', 'true', 'y',
    }
    GOOD_OUTCOME = {
        'retained', 'retain', 'active', 'on treatment', 'on art',
        'suppressed', 'virally suppressed', 'vls',
        'alive', 'living',
        'enrolled', 'current', 'in care',
        'adherent', 'compliant',
        '0', 'no', 'false', 'n',
    }

    df_out = df.copy()
    col = df_out[outcome_col].copy()

    # Fast path: genuine numeric integer/float dtype — check if already 0/1
    # Do this BEFORE dtype conversion to preserve int columns
    if pd.api.types.is_numeric_dtype(col):
        numeric = pd.to_numeric(col, errors='coerce')
        numeric_unique = set(numeric.dropna().unique())
        if numeric_unique <= {0, 1, 0.0, 1.0}:
            df_out[outcome_col] = numeric.astype(float)
            return df_out, {}, []

    # Normalise ArrowStringArray and other non-standard dtypes to plain object
    try:
        col = col.astype(object)
    except Exception:
        pass

    # Attempt text mapping
    mapping_applied = {}
    ambiguous_values = []
    recoded = []

    for raw_val in col:
        if pd.isna(raw_val):
            recoded.append(np.nan)
            continue
        # Try numeric first
        try:
            n = float(raw_val)
            if n in (0.0, 1.0):
                recoded.append(n)
                mapping_applied[str(raw_val)] = int(n)
                continue
        except (ValueError, TypeError):
            pass
        # Text normalisation
        norm = str(raw_val).strip().lower()
        if norm in POOR_OUTCOME:
            recoded.append(1.0)
            mapping_applied[str(raw_val)] = 1
        elif norm in GOOD_OUTCOME:
            recoded.append(0.0)
            mapping_applied[str(raw_val)] = 0
        else:
            recoded.append(np.nan)
            if str(raw_val) not in ambiguous_values:
                ambiguous_values.append(str(raw_val))

    df_out[outcome_col] = recoded
    return df_out, mapping_applied, ambiguous_values


def validate_outcome_column(df, outcome_col, available_features):
    """
    Run the four validation checks.
    Returns dict with pass/fail per check and overall can_proceed bool.
    """
    results = {
        'checks': {},
        'can_proceed': False,
        'warnings': [],
    }

    try:
        outcome = pd.to_numeric(df[outcome_col], errors='coerce')
    except Exception:
        results['checks']['outcome_readable'] = {
            'passed': False,
            'label': 'Outcome column readable',
            'detail': 'Could not parse outcome column as numeric.',
        }
        return results

    n_total = len(outcome)
    n_missing = outcome.isnull().sum()
    outcome_clean = outcome.dropna()
    n_clean = len(outcome_clean)

    # Check 1 — Sample size
    passed_n = n_clean >= RECAL_MIN_PATIENTS
    results['checks']['sample_size'] = {
        'passed': passed_n,
        'label': 'Sufficient sample size',
        'detail': (f'{n_clean:,} patients with known outcome '
                   f'(minimum required: {RECAL_MIN_PATIENTS:,})'),
        'value': n_clean,
        'threshold': RECAL_MIN_PATIENTS,
    }

    # Guard: if all outcome values are non-numeric / NaN, fail gracefully
    if n_clean == 0:
        results['checks']['outcome_events'] = {
            'passed': False,
            'label': 'Sufficient outcome events',
            'detail': (
                'Outcome column contains no valid numeric values after parsing. '
                'Expected binary 0/1. If your outcome column uses text values '
                '(e.g. "Active", "LTFU", "Dead"), please recode to 1=poor outcome, '
                '0=good outcome before uploading.'
            ),
            'value': 0,
            'threshold': RECAL_MIN_POS_EVENTS,
            'prevalence': 0,
        }
        results['can_proceed'] = False
        return results

    # Check 2 — Outcome events
    unique_vals = set(outcome_clean.unique())
    if not unique_vals <= {0, 1, 0.0, 1.0}:
        results['checks']['outcome_binary'] = {
            'passed': False,
            'label': 'Recognised outcome definition',
            'detail': (f'Outcome column contains unexpected values: '
                       f'{sorted(unique_vals)[:5]}. '
                       f'Expected binary 0/1 only.'),
        }
        return results

    n_pos = int((outcome_clean == 1).sum())
    n_neg = int((outcome_clean == 0).sum())
    passed_events = n_pos >= RECAL_MIN_POS_EVENTS
    prevalence_pct = (n_pos / n_clean * 100) if n_clean > 0 else 0.0
    results['checks']['outcome_events'] = {
        'passed': passed_events,
        'label': 'Sufficient outcome events',
        'detail': (f'{n_pos:,} positive outcome events '
                   f'(minimum required: {RECAL_MIN_POS_EVENTS:,}). '
                   f'Outcome prevalence: {prevalence_pct:.1f}%'),
        'value': n_pos,
        'threshold': RECAL_MIN_POS_EVENTS,
        'prevalence': n_pos / n_clean if n_clean > 0 else 0,
    }

    # Check 3 — Outcome missingness
    miss_pct = n_missing / n_total if n_total > 0 else 1.0
    passed_miss = miss_pct <= RECAL_MAX_OUTCOME_MISS
    results['checks']['outcome_missingness'] = {
        'passed': passed_miss,
        'label': 'Acceptable outcome missingness',
        'detail': (f'{n_missing:,} missing outcome values '
                   f'({miss_pct*100:.1f}% of cohort). '
                   f'Maximum allowed: {RECAL_MAX_OUTCOME_MISS*100:.0f}%'),
        'value': miss_pct,
        'threshold': RECAL_MAX_OUTCOME_MISS,
    }

    # Check 4 — Recognised outcome definition (binary confirmed)
    passed_def = True
    results['checks']['outcome_definition'] = {
        'passed': passed_def,
        'label': 'Recognised outcome definition',
        'detail': (f'Binary outcome confirmed: '
                   f'{n_pos:,} positive (1) and {n_neg:,} negative (0). '
                   f'Values: {sorted(unique_vals)}'),
    }

    # Check 5 — Feature missingness (warning only, not blocking)
    if available_features:
        feat_miss = []
        for feat in available_features:
            if feat in df.columns:
                miss = df[feat].isnull().mean()
                if miss > RECAL_MAX_FEAT_MISS:
                    feat_miss.append(
                        f'{FEAT_LABELS.get(feat, feat)} '
                        f'({miss*100:.0f}% missing)'
                    )
        if feat_miss:
            results['warnings'].append(
                f'High missingness in predictor features: '
                f'{", ".join(feat_miss[:3])}. '
                f'Recalibration will proceed but results may be less reliable.'
            )

    # Overall decision
    blocking = ['sample_size', 'outcome_events', 'outcome_missingness',
                'outcome_definition']
    results['can_proceed'] = all(
        results['checks'].get(c, {}).get('passed', False)
        for c in blocking
        if c in results['checks']
    )

    return results


# ── 3. Bootstrap AUC with confidence interval ─────────────────

def bootstrap_auc(y_true, y_prob, n_boot=BOOTSTRAP_N, seed=42):
    """
    Returns (auc, ci_lower, ci_upper) using percentile bootstrap.
    """
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = np.array(y_true)[idx]
        yp = np.array(y_prob)[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            continue
    if len(aucs) < 10:
        base = roc_auc_score(y_true, y_prob)
        return base, None, None
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── 4. Find optimal threshold ────────────────────────────────

def find_optimal_threshold(y_true, y_prob, method='youden'):
    """
    Find threshold maximising Youden's J (sensitivity + specificity - 1).
    Returns (threshold, sensitivity, specificity).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    # Compute confusion matrix at threshold
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
    }


# ── 5. Run recalibration ──────────────────────────────────────

def sanitize_feature_matrix(df, features=None):
    """
    Sanitize all model feature columns to numeric float before any
    matrix operation (.astype(float), model.predict_proba, etc.).

    Handles real-world programme export values that pass column normalisation
    but survive as strings into the feature matrix:
      T / True / Yes / Y / 1     → 1.0
      F / False / No / N / 0     → 0.0
      Female / Woman / F / W     → 1.0  (sex_female encoding)
      Male / Man / M             → 0.0
      Stage I/II/III/IV text     → 1.0/2.0/3.0/4.0
      Numeric strings ('3.5')    → 3.5
      Unparseable text           → 0.0 (neutral default, logged)

    Returns (df_clean, audit_log) where audit_log is a list of strings
    describing every column that was coerced and what values were found.
    Does NOT crash on unresolvable values — substitutes 0.0 and logs.
    """
    import re as _re
    df = df.copy()
    features = features or FEATURES
    audit_log = []

    # Value maps applied before generic numeric coercion
    BOOL_TRUE  = {'t', 'true', 'yes', 'y', '1', 'positive', 'pos'}
    BOOL_FALSE = {'f', 'false', 'no', 'n', '0', 'negative', 'neg', 'none', ''}
    FEMALE_STR = {'female', 'woman', 'w', '2'}
    MALE_STR   = {'m', 'male', 'man', '0', '1'}
    ROMAN      = {'i': 1.0, 'ii': 2.0, 'iii': 3.0, 'iv': 4.0}

    # Binary features — True=1, False=0
    BINARY_FEATURES = {
        'sex_female', 'had_interruption', 'opp_infection',
        'side_effects', 'tb_positive', 'stage_worsened',
    }

    for feat in features:
        if feat not in df.columns:
            continue

        col = df[feat]

        # Already numeric — just coerce safely
        if pd.api.types.is_numeric_dtype(col):
            df[feat] = pd.to_numeric(col, errors='coerce').fillna(0.0)
            continue

        # String / object column — needs parsing
        original_dtype = str(col.dtype)
        try:
            col_str = col.astype(str)
        except Exception:
            df[feat] = 0.0
            audit_log.append(f"{feat}: could not convert to string — filled with 0.0")
            continue

        unique_vals = col_str.dropna().unique()
        non_numeric = [v for v in unique_vals
                       if v not in ('nan', 'None', '')
                       and pd.to_numeric(v, errors='coerce') != pd.to_numeric(v, errors='coerce')]

        if not any(True for _ in non_numeric):
            # All values are numeric strings — simple coerce
            df[feat] = pd.to_numeric(col_str, errors='coerce').fillna(0.0)
            if len(non_numeric) == 0 and original_dtype != 'float64':
                audit_log.append(
                    f"{feat}: numeric strings coerced to float "
                    f"(dtype was {original_dtype})"
                )
            continue

        # Non-numeric strings present — apply feature-specific mapping
        offending = sorted(set(non_numeric))
        recoded = []
        unresolved = []

        for raw in col_str:
            s = str(raw).strip().lower()
            if s in ('nan', 'none', ''):
                recoded.append(0.0)
                continue

            # Try plain numeric first
            try:
                recoded.append(float(s))
                continue
            except (ValueError, TypeError):
                pass

            # sex_female special handling
            if feat == 'sex_female':
                if s in FEMALE_STR or s in ('t', 'true', 'yes', 'y'):
                    recoded.append(1.0)
                elif s in MALE_STR or s in ('f', 'false', 'no', 'n'):
                    recoded.append(0.0)
                else:
                    recoded.append(0.0)
                    if s not in unresolved: unresolved.append(s)
                continue

            # stage_start_num — Roman numerals and "Stage N" text
            if feat == 'stage_start_num':
                if s in ROMAN:
                    recoded.append(ROMAN[s])
                    continue
                # Extract roman numeral from compound string e.g. "stage iii", "who stage iv"
                roman_match = _re.search(r'\b(iv|iii|ii|i)\b', s)
                if roman_match:
                    recoded.append(ROMAN[roman_match.group(1)])
                    continue
                # Extract digit from string e.g. "stage 3", "who stage 3"
                m = _re.search(r'(\d)', s)
                if m:
                    recoded.append(float(m.group(1)))
                    continue
                recoded.append(2.0)  # default neutral stage
                if s not in unresolved: unresolved.append(s)
                continue

            # Binary features
            if feat in BINARY_FEATURES:
                if s in BOOL_TRUE:
                    recoded.append(1.0)
                elif s in BOOL_FALSE:
                    recoded.append(0.0)
                else:
                    recoded.append(0.0)
                    if s not in unresolved: unresolved.append(s)
                continue

            # Generic fallback — boolean-like then zero
            if s in BOOL_TRUE:
                recoded.append(1.0)
            elif s in BOOL_FALSE:
                recoded.append(0.0)
            else:
                recoded.append(0.0)
                if s not in unresolved: unresolved.append(s)

        df[feat] = recoded

        # Build audit entry
        mapping_summary = f"offending values: {offending}"
        if unresolved:
            audit_log.append(
                f"{feat}: {len(offending)} non-numeric value(s) found "
                f"{offending} — coerced to 0/1 where recognised; "
                f"unresolvable values {unresolved} → 0.0 (neutral default)"
            )
        else:
            audit_log.append(
                f"{feat}: {len(offending)} non-numeric value(s) "
                f"{offending} — all successfully coerced"
            )

    return df, audit_log


def run_recalibration(df_mapped, outcome_col, model, available_features):
    """
    Full recalibration pipeline.
    Returns recal_results dict with all metrics and calibrated model.
    """
    # ── Step 0: Sanitize feature matrix ──────────────────────
    # Coerces any non-numeric values in feature columns to float
    # before matrix operations. Prevents crashes on real-world
    # programme exports containing T/F, Yes/No, text categories.
    df_mapped, _sanitize_log = sanitize_feature_matrix(df_mapped, FEATURES)

    # Align outcome with available rows
    outcome = pd.to_numeric(df_mapped[outcome_col], errors='coerce')
    valid_mask = outcome.notna()
    df_valid = df_mapped[valid_mask].copy()
    y_true = outcome[valid_mask].astype(int).values

    # Ensure all features present
    for feat in FEATURES:
        if feat not in df_valid.columns:
            df_valid[feat] = 0

    X = df_valid[FEATURES].values.astype(float)

    # Base model predictions
    y_prob_base = model.predict_proba(X)[:, 1]

    # AUC before calibration
    auc_base = roc_auc_score(y_true, y_prob_base)

    # Decide calibration method
    n_clean = len(y_true)
    use_isotonic = n_clean >= RECAL_ISOTONIC_MIN

    # ── Fit calibration layer with held-out evaluation ────────────────────────
    # IMPORTANT: calibrator is fit on a 70% split and evaluated on the held-out
    # 30% to avoid in-sample AUC inflation (data leakage). The final calibrator
    # returned is refit on all data so it is as accurate as possible for
    # operational use — but the reported AUC comes from the held-out fold.
    from sklearn.model_selection import train_test_split
    (y_prob_cal_train, y_prob_cal_test,
     y_true_train,     y_true_test) = train_test_split(
        y_prob_base, y_true,
        test_size=0.30, random_state=42, stratify=y_true
    )

    if use_isotonic:
        # Fit on train split
        _cal_tmp = IsotonicRegression(out_of_bounds='clip')
        _cal_tmp.fit(y_prob_cal_train, y_true_train)
        # Evaluate on held-out test split
        y_prob_cal_eval = _cal_tmp.predict(y_prob_cal_test)
        # Refit on all data for operational use
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_prob_base, y_true)
        y_prob_cal = calibrator.predict(y_prob_base)
        cal_method = 'Isotonic Regression'
    else:
        # Fit on train split
        _cal_tmp = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000)
        _cal_tmp.fit(y_prob_cal_train.reshape(-1, 1), y_true_train)
        # Evaluate on held-out test split
        y_prob_cal_eval = _cal_tmp.predict_proba(
            y_prob_cal_test.reshape(-1, 1))[:, 1]
        # Refit on all data for operational use
        calibrator = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000)
        calibrator.fit(y_prob_base.reshape(-1, 1), y_true)
        y_prob_cal = calibrator.predict_proba(
            y_prob_base.reshape(-1, 1))[:, 1]
        cal_method = 'Platt Scaling (Logistic Regression)'

    # AUC computed on held-out evaluation split — not in-sample
    auc_cal, ci_lo, ci_hi = bootstrap_auc(y_true_test, y_prob_cal_eval)

    # Optimal threshold — computed on held-out eval split for honesty
    thresh_metrics = find_optimal_threshold(y_true_test, y_prob_cal_eval)

    # Brier score on held-out eval split
    brier = brier_score_loss(y_true_test, y_prob_cal_eval)

    # Outcome prevalence (full dataset)
    prevalence = float(y_true.mean())

    # ROC curve data for plotting — held-out eval split
    fpr, tpr, _ = roc_curve(y_true_test, y_prob_cal_eval)

    return {
        'calibrator': calibrator,
        'cal_method': cal_method,
        'use_isotonic': use_isotonic,
        'n_patients': n_clean,
        'n_positive': int(y_true.sum()),
        'prevalence': prevalence,
        'auc_base': float(auc_base),
        'auc_cal': float(auc_cal),
        'auc_ci_lo': ci_lo,
        'auc_ci_hi': ci_hi,
        'brier': float(brier),
        'threshold': thresh_metrics,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'baseline_auc': BASELINE_AUC,
        'outcome_col': outcome_col,
        'y_true': y_true_test.tolist(),
        'y_prob_cal': y_prob_cal_eval.tolist(),
        'eval_note': '70/30 split — AUC, threshold, Brier computed on held-out 30%',
    }


# ── 6. Apply calibration to new predictions ───────────────────

def apply_calibration(probs_raw, calibrator, use_isotonic):
    """
    Apply stored calibration layer to new predicted probabilities.
    """
    try:
        if use_isotonic:
            return calibrator.predict(probs_raw)
        else:
            return calibrator.predict_proba(
                probs_raw.reshape(-1, 1))[:, 1]
    except Exception:
        return probs_raw


# ── 7. Synthetic test data generator ─────────────────────────

def generate_synthetic_recal_data(scenario='good', seed=42):
    """
    Generate synthetic data for testing recalibration.
    Outcomes are correlated with clinical features using a log-odds model
    that mirrors the actual SHAP importances from SmartDaaS training —
    producing a realistic AUC in the 0.70-0.75 range.
    Scenario A: good       — 500 patients, ~13% outcome, passes all checks
    Scenario B: few_events — 500 patients, 2% outcome, fails events check
    Scenario C: small_n    — 150 patients, ~13% outcome, fails sample size
    """
    from scipy.special import expit

    rng = np.random.RandomState(seed)

    scenarios = {
        'good':       {'n': 500,  'base': -2.6},
        'few_events': {'n': 500,  'base': -4.5},
        'small_n':    {'n': 150,  'base': -2.6},
    }
    cfg = scenarios.get(scenario, scenarios['good'])
    n, base = cfg['n'], cfg['base']

    # Realistic Nigerian HIV programme feature distributions
    age              = np.clip(rng.normal(35, 10, n), 18, 65)
    sex_female       = rng.binomial(1, 0.60, n).astype(float)
    cd4_start        = np.clip(np.exp(rng.normal(5.4, 0.9, n)), 1, 1500)
    stage_start      = rng.choice([1, 2, 3, 4], n, p=[0.20, 0.38, 0.30, 0.12]).astype(float)
    weight_start     = np.clip(rng.normal(60, 12, n), 30, 120)
    height_m         = np.clip(rng.normal(165, 8, n), 140, 195) / 100
    bmi_start        = np.clip(weight_start / (height_m ** 2), 12, 45)
    days_to_art      = np.where(
        rng.binomial(1, 0.35, n) == 1, 0,
        np.clip(rng.exponential(60, n), 1, 730)
    ).astype(float)
    had_interruption = rng.binomial(1, 0.18, n).astype(float)
    opp_infection    = rng.binomial(1, 0.15, n).astype(float)
    side_effects     = rng.binomial(1, 0.12, n).astype(float)
    tb_positive      = rng.binomial(1, 0.08, n).astype(float)
    cd4_improvement  = np.clip(rng.normal(80, 120, n), -400, 600)
    cd4_recent       = np.clip(cd4_start + cd4_improvement, 1, 1500)
    weight_change    = np.clip(rng.normal(1.5, 6, n), -25, 25)
    stage_worsened   = rng.binomial(1, 0.10, n).astype(float)

    # Outcomes correlated with features — coefficients mirror SHAP importances
    log_odds = (
        base
        + 2.2  * had_interruption
        + 0.50 * (stage_start / 4)
        + 0.30 * sex_female
        - 0.40 * (cd4_start / 500)
        - 0.35 * (cd4_recent / 500)
        + 0.28 * opp_infection
        - 0.22 * (cd4_improvement / 300)
        + 0.25 * stage_worsened
        + 0.15 * (days_to_art / 365)
        + 0.10 * side_effects
        + 0.08 * tb_positive
        + 0.05 * (age / 50)
        + rng.normal(0, 0.5, n)
    )
    poor_outcome = rng.binomial(1, expit(log_odds), n).astype(float)

    return pd.DataFrame({
        'Age':               age.round(1),
        'sex_female':        sex_female,
        'Cd4AtStart':        cd4_start.round(0),
        'MostRecentCd4Count':cd4_recent.round(0),
        'CD4_improvement':   cd4_improvement.round(0),
        'stage_start_num':   stage_start,
        'WeightAtStart':     weight_start.round(1),
        'weight_change':     weight_change.round(1),
        'BMI_start':         bmi_start.round(1),
        'days_to_ART':       days_to_art.round(0),
        'had_interruption':  had_interruption,
        'opp_infection':     opp_infection,
        'side_effects':      side_effects,
        'tb_positive':       tb_positive,
        'stage_worsened':    stage_worsened,
        'patient_id':        [f'SYN-{i:04d}' for i in range(n)],
        'poor_outcome':      poor_outcome,
    })


# ── 8. Render recalibration UI ────────────────────────────────

def render_recalibration_page(model):
    """
    Full Local Validation page UI.
    """
    st.markdown("""
### ✅ Local Validation & Recalibration

Validate SmartDaaS performance on your programme's own data and generate
a locally-calibrated risk model specific to your context.
""")

    st.markdown("""<div class="info-box">
    <strong>What this does:</strong> When you upload historical programme data with known
    patient outcomes, this module fits a calibration layer on top of the SmartDaaS base model.
    The result is a <strong>locally-validated AUC</strong> specific to your programme —
    the number you report to funders, not the Nigerian discovery cohort baseline of 0.772.<br><br>
    <strong>What you need:</strong> A CSV of historical patients where outcomes are already known
    (patients who completed treatment, interrupted, or died). Minimum 200 patients,
    minimum 30 positive outcome events.
    </div>""", unsafe_allow_html=True)

    # ── Baseline reference ─────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">0.772</div>'
            '<div class="metric-lbl">Baseline AUC (Nigerian cohort)</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">27,288</div>'
            '<div class="metric-lbl">Discovery cohort patients</div></div>',
            unsafe_allow_html=True)
    with c3:
        local_auc = st.session_state.get('local_auc', None)
        if local_auc:
            st.markdown(
                f'<div class="metric-box" style="border-color:#3fb950">'
                f'<div class="metric-val" style="color:#3fb950">{local_auc:.3f}</div>'
                f'<div class="metric-lbl">Your local AUC ✓</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="metric-box" style="opacity:0.4">'
                '<div class="metric-val">—</div>'
                '<div class="metric-lbl">Your local AUC (pending)</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Test mode toggle ──────────────────────────────────
    st.markdown('<p class="section-hdr">Data Source</p>', unsafe_allow_html=True)
    use_synthetic = st.checkbox(
        "🧪 Use synthetic test data (for testing and demonstration)",
        value=False,
        help="Generates synthetic patient data with known outcomes to test the recalibration pipeline."
    )

    if use_synthetic:
        scenario = st.selectbox(
            "Select test scenario:",
            options=['good', 'few_events', 'small_n'],
            format_func=lambda x: {
                'good': 'Scenario A — Good data (500 patients, 15% outcome rate) — should PASS all checks',
                'few_events': 'Scenario B — Few outcome events (500 patients, 2% rate) — should FAIL events check',
                'small_n': 'Scenario C — Small sample (150 patients, 15% rate) — should FAIL sample size check',
            }[x]
        )
        df_recal = generate_synthetic_recal_data(scenario=scenario)
        st.info(
            f"🧪 Synthetic data generated: {len(df_recal):,} patients, "
            f"{int(df_recal['poor_outcome'].sum())} positive outcomes "
            f"({df_recal['poor_outcome'].mean()*100:.1f}% prevalence). "
            f"Outcome column: `poor_outcome`."
        )
        outcome_col_confirmed = 'poor_outcome'

    else:
        uploaded_recal = st.file_uploader(
            "Upload historical programme data with known outcomes (CSV)",
            type=['csv'],
            help="Must include patient clinical variables AND a known outcome column (0=good, 1=poor outcome).",
            key="recal_uploader"
        )
        if uploaded_recal is None:
            st.info(
                "Upload a CSV file containing historical patient data with known outcomes. "
                "The outcome column should be binary: 1 = poor outcome "
                "(non-adherence, interruption, or death), 0 = good outcome."
            )
            st.markdown("---")
            _render_recal_requirements()
            return

        try:
            df_raw = pd.read_csv(uploaded_recal)
            df_raw, _recal_log = preprocess_phia_compatible(df_raw)
            df_recal, _, _ = normalize_columns(df_raw)
            if 'patient_id' not in df_recal.columns:
                df_recal['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_recal))]
            st.success(f"✓ {len(df_recal):,} patients loaded from {uploaded_recal.name}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Detect outcome columns
        candidates = detect_outcome_columns(df_recal)
        if not candidates:
            st.error(
                "No outcome column detected in your upload. "
                "Please add a column named `poor_outcome`, `outcome`, or similar "
                "containing binary values (1 = poor outcome, 0 = good outcome)."
            )
            return

        st.markdown('<p class="section-hdr">Outcome Column Selection</p>',
                    unsafe_allow_html=True)
        if len(candidates) == 1:
            col_name, reason, vc = candidates[0]
            st.info(f"Outcome column auto-detected: **`{col_name}`** — {reason}")
            outcome_col_confirmed = col_name
        else:
            col_options = [c[0] for c in candidates]
            col_reasons = {c[0]: c[1] for c in candidates}
            outcome_col_confirmed = st.selectbox(
                "Multiple outcome columns detected. Select the correct one:",
                options=col_options,
                format_func=lambda x: f"{x} — {col_reasons[x]}"
            )

    # ── Outcome normalisation ──────────────────────────────
    # Automatically recode common programme text values to binary 0/1
    # before validation. Surfaces mapping to user for transparency.
    st.markdown("---")
    df_recal, _outcome_mapping, _ambiguous = normalize_outcome_column(
        df_recal, outcome_col_confirmed
    )

    if _outcome_mapping:
        # Show what was recoded — user can verify
        recode_lines = []
        for orig, mapped in sorted(set(_outcome_mapping.items())):
            label = "poor outcome (1)" if mapped == 1 else "good outcome (0)"
            recode_lines.append(f"- `{orig}` → {label}")
        with st.expander(
            f"ℹ️ Outcome values auto-recoded ({len(set(_outcome_mapping.keys()))} "
            f"unique value(s) normalised to binary 0/1)", expanded=True
        ):
            st.markdown("\n".join(recode_lines))
            st.caption(
                "Review the mapping above. If any value has been assigned incorrectly, "
                "recode your outcome column manually before re-uploading."
            )

    if _ambiguous:
        # ── Interactive ambiguous value mapping ───────────────
        # Instead of forcing the user to leave the app and re-upload,
        # present a dropdown for each ambiguous value so they can
        # resolve the mapping in-session.
        st.warning(
            f"⚠️ **{len(_ambiguous)} outcome value(s) could not be auto-mapped.** "
            f"Please assign each value below — then recalibration will proceed."
        )
        st.markdown("**Assign ambiguous outcome values:**")

        _user_mappings = {}
        _all_resolved = True

        for _amb_val in _ambiguous:
            _user_choice = st.selectbox(
                f"What does `{_amb_val}` mean in your programme?",
                options=[
                    "— select —",
                    "Poor outcome (1) — LTFU / Death / Interruption / Failure",
                    "Good outcome (0) — Retained / Active / Suppressed / On Treatment",
                ],
                key=f"_amb_map_{_amb_val}"
            )
            if _user_choice == "— select —":
                _all_resolved = False
            elif _user_choice.startswith("Poor"):
                _user_mappings[_amb_val] = 1
            else:
                _user_mappings[_amb_val] = 0

        if not _all_resolved:
            st.info(
                "Assign all ambiguous values above to continue. "
                "If you are unsure, check your programme's outcome definitions."
            )
            st.stop()

        # Apply user mappings to the outcome column
        if _user_mappings:
            _col = df_recal[outcome_col_confirmed].copy()
            _recoded = []
            for _v in _col:
                if pd.isna(_v):
                    _recoded.append(np.nan)
                elif str(_v) in _user_mappings:
                    _recoded.append(float(_user_mappings[str(_v)]))
                else:
                    try:
                        _recoded.append(float(_v))
                    except (ValueError, TypeError):
                        _recoded.append(np.nan)
            df_recal[outcome_col_confirmed] = _recoded

            # Show confirmed mapping for audit trail
            _combined_map = {**_outcome_mapping,
                             **{k: v for k, v in _user_mappings.items()}}
            with st.expander(
                f"✅ Ambiguous values resolved — "
                f"{len(_user_mappings)} value(s) mapped by user", expanded=False
            ):
                for _orig, _mapped in sorted(_user_mappings.items()):
                    _lbl = "poor outcome (1)" if _mapped == 1 else "good outcome (0)"
                    st.markdown(f"- `{_orig}` → {_lbl} *(user-assigned)*")
                st.caption(
                    "This mapping is applied in-session only. "
                    "To make it permanent, update your outcome column "
                    "before re-uploading."
                )

    # ── Validation checks ─────────────────────────────────
    st.markdown('<p class="section-hdr">Validation Checks</p>',
                unsafe_allow_html=True)

    available_features = [f for f in FEATURES if f in df_recal.columns]
    val_results = validate_outcome_column(
        df_recal, outcome_col_confirmed, available_features)

    # Render validation checks
    check_icons = {True: '✅', False: '❌'}
    for check_key, check_data in val_results['checks'].items():
        passed = check_data['passed']
        st.markdown(
            f"{check_icons[passed]} **{check_data['label']}** — "
            f"{check_data['detail']}"
        )

    for warning in val_results['warnings']:
        st.warning(f"⚠️ {warning}")

    if not val_results['can_proceed']:
        st.error(
            "**Recalibration cannot proceed** — one or more required checks failed. "
            "Review the issues above, correct your data, and re-upload."
        )
        # Log blocked recalibration
        failed = [k for k, v in val_results['checks'].items()
                  if not v.get('passed', False)]
        n_pts = val_results['checks'].get(
            'sample_size', {}).get('value', 0)
        n_pos = val_results['checks'].get(
            'outcome_events', {}).get('value', 0)
        prev = val_results['checks'].get(
            'outcome_events', {}).get('prevalence', 0)
        log_recalibration(
            supabase, n_pts, n_pos, prev,
            local_auc=None, cal_method=None,
            passed=False, failed_checks=failed
        )
        return

    st.success("✅ All validation checks passed. Ready to run recalibration.")

    # ── Run recalibration ─────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Run Recalibration</p>',
                unsafe_allow_html=True)

    prev = val_results['checks']['outcome_events']['prevalence']
    n_pts = val_results['checks']['sample_size']['value']
    cal_method_preview = ('Isotonic Regression'
                          if n_pts >= RECAL_ISOTONIC_MIN
                          else 'Platt Scaling (Logistic Regression)')

    st.markdown(
        f"**Programme:** {n_pts:,} patients · "
        f"Outcome prevalence: {prev*100:.1f}% · "
        f"Calibration method: {cal_method_preview}"
    )

    run_btn = st.button(
        "🔬 Run Local Recalibration",
        type="primary",
        use_container_width=True
    )

    if run_btn:
        with st.spinner("Running recalibration — fitting calibration layer and computing metrics..."):
            try:
                # Pre-sanitize and surface audit log before running
                _df_san, _san_log = sanitize_feature_matrix(df_recal, FEATURES)
                if _san_log:
                    with st.expander(
                        f"⚙️ Feature matrix coercion — "
                        f"{len(_san_log)} column(s) normalised to numeric",
                        expanded=False
                    ):
                        st.caption(
                            "The following feature columns contained non-numeric "
                            "values and were automatically coerced before recalibration."
                        )
                        for _entry in _san_log:
                            st.markdown(f"- {_entry}")

                recal = run_recalibration(
                    df_recal, outcome_col_confirmed,
                    model, available_features
                )

                # Store in session
                st.session_state['recal_results']   = recal
                st.session_state['local_auc']       = recal['auc_cal']
                st.session_state['local_threshold'] = recal['threshold']['threshold']
                st.session_state['calibrator']      = recal['calibrator']
                st.session_state['use_isotonic']    = recal['use_isotonic']
                st.session_state['recal_done']      = True

                # Log successful recalibration
                log_recalibration(
                    supabase,
                    n_patients=recal['n_patients'],
                    n_positive=recal['n_positive'],
                    prevalence=recal['prevalence'],
                    local_auc=recal['auc_cal'],
                    cal_method=recal['cal_method'],
                    passed=True,
                )

                st.success(
                    f"✅ Recalibration complete. "
                    f"Local AUC: **{recal['auc_cal']:.3f}** "
                    f"(baseline: {BASELINE_AUC})"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Recalibration failed: {e}")
                return

    # ── Display results if recalibration done ─────────────
    if st.session_state.get('recal_done') and 'recal_results' in st.session_state:
        recal = st.session_state['recal_results']
        _render_recal_results(recal)


def _render_recal_requirements():
    """Show data requirements for recalibration."""
    st.markdown('<p class="section-hdr">Data Requirements</p>',
                unsafe_allow_html=True)
    reqs = [
        ('Minimum patients', f'{RECAL_MIN_PATIENTS:,}', 'Patients with known outcome'),
        ('Minimum positive events', f'{RECAL_MIN_POS_EVENTS:,}',
         'Patients with poor outcome = 1'),
        ('Maximum outcome missingness',
         f'{RECAL_MAX_OUTCOME_MISS*100:.0f}%',
         'Missing values in outcome column'),
        ('Outcome format', 'Binary 0/1',
         '1 = poor outcome, 0 = good outcome'),
        ('Recommended minimum', '500+',
         'Enables isotonic regression for better calibration'),
    ]
    req_df = pd.DataFrame(reqs,
                          columns=['Requirement', 'Threshold', 'Notes'])
    st.dataframe(req_df, use_container_width=True)


def _render_recal_results(recal):
    """Render full recalibration results and pilot validation summary."""
    st.markdown("---")
    st.markdown('<p class="section-hdr">Local Validation Results</p>',
                unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    ci_str = (f"(95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f})"
              if recal['auc_ci_lo'] else "")
    auc_delta = recal['auc_cal'] - recal['baseline_auc']
    delta_colour = '#3fb950' if auc_delta >= 0 else '#f85149'
    delta_sign = '+' if auc_delta >= 0 else ''

    with c1:
        st.markdown(
            f'<div class="metric-box" style="border-color:#3fb950">'
            f'<div class="metric-val" style="color:#3fb950">'
            f'{recal["auc_cal"]:.3f}</div>'
            f'<div class="metric-lbl">Local AUC</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val" style="color:{delta_colour}">'
            f'{delta_sign}{auc_delta:.3f}</div>'
            f'<div class="metric-lbl">vs Baseline (0.772)</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["threshold"]["sensitivity"]*100:.1f}%</div>'
            f'<div class="metric-lbl">Sensitivity</div></div>',
            unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["threshold"]["specificity"]*100:.1f}%</div>'
            f'<div class="metric-lbl">Specificity</div></div>',
            unsafe_allow_html=True)
    with c5:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["brier"]:.3f}</div>'
            f'<div class="metric-lbl">Brier Score</div></div>',
            unsafe_allow_html=True)

    if ci_str:
        st.caption(f"Local AUC {ci_str} — based on {BOOTSTRAP_N:,} bootstrap samples.")

    # ── ROC curve ─────────────────────────────────────────
    st.markdown('<p class="section-hdr">ROC Curve — Local Validation</p>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    ax.plot(recal['fpr'], recal['tpr'],
            color='#21d4fd', lw=2,
            label=f"Local model (AUC = {recal['auc_cal']:.3f})")
    ax.plot([0, 1], [0, 1],
            color='#8b949e', lw=1, linestyle='--',
            label='Random classifier (AUC = 0.500)')
    ax.axhline(recal['threshold']['sensitivity'],
               color='#f0a500', lw=1, linestyle=':',
               label=f"Optimal threshold: {recal['threshold']['threshold']:.3f}")
    ax.set_xlabel('False Positive Rate (1 − Specificity)',
                  color='#8b949e', fontsize=9)
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  color='#8b949e', fontsize=9)
    ax.set_title(
        f'ROC Curve — Local Validation\n'
        f'(n={recal["n_patients"]:,} patients, '
        f'{recal["n_positive"]:,} positive outcomes, '
        f'{recal["prevalence"]*100:.1f}% prevalence)',
        color='#e6edf3', fontsize=9, pad=10)
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion matrix at optimal threshold ─────────────
    st.markdown('<p class="section-hdr">Confusion Matrix at Optimal Threshold</p>',
                unsafe_allow_html=True)
    t = recal['threshold']
    cm_data = pd.DataFrame({
        '': ['Predicted: POOR OUTCOME', 'Predicted: GOOD OUTCOME'],
        'Actual: POOR OUTCOME': [f"✅ TP: {t['tp']:,}", f"❌ FN: {t['fn']:,}"],
        'Actual: GOOD OUTCOME': [f"❌ FP: {t['fp']:,}", f"✅ TN: {t['tn']:,}"],
    }).set_index('')
    st.dataframe(cm_data, use_container_width=True)
    st.caption(
        f"At threshold {t['threshold']:.3f} — "
        f"PPV: {t['ppv']*100:.1f}% · NPV: {t['npv']*100:.1f}% · "
        f"Calibration method: {recal['cal_method']}"
    )

    # ── Pilot Validation Summary card ─────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Pilot Validation Summary</p>',
                unsafe_allow_html=True)
    st.markdown(
        "_This summary is formatted for funder and programme director audiences._"
    )

    perf_interp = (
        "substantially better than" if recal['auc_cal'] > recal['baseline_auc'] + 0.05
        else "consistent with" if abs(recal['auc_cal'] - recal['baseline_auc']) <= 0.05
        else "below"
    )

    brier_interp = (
        "excellent" if recal['brier'] < 0.10
        else "good" if recal['brier'] < 0.15
        else "moderate" if recal['brier'] < 0.20
        else "poor"
    )

    ci_str = (f" (95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f})"
              if recal['auc_ci_lo'] else "")

    # ── Two-column card layout — prevents text clustering ────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"""<div style="border:1px solid #3fb950;border-radius:8px;
            padding:16px;background:#0d2137;height:100%">
            <div style="color:#3fb950;font-size:0.95rem;font-weight:700;
                margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">
                Performance Metrics
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
            <tr><td style="padding:5px 6px;color:#8b949e;width:50%">Local AUC</td>
                <td style="padding:5px 6px;color:#3fb950;font-weight:700">
                {recal['auc_cal']:.3f}{ci_str}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">vs Baseline (0.772)</td>
                <td style="padding:5px 6px;color:#cdd9e5">{perf_interp.capitalize()}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Sensitivity</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['sensitivity']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Specificity</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['specificity']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">PPV</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['ppv']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">NPV</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['npv']*100:.1f}%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""<div style="border:1px solid #444c56;border-radius:8px;
            padding:16px;background:#161b22;height:100%">
            <div style="color:#21d4fd;font-size:0.95rem;font-weight:700;
                margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">
                Validation Details
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
            <tr><td style="padding:5px 6px;color:#8b949e;width:55%">Validation cohort</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['n_patients']:,} patients</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Outcome prevalence</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['prevalence']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Positive events</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['n_positive']:,}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Brier score</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['brier']:.3f} ({brier_interp})</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Optimal threshold</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['threshold']:.3f}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Calibration method</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['cal_method'].split(" (")[0]}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#1c1a10;border-left:3px solid #e3b341;
        border-radius:0 6px 6px 0;padding:10px 14px;margin-top:12px;font-size:0.8rem;color:#e3b341">
        ⚠️ This validation used historical retrospective data. Prospective validation is
        recommended before full operational deployment. All outputs require review by
        qualified programme and clinical staff before use.
    </div>""", unsafe_allow_html=True)

    # ── Download pilot summary ────────────────────────────
    summary_csv = pd.DataFrame([{
        'Metric': 'Local AUC',
        'Value': f"{recal['auc_cal']:.3f}",
        'Notes': f"95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f}" if recal['auc_ci_lo'] else 'Bootstrap CI unavailable',
    }, {
        'Metric': 'Baseline AUC (Nigerian cohort)',
        'Value': f"{recal['baseline_auc']:.3f}",
        'Notes': 'Pre-recalibration reference',
    }, {
        'Metric': 'Sensitivity at optimal threshold',
        'Value': f"{recal['threshold']['sensitivity']*100:.1f}%",
        'Notes': f"Threshold: {recal['threshold']['threshold']:.3f}",
    }, {
        'Metric': 'Specificity at optimal threshold',
        'Value': f"{recal['threshold']['specificity']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'PPV',
        'Value': f"{recal['threshold']['ppv']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'NPV',
        'Value': f"{recal['threshold']['npv']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'Brier Score',
        'Value': f"{recal['brier']:.4f}",
        'Notes': 'Lower is better. <0.10 excellent, <0.15 good',
    }, {
        'Metric': 'Validation patients',
        'Value': f"{recal['n_patients']:,}",
        'Notes': '',
    }, {
        'Metric': 'Positive outcome events',
        'Value': f"{recal['n_positive']:,}",
        'Notes': f"{recal['prevalence']*100:.1f}% prevalence",
    }, {
        'Metric': 'Calibration method',
        'Value': recal['cal_method'],
        'Notes': '',
    }])

    st.download_button(
        "📥 Download Pilot Validation Summary (CSV)",
        data=summary_csv.to_csv(index=False).encode(),
        file_name="smartdaas_pilot_validation_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Local validation complete. Risk scores on the Patient Risk page will now "
        "use locally-calibrated probabilities and the locally-optimised threshold "
        "for this session. Re-upload and re-run if your programme data changes."
    )


def derive_engineered_features(df):
    """
    Derive computed model features from raw uploaded columns where possible.
    Operates on a copy. Returns (df_out, list_of_derived, list_of_defaulted).

    Features computed:
      CD4_improvement   = MostRecentCd4Count - Cd4AtStart
      weight_change     = current_weight - WeightAtStart  (if both present)
      had_interruption  = 1 if treatment_interruptions >= 1 else 0
      stage_worsened    = 1 if current WHO stage > baseline WHO stage else 0

    Any feature still missing after derivation is filled with 0 (neutral default).
    """
    df = df.copy()
    derived = []
    defaulted = []

    # ── sex_female encoding ──────────────────────────────────
    # Convert string sex values to binary 0/1 before any numeric operations
    if 'sex_female' in df.columns:
        col = df['sex_female']
        if col.dtype == object or col.dtype.name == 'category':
            female_strings = {'f', 'female', 'woman', 'w', '1', 'yes', 'true'}
            df['sex_female'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in female_strings else (0.0 if x in {'m', 'male', 'man', '0', 'no', 'false'} else np.nan)
            ).fillna(0.0)

    # ── Binary categorical encoding ──────────────────────────
    # Features that may arrive as text categories — encode to 0/1
    # opp_infection: any value other than None/No/Negative/0 = 1
    if 'opp_infection' in df.columns:
        col = df['opp_infection']
        if col.dtype == object or col.dtype.name == 'category':
            none_strings = {'none', 'no', 'negative', '0', 'false', 'nan', ''}
            df['opp_infection'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 0.0 if x in none_strings else 1.0
            )

    # side_effects: any value other than None/No/0 = 1
    if 'side_effects' in df.columns:
        col = df['side_effects']
        if col.dtype == object or col.dtype.name == 'category':
            none_strings = {'none', 'no', 'negative', '0', 'false', 'nan', ''}
            df['side_effects'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 0.0 if x in none_strings else 1.0
            )

    # tb_positive: Yes/Positive/1 = 1, No/Negative/0 = 0
    if 'tb_positive' in df.columns:
        col = df['tb_positive']
        if col.dtype == object or col.dtype.name == 'category':
            pos_strings = {'yes', 'positive', '1', 'true', 'tb', 'tb positive'}
            df['tb_positive'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in pos_strings else 0.0
            )

    # who_stage / stage_start_num: extract numeric from text like "Stage 3" or "III"
    if 'stage_start_num' in df.columns:
        col = df['stage_start_num']
        if col.dtype == object or col.dtype.name == 'category':
            roman = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}
            def parse_stage(x):
                s = str(x).lower().strip()
                # Roman numeral
                if s in roman: return float(roman[s])
                # "stage 3" or "who stage 3"
                import re
                m = re.search(r'(\d)', s)
                if m: return float(m.group(1))
                return np.nan
            df['stage_start_num'] = col.map(parse_stage).fillna(
                pd.to_numeric(col, errors='coerce')
            ).fillna(2.0)  # default to stage 2 if unparseable

    # ── General numeric coercion for all FEATURES ────────────
    # Ensures no string values reach .astype(float) in run_predictions
    for feat in FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0.0)

    # ── CD4_improvement ─────────────────────────────────────
    if 'CD4_improvement' not in df.columns:
        if 'Cd4AtStart' in df.columns and 'MostRecentCd4Count' in df.columns:
            cd4_start   = pd.to_numeric(df['Cd4AtStart'],        errors='coerce')
            cd4_recent  = pd.to_numeric(df['MostRecentCd4Count'], errors='coerce')
            df['CD4_improvement'] = cd4_recent - cd4_start
            derived.append('CD4_improvement')
        else:
            df['CD4_improvement'] = 0
            defaulted.append('CD4_improvement')

    # ── weight_change ────────────────────────────────────────
    if 'weight_change' not in df.columns:
        # Try current_weight or weight_kg columns
        current_weight_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'weight_kg', 'current_weight', 'weight_current', 'weight_now',
              'recent_weight', 'weight_recent', 'weight'}),
            None
        )
        if current_weight_col and 'WeightAtStart' in df.columns:
            w_now   = pd.to_numeric(df[current_weight_col], errors='coerce')
            w_start = pd.to_numeric(df['WeightAtStart'],    errors='coerce')
            df['weight_change'] = w_now - w_start
            derived.append('weight_change')
        else:
            df['weight_change'] = 0
            defaulted.append('weight_change')

    # ── had_interruption ─────────────────────────────────────
    if 'had_interruption' not in df.columns:
        interruption_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'treatment_interruptions', 'interruptions', 'n_interruptions',
              'num_interruptions', 'number_of_interruptions', 'art_interruptions',
              'missed_appointments', 'treatment_gaps'}),
            None
        )
        if interruption_col:
            n_int = pd.to_numeric(df[interruption_col], errors='coerce').fillna(0)
            df['had_interruption'] = (n_int >= 1).astype(float)
            derived.append('had_interruption')
        else:
            df['had_interruption'] = 0
            defaulted.append('had_interruption')

    # ── stage_worsened ───────────────────────────────────────
    if 'stage_worsened' not in df.columns:
        # Look for a current WHO stage column distinct from stage_start_num
        current_stage_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'current_who_stage', 'who_stage_current', 'recent_who_stage',
              'who_stage_now', 'clinical_stage_current', 'stage_current'}),
            None
        )
        if current_stage_col and 'stage_start_num' in df.columns:
            stage_now   = pd.to_numeric(df[current_stage_col],  errors='coerce')
            stage_start = pd.to_numeric(df['stage_start_num'],  errors='coerce')
            df['stage_worsened'] = (stage_now > stage_start).astype(float)
            df['stage_worsened'] = df['stage_worsened'].fillna(0)
            derived.append('stage_worsened')
        else:
            df['stage_worsened'] = 0
            defaulted.append('stage_worsened')

    # ── Fill any remaining missing FEATURES with 0 ──────────
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0
            if feat not in defaulted and feat not in derived:
                defaulted.append(feat)

    return df, derived, defaulted


