"""
SmartDaaS v1.0 — Executive Report
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

One-click PDF executive programme intelligence report.
Generates risk summary, facility findings, SHAP analysis,
recommended actions, and economic impact for funder audiences.

Exports:
    render_executive_report(supabase=None, log_report=None)
"""

import io
from io import BytesIO
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import (
    FEATURES, FEAT_LABELS, COST_PER_POOR_OUTCOME, INTERVENTIONS, BASELINE_THRESHOLD,
    MEDIUM_THRESHOLD, TRAINING_MEDIANS,
)
from model import (
    MODEL_OK, model, X_DEMO, Y_DEMO,
    compute_shap_single, safe_feature_importances,
)
from pipeline import (
    normalize_columns, derive_engineered_features, run_data_quality_screening,
)


def render_executive_report(supabase=None, log_report=None):
    """
    Full Executive Report page UI and PDF generation.
    supabase: optional supabase client for audit logging.
    log_report: optional callable for audit logging, passed in from app.py.
    """
    # Safe no-ops if not provided
    if log_report is None:
        def log_report(*args, **kwargs): pass

    st.markdown("""
    ### Executive Programme Intelligence Report

    **One button. Professional PDF. Ready to hand to your programme director or implementing partner.**

    Upload patient data (or use demo data), and SmartDaaS generates a complete
    programme intelligence report — risk summary, facility findings, key recommendations,
    and economic impact — formatted for executive and funder audiences.
    """)


    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # -- Data governance note (no checkboxes on this page) ---------
    # DUA is acknowledged once on Patient Risk page per session.
    # Executive Report simply shows a small notice and proceeds.
    st.markdown("""<div style='background:#0d1f17;border:1px solid #3fb95044;
        border-radius:6px;padding:8px 14px;font-size:0.82rem;color:#3fb950;margin-bottom:8px'>
        🔒 Patient data processed in-browser only · Not stored or transmitted externally · 
        Decision-support output — review by qualified programme staff required before operational use.
        </div>""", unsafe_allow_html=True)

    # -- Data source: prefer session state from Patient Risk page ----
    st.markdown('<p class="section-hdr">Report Data</p>', unsafe_allow_html=True)

    _scored = st.session_state.get('df_scored')
    _has_scored = (_scored is not None and
                   isinstance(_scored, pd.DataFrame) and
                   len(_scored) > 0 and
                   'risk_pct' in _scored.columns)

    if _has_scored:
        st.markdown(
            f"""<div style='background:#0d1f17;border:1px solid #3fb95044;
            border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#3fb950;margin-bottom:8px'>
            ✅ Using dataset from Patient Risk page — {len(_scored):,} patients already scored.
            Navigate to Patient Risk to change dataset.
            </div>""", unsafe_allow_html=True
        )
        uploaded_rep = None
        use_demo_rep = False
    else:
        col_up, col_demo = st.columns([2, 1])
        with col_up:
            uploaded_rep = st.file_uploader(
                "Upload patient CSV for report", type=['csv'],
                help="Or score your data on the Patient Risk page first."
            )
        with col_demo:
            st.markdown("<br>", unsafe_allow_html=True)
            use_demo_rep = st.checkbox(
                "Use demo data instead",
                value=False,
                help="300 patients from the training set - no upload required"
            )
    st.markdown('<p class="section-hdr">Report Details</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        org_name = st.text_input("Organisation name", value="APIN Public Health Initiatives")
    with c2:
        programme_name = st.text_input("Programme / site name", value="Nigerian National HIV Programme")
    with c3:
        report_date = st.date_input("Report date", value=datetime.date.today())

    prepared_by = st.text_input("Prepared by", value="SmartDaaS Analytics · Lakshmi Kalyani Chinthala")

    # Cache uploaded file in session state so PDF generation rerender doesn't lose it
    if uploaded_rep is not None:
        st.session_state['rep_uploaded_bytes'] = uploaded_rep.read()
        st.session_state['rep_uploaded_name'] = uploaded_rep.name
    if use_demo_rep:
        st.session_state.pop('rep_uploaded_bytes', None)
        st.session_state.pop('rep_uploaded_name', None)

    has_real_upload = (
        not use_demo_rep and
        st.session_state.get('rep_uploaded_bytes') is not None
    )

    # Load data — priority: session state scored df > uploaded CSV > demo
    if _has_scored and not has_real_upload and not use_demo_rep:
        df_rep = _scored.copy()
        # Ensure all FEATURES columns present and numeric
        for f in FEATURES:
            if f not in df_rep.columns:
                df_rep[f] = TRAINING_MEDIANS[f]
            else:
                df_rep[f] = pd.to_numeric(df_rep[f], errors='coerce').fillna(TRAINING_MEDIANS[f])
        if 'patient_id' not in df_rep.columns:
            df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]
        # risk_pct and risk_label already computed on Patient Risk page
        X_rep = df_rep[FEATURES].values.astype(float)
        data_source = f"Patient Risk page dataset ({len(df_rep):,} patients)"
        rep_tier = st.session_state.get('upload_tier', 'ENHANCED')
        rep_pediatric = st.session_state.get('pediatric_indices', [])
    elif not has_real_upload:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
        df_rep = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]
        data_source = "Demo dataset (300 patients from training set)"
        rep_tier = 'ENHANCED'
        rep_pediatric = []
    else:
        try:
            from io import BytesIO as _BIO
            df_raw = pd.read_csv(_BIO(st.session_state['rep_uploaded_bytes']))

            # ── PHIA / population-survey pre-processing ───────────
            df_raw, _rep_derivation_log = preprocess_phia_compatible(df_raw)
            st.session_state['_phia_derivation_log'] = _rep_derivation_log

            # ── Tier detection (same rules as Patient Risk page) ──
            df_rep, missing, mappings_applied = normalize_columns(df_raw)

            if mappings_applied:
                with st.expander(
                    f"ℹ️ Auto-mapped {len(mappings_applied)} column name(s)",
                    expanded=False
                ):
                    for orig, mapped in mappings_applied:
                        if mapped != '__art_inferred__':
                            st.markdown(f"- `{orig}` → `{mapped}`")

            art_confirmed, art_inferred, art_note = detect_art_status(df_raw)
            rep_tier, present, missing_core, std_present, enh_present, _ = \
                detect_tier(df_rep, art_confirmed, art_inferred)
            rep_pediatric = check_pediatric_patients(df_rep)

            st.markdown("---")
            can_proceed = render_tier_report(
                rep_tier, present, missing_core, std_present,
                enh_present, art_confirmed, art_inferred,
                art_note, rep_pediatric, df_rep
            )
            st.markdown("---")

            if not can_proceed:
                st.stop()

            # Core tier: no risk scores in report
            if rep_tier == 'CORE':
                st.info(
                    "**Core Tier upload:** The Executive Report will contain a "
                    "population summary only. Patient-level risk scores are not "
                    "generated for Core tier uploads. Add CD4, WHO stage, TB status, "
                    "and days to ART to unlock full report generation."
                )
                st.stop()

            # Derive engineered features + fill all gaps for ALL tiers
            df_rep, derived_feats_rep, defaulted_feats_rep = derive_engineered_features(df_rep)

            if rep_tier == 'STANDARD':
                st.warning(
                    "**Standard Tier upload:** Risk estimates in this report were "
                    "generated using partial feature availability. Prediction confidence "
                    "and stability may vary depending on which clinical variables are "
                    "present. Interpret all findings alongside clinical judgement and "
                    "local programme context."
                )

            # Data quality screening for uploaded report data
            available_feats = [f for f in FEATURES if f in df_rep.columns]
            dq_rep = run_data_quality_screening(df_rep, available_feats)
            render_data_quality_report(dq_rep, len(df_rep))

            # Validation metadata — surfaced in report for pilot partner audit
            _rep_val_meta = build_validation_metadata(
                df_raw=df_raw,
                df_mapped=df_rep,
                mappings_applied=mappings_applied,
                missing_features=missing,
                derivation_log=st.session_state.get('_phia_derivation_log', []),
                dq_results=dq_rep,
                tier=rep_tier,
            )
            st.session_state['validation_metadata'] = _rep_val_meta
            render_validation_metadata(_rep_val_meta)

            # Apply imputation from quality results
            for feat, info in dq_rep['missing'].items():
                if feat in df_rep.columns:
                    df_rep[feat] = df_rep[feat].fillna(info['impute_val'])

            # Fill any remaining nulls with neutral default
            for col in available_feats:
                if df_rep[col].isnull().any():
                    df_rep[col] = df_rep[col].fillna(df_rep[col].median())

            if 'patient_id' not in df_rep.columns:
                df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]

            data_source = (
                f"Uploaded dataset ({len(df_rep):,} patients — "
                f"{rep_tier} tier, Quality Grade: {dq_rep['grade']})"
            )

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # ── Run predictions ───────────────────────────────────
    # Safety net: ensure all features exist and are numeric
    for f in FEATURES:
        if f not in df_rep.columns:
            df_rep[f] = TRAINING_MEDIANS[f]
        else:
            df_rep[f] = pd.to_numeric(df_rep[f], errors='coerce').fillna(TRAINING_MEDIANS[f])
    X_rep = df_rep[FEATURES].values.astype(float)
    probs_rep = model.predict_proba(X_rep)[:, 1]
    df_rep['risk_pct'] = (probs_rep * 100).round(1)
    df_rep['risk_label'] = ['HIGH' if p >= BASELINE_THRESHOLD else 'MEDIUM' if p >= MEDIUM_THRESHOLD else 'LOW'
                             for p in probs_rep]

    n_total = len(df_rep)
    n_high = (df_rep['risk_label'] == 'HIGH').sum()
    n_med = (df_rep['risk_label'] == 'MEDIUM').sum()
    n_low = (df_rep['risk_label'] == 'LOW').sum()
    avg_risk = df_rep['risk_pct'].mean()
    pct_high = n_high / n_total * 100
    pct_interruption = (df_rep['had_interruption'] > 0.5).mean() * 100
    pct_tb = (df_rep['tb_positive'] > 0.5).mean() * 100
    pct_adv_disease = (df_rep['stage_start_num'] >= 3).mean() * 100
    pct_low_cd4 = (df_rep['Cd4AtStart'] < 200).mean() * 100
    # Consistent with Outreach Optimiser (23% assumed reduction)
    REDUCTION_RATE = 0.23
    # Conservative estimate using Menzies 2011 figure
    est_avoidable_cost = int(n_high * REDUCTION_RATE * COST_PER_POOR_OUTCOME)
    # Mid and upper estimates for methodology section
    est_avoidable_cost_mid   = int(n_high * REDUCTION_RATE * 3500)   # 2024 CPI-adjusted
    est_avoidable_cost_upper = int(n_high * REDUCTION_RATE * 5000)   # full re-engagement cost

    # Preview
    st.markdown('<p class="section-hdr">Report Preview</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <strong>Report will include:</strong> Programme overview · Risk stratification summary ·
    Key clinical findings · Top 10 highest-risk patients · Facility intelligence ·
    Economic impact estimate · Recommended actions · SmartDaaS methodology notes
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_total:,}</div><div class="metric-lbl">Total Patients</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-high" style="padding:0.8rem"><div class="risk-number" style="font-size:1.6rem">{n_high}</div><div class="risk-label">HIGH Risk</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_risk:.1f}%</div><div class="metric-lbl">Avg Risk Score</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">${est_avoidable_cost/1000:.0f}K</div><div class="metric-lbl">Est. Avoidable Cost</div></div>', unsafe_allow_html=True)

    # ── GENERATE PDF ──────────────────────────────────────
    st.markdown('<p class="section-hdr">Generate Report</p>', unsafe_allow_html=True)

    if st.button("📄 Generate Executive PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating professional PDF report..."):
            try:
                from fpdf import FPDF

                # Sanitize all text going into PDF - replace chars Helvetica can't handle
                def _s(text):
                    if not isinstance(text, str):
                        text = str(text)
                    return (text
                        .replace('\u2014', '-')   # em dash
                        .replace('\u2013', '-')   # en dash
                        .replace('\u2192', '->')  # arrow
                        .replace('\u00b7', '.')   # middle dot
                        .replace('\u2019', "'")  # right single quote
                        .replace('\u2018', "'")  # left single quote
                        .replace('\u201c', '"')  # left double quote
                        .replace('\u201d', '"')  # right double quote
                        .replace('\u2026', '...')  # ellipsis
                        .replace('\u00b5', 'u')   # micro sign
                        .replace('\u00d7', 'x')   # multiplication sign
                        .replace('\u00b0', ' deg') # degree sign
                        .encode('latin-1', errors='replace').decode('latin-1')
                    )


                # ── COLOR PALETTE ────────────────────────────────────
                _BG        = (13, 17, 23)
                _CARD      = (30, 37, 48)
                _CYAN      = (0, 229, 255)
                _AMBER     = (255, 179, 0)
                _WHITE     = (255, 255, 255)
                _MUTED     = (176, 190, 197)
                _TEXT      = (226, 234, 243)
                _RED       = (255, 69, 58)
                _GREEN     = (48, 209, 88)
                _LOGO_DARK = (10, 22, 40)

                def _draw_bars(p, bx, by, bar_w, gap, scale=1.0):
                    bars = [
                        (0, 20, (0,136,170)), (1, 14, (0,136,170)),
                        (2,  8, _CYAN),       (3, 17, _CYAN),
                        (4, 11, (0,136,170)),
                    ]
                    for i, top_off, color in bars:
                        bar_h = 28*scale - top_off*scale
                        p.set_fill_color(*color)
                        p.rect(bx + i*(bar_w+gap), by + top_off*scale, bar_w, bar_h, 'F')

                def _smartdaas_text(p, x, y, size):
                    p.set_font('Helvetica', 'B', size)
                    sw = p.get_string_width('Smart')
                    p.set_text_color(*_WHITE)
                    p.set_xy(x, y); p.cell(sw, size*0.5, 'Smart')
                    p.set_text_color(*_CYAN)
                    p.set_xy(x+sw, y); p.cell(sw, size*0.5, 'DaaS')

                def _inner_header(p, page_title):
                    p.set_fill_color(*_LOGO_DARK)
                    p.rect(0, 0, 210, 16, 'F')
                    p.set_fill_color(*_CYAN)
                    p.rect(0, 16, 210, 0.8, 'F')
                    p.set_fill_color(*_BG)
                    p.rect(0, 16.8, 210, 297, 'F')
                    p.set_fill_color(*_CYAN)
                    p.rect(0, 0, 3, 297, 'F')
                    p.set_font('Helvetica', 'B', 11)
                    sw = p.get_string_width('Smart')
                    p.set_text_color(*_WHITE); p.set_xy(8, 4); p.cell(sw, 8, 'Smart')
                    p.set_text_color(*_CYAN);  p.set_xy(8+sw, 4); p.cell(sw, 8, 'DaaS')
                    p.set_font('Helvetica', 'B', 10)
                    p.set_text_color(*_MUTED)
                    p.set_xy(0, 4); p.cell(202, 8, page_title, align='R')

                def _section_title(p, title):
                    p.set_font('Helvetica', 'B', 14)
                    p.set_text_color(*_CYAN)
                    p.set_x(15); p.cell(180, 10, title, ln=True)
                    p.set_draw_color(*_CYAN)
                    p.set_line_width(0.5)
                    p.line(15, p.get_y(), 195, p.get_y())
                    p.ln(5)

                class SmartDaaSReport(FPDF):
                    def header(self): pass
                    def footer(self): pass

                pdf = SmartDaaSReport()

                # Patch pdf methods to auto-sanitize all text
                _orig_cell = pdf.cell
                _orig_multi = pdf.multi_cell
                def _safe_cell(*a, **kw):
                    a = tuple(_s(x) if isinstance(x, str) else x for x in a)
                    kw = {k: _s(v) if isinstance(v, str) else v for k, v in kw.items()}
                    return _orig_cell(*a, **kw)
                def _safe_multi(*a, **kw):
                    a = tuple(_s(x) if isinstance(x, str) else x for x in a)
                    kw = {k: _s(v) if isinstance(v, str) else v for k, v in kw.items()}
                    return _orig_multi(*a, **kw)
                pdf.cell = _safe_cell
                pdf.multi_cell = _safe_multi

                # ── COVER PAGE ───────────────────────────────────────
                pdf.set_auto_page_break(auto=False)
                pdf.add_page()
                pdf.set_fill_color(*_BG); pdf.rect(0, 0, 210, 297, 'F')
                pdf.set_fill_color(*_CYAN); pdf.rect(0, 0, 3, 297, 'F')

                # Header bar
                pdf.set_fill_color(*_LOGO_DARK); pdf.rect(0, 0, 210, 42, 'F')
                pdf.set_fill_color(*_CYAN); pdf.rect(0, 42, 210, 1.5, 'F')
                _draw_bars(pdf, 12, 7, 7, 2, scale=1.0)
                _smartdaas_text(pdf, 60, 10, 22)
                pdf.set_font('Helvetica', '', 9); pdf.set_text_color(0,171,187)
                pdf.set_xy(60, 25); pdf.cell(90, 8, 'HIV PROGRAMME INTELLIGENCE')
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_MUTED)
                pdf.set_xy(110, 12); pdf.cell(85, 7, 'Programme Intelligence', align='R')
                pdf.set_font('Helvetica', '', 10)
                pdf.set_xy(110, 21); pdf.cell(85, 8, _s(f'Executive Report  .  {report_date.strftime("%Y")}'), align='R')

                # Report title
                pdf.set_font('Helvetica', 'B', 22); pdf.set_text_color(*_WHITE)
                pdf.set_xy(15, 52); pdf.cell(180, 13, 'HIV Programme Intelligence Report')
                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                pdf.set_xy(15, 66); pdf.cell(180, 9, 'Predictive Risk  .  Facility Analytics  .  Economic Impact  .  Recommended Actions')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.4)
                pdf.line(15, 75, 195, 75)

                # Org card
                pdf.set_fill_color(*_CARD); pdf.rect(15, 79, 180, 38, 'F')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, 79, 15, 117)
                cover_details = [
                    ('Organisation', org_name),
                    ('Programme',    programme_name),
                    ('Report Date',  report_date.strftime('%d %B %Y')),
                    ('Prepared by',  prepared_by),
                ]
                for i, (lbl, val) in enumerate(cover_details):
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_xy(22, 82+i*9); pdf.cell(42, 10, _s(lbl+':'))
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_WHITE)
                    pdf.cell(1210, 8, _s(val))

                # Snapshot header
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                pdf.set_xy(15, 122); pdf.cell(180, 8, 'PROGRAMME SNAPSHOT')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, 128, 195, 128)

                # 4 metric cards
                snap_cards = [
                    (f'{n_total:,}',       'TOTAL PATIENTS', _AMBER),
                    (f'{n_high}',          'HIGH RISK',      _RED),
                    (f'{avg_risk:.1f}%',   'AVG RISK SCORE', _AMBER),
                    (f'${est_avoidable_cost//1000}K', 'AVOIDABLE COST', _CYAN),
                ]
                cw = 43
                for i, (val, lbl, vc) in enumerate(snap_cards):
                    cx = 15 + i*(cw+2); cy = 131
                    pdf.set_fill_color(*_CARD); pdf.rect(cx, cy, cw, 22, 'F')
                    pdf.set_fill_color(*vc);    pdf.rect(cx, cy, cw, 1.5, 'F')
                    pdf.set_font('Helvetica', 'B', 15); pdf.set_text_color(*vc)
                    pdf.set_xy(cx, cy+3); pdf.cell(cw, 9, val, align='C')
                    pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(cx, cy+13); pdf.cell(cw, 8, lbl, align='C')

                # Risk bar
                bar_total = 180
                high_w = int(bar_total * pct_high/100)
                med_w  = int(bar_total * (n_med/n_total))
                low_w  = bar_total - high_w - med_w
                by_r = 157
                pdf.set_fill_color(*_RED);   pdf.rect(15, by_r, high_w, 9, 'F')
                pdf.set_fill_color(*_AMBER); pdf.rect(15+high_w, by_r, med_w, 9, 'F')
                pdf.set_fill_color(*_GREEN); pdf.rect(15+high_w+med_w, by_r, low_w, 9, 'F')
                pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_WHITE)
                # Always show all three labels — use coloured text below bar if segment too narrow
                pdf.set_xy(15, by_r+10)
                pdf.cell(high_w, 8, _s(f'{n_high} HIGH {pct_high:.1f}%') if high_w > 25 else '', align='C')
                pdf.cell(med_w,  8, _s(f'{n_med} MED {n_med/n_total*100:.1f}%') if med_w > 25 else '', align='C')
                pdf.cell(low_w,  8, _s(f'{n_low} LOW {n_low/n_total*100:.1f}%') if low_w > 25 else '', align='C')
                # For segments too narrow to label inline, show below in colour
                label_y = by_r + 19
                if high_w <= 25:
                    pdf.set_text_color(*_RED)
                    pdf.set_xy(15, label_y); pdf.cell(60, 7, _s(f'{n_high} HIGH {pct_high:.1f}%'))
                if med_w <= 25:
                    pdf.set_text_color(*_AMBER)
                    pdf.set_xy(80, label_y); pdf.cell(60, 7, _s(f'{n_med} MED {n_med/n_total*100:.1f}%'))
                if low_w <= 25:
                    pdf.set_text_color(*_GREEN)
                    pdf.set_xy(145, label_y); pdf.cell(50, 7, _s(f'{n_low} LOW {n_low/n_total*100:.1f}%'))

                # What's inside
                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_CYAN)
                pdf.set_xy(15, 180); pdf.cell(1100, 8, "WHAT'S INSIDE THIS REPORT")
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.4)
                pdf.line(15, 188, 195, 188)
                contents = [
                    ('01','Executive Summary',             'Risk stratification overview and key clinical findings'),
                    ('02','Facility Intelligence',         'Structural drivers of poor outcomes from 27,288 patients'),
                    ('03','Top 10 Highest Risk Patients',  'Patients requiring immediate contact and adherence support'),
                    ('04','Patient Risk Explanation',      'SHAP analysis — why the highest-risk patient was flagged'),
                    ('05','Recommended Programme Actions', 'Immediate, short-term and strategic interventions'),
                    ('06','Methodology & Limitations',     'Model validation, economic assumptions and data governance'),
                ]
                for i, (num, title, desc) in enumerate(contents):
                    y = 191 + i*12
                    pdf.set_fill_color(*_CYAN); pdf.rect(15, y+1, 11, 10, 'F')
                    pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_BG)
                    pdf.set_xy(15, y+2.5); pdf.cell(11, 9, num, align='C')
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(30, y+1); pdf.cell(165, 7, title)
                    pdf.set_font('Helvetica', '', 10); pdf.set_text_color(*_MUTED)
                    pdf.set_xy(30, y+7); pdf.cell(165, 6, _s(desc))

                # Disclaimer
                pdf.set_fill_color(35, 26, 0); pdf.rect(15, 270, 180, 20, 'F')
                pdf.set_draw_color(*_AMBER); pdf.set_line_width(0.5)
                pdf.line(15, 270, 15, 290)
                pdf.set_xy(21, 272); pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(*_AMBER); pdf.cell(30, 7, 'IMPORTANT')
                pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_TEXT)
                pdf.set_xy(21, 278)
                pdf.multi_cell(168, 7, _s('SmartDaaS v1.0 is a decision-support platform for HIV programme intelligence. Not intended to replace clinical judgment. All outputs require review by qualified programme teams before operational use. Local validation required before deployment.'))

                # ── EXECUTIVE SUMMARY ────────────────────────────────
                pdf.set_auto_page_break(auto=True, margin=18)
                pdf.add_page()
                _inner_header(pdf, 'EXECUTIVE SUMMARY')
                pdf.set_y(24); _section_title(pdf, 'EXECUTIVE SUMMARY')

                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_WHITE)
                pdf.set_x(15); pdf.cell(1100, 8, 'Programme Risk Overview', ln=True)
                pdf.ln(2)

                exec_metrics = [
                    ('Total Patients Analysed',              f'{n_total:,}',                          _WHITE),
                    (f'HIGH Risk Patients (>={BASELINE_THRESHOLD*100:.1f}%)',                    f'{n_high} ({pct_high:.1f}%)',            _RED),
                    (f'MEDIUM Risk Patients ({MEDIUM_THRESHOLD*100:.1f}-{BASELINE_THRESHOLD*100:.1f}%)', f'{n_med} ({n_med/n_total*100:.1f}%)',    _AMBER),
                    (f'LOW Risk Patients (<{MEDIUM_THRESHOLD*100:.1f}%)',                       f'{n_low} ({n_low/n_total*100:.1f}%)',    _GREEN),
                    ('Average Risk Score',                   f'{avg_risk:.1f}%',                       _AMBER),
                    ('Patients with Prior ART Interruption', f'{pct_interruption:.1f}%',               _WHITE),
                    ('Patients with TB Co-infection',        f'{pct_tb:.1f}%',                         _WHITE),
                    ('Advanced Disease (WHO Stage 3-4)',     f'{pct_adv_disease:.1f}%',                _WHITE),
                    ('Severe Immunosuppression (CD4 <200)',  f'{pct_low_cd4:.1f}%',                    _WHITE),
                    ('Estimated Avoidable Cost (conservative)', f'USD {est_avoidable_cost:,}',     _CYAN),
                    ('Estimated Avoidable Cost (mid, 2024)',    f'USD {est_avoidable_cost_mid:,}',  _CYAN),
                    ('Estimated Avoidable Cost (upper)',        f'USD {est_avoidable_cost_upper:,}', _CYAN),
                ]
                for label, value, vc in exec_metrics:
                    pdf.set_fill_color(*_CARD)
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15); pdf.cell(125, 10, _s('  '+label), fill=True)
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*vc)
                    pdf.cell(55, 10, value, fill=True, ln=True)
                    pdf.ln(1)

                pdf.ln(4)
                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_WHITE)
                pdf.set_x(15); pdf.cell(1100, 8, 'Key Clinical Findings', ln=True)
                pdf.set_draw_color(*_GREEN); pdf.set_line_width(0.4)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(4)

                findings = [
                    f'{pct_high:.1f}% of patients are HIGH risk (>={BASELINE_THRESHOLD*100:.1f}% predicted probability of poor ART adherence)',
                    f'{pct_interruption:.1f}% have documented prior ART interruptions - the strongest predictor in the SmartDaaS model',
                    f'{pct_adv_disease:.1f}% presented at WHO Stage 3 or 4, indicating late diagnosis and treatment initiation',
                    f'{pct_low_cd4:.1f}% had CD4 <200 cells/uL at ART start - severely immunosuppressed',
                ]
                if pct_tb > 5:
                    findings.append(f'{pct_tb:.1f}% are TB-HIV co-infected, requiring coordinated treatment protocols')
                for finding in findings:
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                    pdf.set_x(15); pdf.cell(9, 7, chr(149))
                    pdf.multi_cell(163, 9, finding); pdf.ln(1)

                # ── FACILITY INTELLIGENCE ────────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'FACILITY INTELLIGENCE')
                pdf.set_y(24); _section_title(pdf, 'FACILITY INTELLIGENCE')

                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                pdf.set_x(15)
                pdf.multi_cell(180, 8, _s('Based on an analysis of 27,288 patients from a Nigerian HIV programme discovery cohort. These are associations across recorded facility attributes (level, ownership, funding source); the dataset contains no facility identifier, so individual facilities cannot be distinguished. Hypothesis-generating only.'))
                pdf.ln(4)

                # Forest plot
                findings_data = [
                    ('Primary HC vs Tertiary', 1.95, 1.45, 2.61, _RED),
                    ('NGO-Funded Facilities',  1.24, 1.10, 1.39, _AMBER),
                    ('Federal-Funded',         1.25, 1.06, 1.48, _AMBER),
                    ('Female Sex (protective)', 0.87, 0.79, 0.96, _GREEN),
                ]
                plot_y = pdf.get_y()
                scale_start = 77; scale_w = 100
                null_x = scale_start + scale_w * (1.0-0.7)/(2.8-0.7)
                pdf.set_font('Helvetica', '', 9); pdf.set_text_color(*_MUTED)
                for val, lbl in [(0.7,'0.7'),(1.0,'1.0'),(1.5,'1.5'),(2.0,'2.0'),(2.6,'2.6')]:
                    sx = scale_start + scale_w*(val-0.7)/(2.8-0.7)
                    pdf.set_xy(sx-4, plot_y-5); pdf.cell(10, 7, lbl, align='C')
                pdf.set_draw_color(*_MUTED); pdf.set_line_width(0.3)
                pdf.set_dash_pattern(dash=2, gap=2)
                pdf.line(null_x, plot_y, null_x, plot_y+len(findings_data)*18+4)
                pdf.set_dash_pattern(dash=0, gap=0)
                for i, (lbl, OR, lo, hi, color) in enumerate(findings_data):
                    fy = plot_y + 4 + i*18
                    pdf.set_fill_color(*_CARD); pdf.rect(15, fy, 180, 16, 'F')
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(17, fy+4); pdf.cell(510, 8, lbl)
                    pdf.set_text_color(*color)
                    pdf.set_xy(77, fy+4); pdf.cell(13, 10, f'{OR:.2f}')
                    lo_x = scale_start + scale_w*(lo-0.7)/(2.8-0.7)
                    hi_x = scale_start + scale_w*(hi-0.7)/(2.8-0.7)
                    or_x = scale_start + scale_w*(OR-0.7)/(2.8-0.7)
                    cy2 = fy+8
                    pdf.set_draw_color(*color); pdf.set_line_width(1.2)
                    pdf.line(lo_x, cy2, hi_x, cy2)
                    pdf.set_fill_color(*color)
                    pdf.rect(or_x-2, cy2-2, 4, 4, 'F')
                pdf.set_y(plot_y + len(findings_data)*18 + 10)
                pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_MUTED)
                pdf.set_x(15)
                pdf.cell(180, 8, _s('Square = OR estimate. Line = 95% CI. Dashed line = null (OR 1.0).'))
                pdf.ln(5)

                facility_findings = [
                    ('Primary HC vs Tertiary Hospital', 'OR 1.95 (95% CI 1.45-2.61)',
                     'Primary HCs have nearly double the odds of composite poor outcome after patient-level adjustment.',
                     'Structural quality improvement - staffing, drug supply, monitoring systems.'),
                    ('NGO-Funded Facilities', 'OR 1.24 (95% CI 1.10-1.39)',
                     'NGO-funded facilities show independently higher odds - may reflect higher case complexity.',
                     'Outcome-adjusted performance monitoring; investigate funding-to-quality translation.'),
                    ('Federal-Funded Facilities', 'OR 1.25 (95% CI 1.06-1.48)',
                     'Federal government funded facilities show similarly elevated risk.',
                     'Review programme management capacity and reporting burden at federal-funded sites.'),
                    ('Female Sex - Protective Effect', 'OR 0.87 (95% CI 0.79-0.96)',
                     'Female sex is independently protective overall, but advantage nearly disappears at primary HCs.',
                     'Male-targeted interventions at secondary/tertiary; structural improvements at primary HCs.'),
                ]
                for finding_title, stat, desc, action in facility_findings:
                    # If less than 40mm remaining on page, start a new one
                    if pdf.get_y() > 245:
                        pdf.add_page()
                        _inner_header(pdf, 'FACILITY INTELLIGENCE (CONTINUED)')
                        pdf.set_y(24)
                    pdf.set_fill_color(*_CARD)
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                    pdf.set_x(15); pdf.cell(180, 9, _s(f'  {finding_title}: {stat}'), fill=True, ln=True)
                    pdf.ln(1)
                    pdf.set_font('Helvetica', '', 11); pdf.set_text_color(*_TEXT)
                    pdf.set_x(15); pdf.multi_cell(180, 7, _s(f'  Finding: {desc}'))
                    pdf.set_text_color(*_GREEN)
                    pdf.set_x(15); pdf.multi_cell(180, 7, _s(f'  Action: {action}'))
                    pdf.ln(2)

                # ── TOP 10 HIGH RISK PATIENTS ────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'TOP 10 HIGHEST RISK PATIENTS')
                pdf.set_y(24); _section_title(pdf, 'TOP 10 HIGHEST RISK PATIENTS')

                pdf.set_font('Helvetica', 'I', 12); pdf.set_text_color(*_MUTED)
                pdf.set_x(15); pdf.cell(180, 9, 'Patients requiring immediate contact and adherence support.', ln=True)
                pdf.ln(3)

                top10 = df_rep.nlargest(10, 'risk_pct').copy()

                # Compute real SHAP top driver per patient
                top10_indices = top10.index.tolist()
                top_drivers = []
                for idx in top10_indices:
                    try:
                        pos = df_rep.index.get_loc(idx)
                        sv, sv_ok = compute_shap_single(X_rep[pos])
                        sv_arr = np.abs(np.array(sv).flatten())
                        top_feat_idx = int(np.argmax(sv_arr))
                        top_feat = FEATURES[top_feat_idx] if top_feat_idx < len(FEATURES) else ''
                        top_drivers.append(FEAT_LABELS.get(top_feat, top_feat))
                    except Exception:
                        top_drivers.append('—')
                top10['top_driver'] = top_drivers

                # Shorten top driver labels for table display
                SHORT_DRIVER = {
                    'Prior ART Interruption': 'Prior Interruption',
                    'WHO Stage (1-4)': 'WHO Stage',
                    'CD4 at ART Start': 'CD4 at Start',
                    'Most Recent CD4': 'Recent CD4',
                    'CD4 Improvement': 'CD4 Change',
                    'Weight Change (kg)': 'Weight Change',
                    'Clinical Stage Worsened': 'Stage Worsened',
                    'Opportunistic Infection': 'Opp. Infection',
                    'Days: Diagnosis to ART': 'Dx to ART (days)',
                }
                top10['top_driver_short'] = top10['top_driver'].apply(
                    lambda x: SHORT_DRIVER.get(x, x)
                )

                pdf.set_fill_color(*_CARD)
                pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_CYAN)
                t_cols = [('Patient ID',30),('Risk %',20),('Age',12),('CD4',18),
                          ('Stage',16),('Prior Int.',18),('Top Driver',46),('Action',26)]
                pdf.set_x(15)
                for col_name, width in t_cols:
                    pdf.cell(width, 11, col_name, fill=True)
                pdf.ln()
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(2)

                for _, row in top10.iterrows():
                    urgency = ("48h contact" if row['risk_pct'] >= BASELINE_THRESHOLD * 100
                               else "This week" if row['risk_pct'] >= MEDIUM_THRESHOLD * 100
                               else "Routine")
                    cells = [
                        (str(row['patient_id']), 30, _TEXT),
                        (f"{row['risk_pct']:.1f}%", 20, _RED),
                        (f"{row['Age']:.0f}", 12, _TEXT),
                        (f"{row['Cd4AtStart']:.0f}", 18, _TEXT),
                        (f"Stage {row['stage_start_num']:.0f}", 16, _TEXT),
                        ("Yes" if row['had_interruption'] > 0.5 else "No", 18, _TEXT),
                        (str(row['top_driver_short'])[:22], 46, _AMBER),
                        (urgency, 26, _AMBER),
                    ]
                    pdf.set_x(15)
                    for val, width, vc in cells:
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*vc)
                        pdf.cell(width, 9, val)
                    pdf.ln()
                    pdf.set_draw_color(*_CARD); pdf.set_line_width(0.2)
                    pdf.line(15, pdf.get_y(), 195, pdf.get_y())

                # ── SHAP EXPLAINABILITY — TOP PATIENT ────────────────
                try:
                    pdf.add_page()
                    _inner_header(pdf, 'PATIENT RISK EXPLANATION — SHAP ANALYSIS')
                    pdf.set_y(24); _section_title(pdf, 'PATIENT RISK EXPLANATION — SHAP ANALYSIS')

                    pdf.set_font('Helvetica', 'I', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15)
                    pdf.multi_cell(180, 8, _s(
                        'SHAP (SHapley Additive exPlanations) shows exactly which clinical factors '
                        'drove the risk score for the highest-risk patient. Red bars increase risk. '
                        'Green bars reduce risk. Each value is the precise contribution to the final score.'
                    ))
                    pdf.ln(4)

                    # Get top patient
                    top_pat = df_rep.nlargest(1, 'risk_pct').iloc[0]
                    top_pos = df_rep.index.get_loc(df_rep['risk_pct'].idxmax())
                    top_sv, sv_ok = compute_shap_single(X_rep[top_pos])
                    top_sv_arr = np.array(top_sv).flatten()

                    # Patient summary box
                    pdf.set_fill_color(*_CARD)
                    pdf.rect(15, pdf.get_y(), 180, 18, 'F')
                    y_box = pdf.get_y() + 4
                    pdf.set_xy(20, y_box)
                    pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_RED)
                    pdf.cell(50, 9, _s(f"{top_pat['risk_pct']:.1f}% — HIGH RISK"))
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                    pdf.cell(50, 9, _s(f"Patient: {top_pat['patient_id']}"))
                    pdf.cell(40, 9, _s(f"Age: {top_pat['Age']:.0f}"))
                    pdf.cell(40, 9, _s(f"CD4: {top_pat['Cd4AtStart']:.0f} cells/uL"))
                    pdf.ln(14)

                    # Build SHAP waterfall chart
                    sv_order = list(np.argsort(np.abs(top_sv_arr)))
                    sv_vals  = [float(top_sv_arr[i]) for i in sv_order if i < len(FEATURES)]
                    sv_names = []
                    for i in sv_order:
                        if i < len(FEATURES):
                            feat = FEATURES[i]
                            val  = float(top_pat[feat]) if feat in top_pat.index else 0.0
                            label = FEAT_LABELS.get(feat, feat)
                            sv_names.append(f"{label} = {val:.1f}")
                    sv_colors = ['#f85149' if v > 0 else '#3fb950' for v in sv_vals]

                    # Use short labels (no value) — value shown in bar annotation
                    sv_short_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i])
                                      for i in sv_order if i < len(FEATURES)]

                    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0d1117')
                    ax.set_facecolor('#0d1117')
                    bars = ax.barh(range(len(sv_short_names)), sv_vals,
                                   color=sv_colors, height=0.6,
                                   edgecolor='#161b22', linewidth=0.3)
                    x_range = max(abs(v) for v in sv_vals) if sv_vals else 0.1
                    min_bar_for_inline = x_range * 0.20
                    for i, (bar, v) in enumerate(zip(bars, sv_vals)):
                        offset = x_range * 0.03
                        abs_v = abs(v)
                        if abs_v >= min_bar_for_inline:
                            # Bar is long enough — annotate just beyond the bar end
                            x_pos = v + offset if v >= 0 else v - offset
                            ha = 'left' if v >= 0 else 'right'
                        else:
                            # Short bar — place annotation to right of zero line always
                            x_pos = x_range * 0.04
                            ha = 'left'
                        ax.text(x_pos, i, f'{v:+.4f}', va='center', ha=ha,
                                fontsize=10, color='#e6edf3', fontweight='bold')
                    ax.set_yticks(range(len(sv_short_names)))
                    ax.set_yticklabels(sv_short_names, fontsize=11, color='#cdd9e5')
                    ax.axvline(0, color='#8b949e', lw=1.5)
                    ax.set_xlabel('SHAP Value — contribution to risk score',
                                  color='#8b949e', fontsize=11)
                    ax.tick_params(colors='#8b949e', labelsize=10)
                    for sp in ax.spines.values():
                        sp.set_color('#21262d')
                    cohort_baseline = df_rep['risk_pct'].mean()
                    ax.set_title(
                        f"SHAP Explanation: {top_pat['patient_id']} | "
                        f"Risk: {top_pat['risk_pct']:.1f}% (HIGH) | "
                        f"Cohort baseline: {cohort_baseline:.1f}%",
                        color='#e6edf3', fontsize=12, pad=12, fontweight='bold'
                    )
                    plt.tight_layout(pad=1.5)

                    # Save to buffer and embed in PDF
                    img_buf = BytesIO()
                    fig.savefig(img_buf, format='png', dpi=200,
                                bbox_inches='tight', facecolor='#0d1117')
                    plt.close(fig)
                    img_buf.seek(0)

                    chart_y = pdf.get_y()
                    pdf.image(img_buf, x=15, y=chart_y, w=180)
                    pdf.set_y(chart_y + 125)
                    pdf.ln(4)

                    # Top 3 drivers as text summary below chart
                    sv_desc_order = list(reversed(np.argsort(np.abs(top_sv_arr))))
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_CYAN)
                    pdf.set_x(15); pdf.cell(180, 9, _s('Top 3 Risk Drivers for this Patient:'), ln=True)
                    for rank, feat_idx in enumerate(sv_desc_order[:3], 1):
                        if feat_idx >= len(FEATURES):
                            continue
                        feat = FEATURES[feat_idx]
                        sv_val = float(top_sv_arr[feat_idx])
                        direction = 'increases risk' if sv_val > 0 else 'reduces risk'
                        label = FEAT_LABELS.get(feat, feat)
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                        pdf.set_x(20)
                        pdf.cell(6, 9, _s(f'{rank}.'))
                        pdf.multi_cell(169, 9, _s(
                            f"{label} {direction} (SHAP: {sv_val:+.4f})"
                        ))
                    pdf.ln(3)
                    pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15)
                    pdf.multi_cell(180, 7, _s(
                        'SHAP values are model-derived. Interpret alongside clinical judgement. '
                        'Feature contributions reflect patterns learned from the Nigerian discovery cohort '
                        'and may vary across populations and facility types.'
                    ))
                except Exception:
                    pass  # SHAP page is best-effort — never break report generation

                # ── RECOMMENDED ACTIONS ──────────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'RECOMMENDED PROGRAMME ACTIONS')
                pdf.set_y(24); _section_title(pdf, 'RECOMMENDED PROGRAMME ACTIONS')

                action_sections = [
                    ('IMMEDIATE (This Week)', _RED, [
                        f'Contact the {n_high} HIGH risk patients - begin with the top 10 listed above',
                        'Activate peer navigator support for patients with prior interruption history',
                        'Schedule viral load tests for patients showing CD4 decline',
                        f'Prioritise TB-HIV co-treatment coordination for {int(pct_tb/100*n_total)} identified co-infected patients',
                    ]),
                    ('SHORT TERM (1-4 Weeks)', _AMBER, [
                        'Review regimen tolerability for patients with reported side effects',
                        'Site visit to primary health centres - structural quality assessment',
                        'Initiate adherence counselling for all MEDIUM risk patients',
                        'Review diagnosis-to-ART delays and implement fast-track protocols where feasible',
                    ]),
                    ('STRATEGIC (1-3 Months)', _GREEN, [
                        'Consider Differentiated Service Delivery (DSD) model expansion at primary HCs',
                        'Develop outcome-adjusted performance metrics for facility-level monitoring',
                        'Male engagement strategy - flexible hours, community dispensing, peer support',
                        'Apply SmartDaaS risk intelligence framework to PEPFAR MER quarterly reporting',
                    ]),
                ]
                for sec_title, color, actions in action_sections:
                    pdf.set_fill_color(*_CARD)
                    pdf.rect(15, pdf.get_y(), 180, 10, 'F')
                    pdf.set_draw_color(*color); pdf.set_line_width(0.5)
                    pdf.line(15, pdf.get_y(), 15, pdf.get_y()+10)
                    pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*color)
                    pdf.set_x(20); pdf.cell(175, 10, sec_title, ln=True)
                    pdf.ln(2)
                    for action in actions:
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                        pdf.set_x(20); pdf.cell(6, 9, chr(149))
                        pdf.multi_cell(164, 9, _s(action)); pdf.ln(1)
                    pdf.ln(5)

                # ── METHODOLOGY & LIMITATIONS ────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'METHODOLOGY & LIMITATIONS')
                pdf.set_y(24); _section_title(pdf, 'METHODOLOGY & LIMITATIONS')

                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                pdf.set_x(15)
                pdf.multi_cell(180, 9, _s(
                    f"Patient risk scores are generated by a Random Forest classifier predicting poor ART "
                    "adherence recorded at the patient's most recent visit. Trained on 23,144 adults aged "
                    "18-100 from a Nigerian HIV programme discovery cohort (poor-adherence prevalence 3.67%, "
                    "850 patients). "
                    "Internal validation: 10-fold stratified cross-validation at the natural class distribution "
                    "- AUC 0.801 (SD 0.023, Brier 0.032). Oversampling is applied inside each training fold "
                    "only, so no synthetic patient appears in a fold used for scoring. "
                    "Temporal validation: trained on patients initiating ART up to September 2016, tested on "
                    "6,942 later initiators - AUC 0.806 (95% CI 0.774-0.837, Brier 0.027). The internal and "
                    "temporal estimates agree, which is the evidence that the signal generalises rather than "
                    f"reflecting leakage between training and test. "
                    f"At the deployment threshold of {BASELINE_THRESHOLD*100:.1f}%, sensitivity is 27.7%, specificity 97.0% and "
                    "PPV 21.9% against a 3.0% base rate - roughly a sevenfold concentration of risk. Lowering "
                    f"the threshold to {MEDIUM_THRESHOLD*100:.1f}% raises sensitivity to 52.4% while flagging 11.6% of the cohort; "
                    "which point is appropriate is a programme capacity decision. "
                    "The model has not been externally validated on patient-level data from any other health "
                    "system, and the source dataset is a public deposit whose sampling frame, custodian and "
                    "collection methodology are not documented. "
                    "SHAP (SHapley Additive exPlanations) values provide per-patient clinical reasoning. "
                    "Facility intelligence describes associations across recorded facility attributes (level, "
                    "ownership, funding source) in 27,288 patients; the dataset contains no facility identifier, "
                    "so individual facilities cannot be distinguished and these findings are hypothesis-generating "
                    "only. Economic estimates apply a conservative 23% interruption reduction "
                    "assumption for contacted high-risk patients (PEPFAR retention literature). "
                    "Three cost scenarios are used: (1) Conservative — USD 1,850 per averted poor outcome "
                    "(Menzies et al., AIDS 2011, Nigeria-specific PEPFAR data); "
                    "(2) Mid — USD 3,500, reflecting CPI inflation adjustment to 2024 USD (~89% increase since 2009); "
                    "(3) Upper — USD 5,000, reflecting full programme cost of re-engagement including "
                    "viral load testing, tracing costs, and downstream second-line therapy risk "
                    "(Haacker et al., Health Affairs 2022; ACT model estimates). "
                    f"This report uses the conservative estimate (USD {est_avoidable_cost:,}) as the headline figure. "
                    f"Mid-range estimate: USD {est_avoidable_cost_mid:,}. Upper estimate: USD {est_avoidable_cost_upper:,}. "
                    "All findings are illustrative. Prospective validation is required before programmatic application. "
                    "SmartDaaS v1.0 is a decision-support platform for HIV programme intelligence. "
                    "All outputs require review by qualified programme teams prior to operational use. "
                    "Code: github.com/Kchinthala15/smartdaas-hiv-validation"
                ))
                pdf.ln(8)
                # Data source note
                pdf.set_fill_color(*_CARD); pdf.rect(15, pdf.get_y(), 180, 16, 'F')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, pdf.get_y(), 15, pdf.get_y()+16)
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                pdf.set_xy(20, pdf.get_y()+2); pdf.cell(170, 8, 'DATA SOURCE')
                pdf.set_font('Helvetica', '', 11); pdf.set_text_color(*_TEXT)
                pdf.set_xy(20, pdf.get_y()+6); pdf.cell(170, 8, _s(data_source))

                # Save to bytes
                pdf_bytes = bytes(pdf.output())
                pdf_buffer = BytesIO(pdf_bytes)

                st.success("✓ Report generated successfully!")
                log_report(supabase, n_total, "Executive PDF")
                st.download_button(
                    label="📥 Download Executive PDF Report",
                    data=pdf_buffer,
                    file_name=f"SmartDaaS_Report_{org_name.replace(' ','_')}_{report_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"PDF generation error: {e}")
                st.info("If fpdf2 is not installed, run: pip install fpdf2")

    st.markdown("""<div class="warn-box">
    ⚠️ <strong>Decision-support output.</strong> This report is generated by an AI-powered analytics platform
    and should not replace clinical judgement or programme expertise. All findings require
    validation before operational use.
    </div>""", unsafe_allow_html=True)


    # ═════════════════════════════════════════════════════════════
