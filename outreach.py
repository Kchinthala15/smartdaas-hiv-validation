"""
SmartDaaS v1.0 — Outreach Optimiser
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

Capacity-constrained weekly outreach planning engine.
Converts patient risk scores into a prioritised action plan
fitted to actual staff hours.

Exports:
    render_outreach_optimiser(supabase=None)
    All _oo_* helper functions (internal use)
"""

import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import FEATURES, COST_PER_POOR_OUTCOME, FEAT_LABELS

_OO_CONTACT_MINUTES = {'HIGH': 45, 'MEDIUM': 20, 'LOW': 5}
_OO_MODIFIABLE_FEATURES = {
    'side_effects': 3.0, 'had_interruption': 2.5, 'opp_infection': 2.0,
    'tb_positive': 2.0, 'stage_worsened': 1.5, 'days_to_ART': 1.2,
    'weight_change': 1.1,
}
_OO_BASE_REDUCTION_PROB = 0.23
_OO_COL_HIGH, _OO_COL_MEDIUM, _OO_COL_LOW = '#ff453a', '#ffb300', '#30d158'
_OO_COL_ACCENT, _OO_COL_TEXT, _OO_COL_MUTED = '#00e5ff', '#cdd9e5', '#8b949e'


def _oo_intervention_leverage(row):
    leverage = 1.0
    for feat, weight in _OO_MODIFIABLE_FEATURES.items():
        if feat in row.index:
            val = pd.to_numeric(row[feat], errors='coerce')
            if pd.notna(val) and val > 0.5:
                leverage += (weight - 1.0) * 0.3
    return min(leverage, 3.0)


def _oo_urgency(row):
    urgency = 1.0
    if 'MostRecentCd4Count' in row.index:
        cd4 = pd.to_numeric(row['MostRecentCd4Count'], errors='coerce')
        if pd.notna(cd4):
            if cd4 < 100:   urgency += 0.8
            elif cd4 < 200: urgency += 0.5
            elif cd4 < 350: urgency += 0.2
    if 'stage_start_num' in row.index:
        stage = pd.to_numeric(row['stage_start_num'], errors='coerce')
        if pd.notna(stage) and stage >= 3:
            urgency += 0.3
    return min(urgency, 2.0)


def _oo_top_modifiable_factor(row):
    LABELS = {
        'side_effects':     'Side effects -> regimen review',
        'had_interruption': 'Prior interruption -> re-engagement',
        'opp_infection':    'Opportunistic infection -> co-management',
        'tb_positive':      'TB positive -> TB/HIV integration',
        'stage_worsened':   'Clinical deterioration -> urgent review',
        'days_to_ART':      'Late ART start -> linkage support',
        'weight_change':    'Weight loss -> nutritional support',
    }
    for feat in sorted(_OO_MODIFIABLE_FEATURES,
                       key=_OO_MODIFIABLE_FEATURES.get, reverse=True):
        if feat in row.index:
            val = pd.to_numeric(row[feat], errors='coerce')
            if pd.notna(val) and val > 0.5:
                return LABELS.get(feat, feat)
    return 'Standard retention follow-up'


def _oo_contact_type_label(tier):
    return {
        'HIGH':   'Home visit / Urgent counselling (45 min)',
        'MEDIUM': 'Phone call + brief session (20 min)',
        'LOW':    'SMS reminder (5 min)',
    }.get(tier, 'Standard outreach')


def _oo_build_action_plan(df, n_workers, days_available,
                          hours_per_worker_per_day, include_medium=True):
    df = df.copy()
    total_minutes = n_workers * days_available * hours_per_worker_per_day * 60
    df['_leverage'] = df.apply(_oo_intervention_leverage, axis=1)
    df['_urgency']  = df.apply(_oo_urgency, axis=1)
    df['_priority_raw'] = (
        df.get('risk_score', 0.0).astype(float) * df['_leverage'] * df['_urgency']
    )
    p_max, p_min = df['_priority_raw'].max(), df['_priority_raw'].min()
    if p_max > p_min:
        df['priority_score'] = (
            (df['_priority_raw'] - p_min) / (p_max - p_min) * 100).round(1)
    else:
        df['priority_score'] = 50.0

    tiers = ['HIGH', 'MEDIUM'] if include_medium else ['HIGH']
    candidates = df[df['risk_label'].isin(tiers)].copy()
    candidates = candidates.sort_values('priority_score', ascending=False)

    rows, used_minutes, rank = [], 0.0, 1
    for _, row in candidates.iterrows():
        tier = str(row['risk_label'])
        mins = _OO_CONTACT_MINUTES.get(tier, 20)
        if used_minutes + mins > total_minutes:
            break
        if tier == 'HIGH':
            proj = round(float(row.get('risk_score', 0.5)) *
                         _OO_BASE_REDUCTION_PROB * float(row['_leverage']), 3)
        else:
            proj = round(float(row.get('risk_score', 0.3)) *
                         _OO_BASE_REDUCTION_PROB * 0.5 * float(row['_leverage']), 3)
        rows.append({
            'rank': rank,
            'patient_id': row.get('patient_id', f'PT-{rank:04d}'),
            'risk_pct': round(float(row.get('risk_pct',
                              row.get('risk_score', 0) * 100)), 1),
            'risk_label': tier,
            'priority_score': round(float(row['priority_score']), 1),
            'leverage': round(float(row['_leverage']), 2),
            'urgency': round(float(row['_urgency']), 2),
            'contact_type': _oo_contact_type_label(tier),
            'estimated_minutes': mins,
            'top_modifiable_factor': _oo_top_modifiable_factor(row),
            'projected_interruption_reduction': proj,
        })
        used_minutes += mins
        rank += 1
    return pd.DataFrame(rows), used_minutes, total_minutes


def _oo_impact_estimates(plan_df, full_cohort_df, cost_per_poor_outcome=1850.0):
    if plan_df.empty:
        return {}
    n_planned = len(plan_df)
    n_high = int((plan_df['risk_label'] == 'HIGH').sum())
    n_medium = int((plan_df['risk_label'] == 'MEDIUM').sum())
    n_full_high = (int((full_cohort_df['risk_label'] == 'HIGH').sum())
                   if 'risk_label' in full_cohort_df.columns else n_high)
    prevented = plan_df['projected_interruption_reduction'].sum()
    cost = prevented * cost_per_poor_outcome
    coverage = (n_high / n_full_high * 100) if n_full_high > 0 else 0
    mins = plan_df['estimated_minutes'].sum()
    return {
        'n_planned': n_planned, 'n_high': n_high, 'n_medium': n_medium,
        'n_full_high': n_full_high, 'coverage_pct': round(coverage, 1),
        'interruptions_prevented': round(prevented, 1),
        'cost_savings_usd': round(cost, 0),
        'total_mins_planned': int(mins), 'total_hrs_planned': round(mins / 60, 1),
    }


# ── Narrative brief: deterministic fact extraction + template ────────────
def _nb_extract_facts(df_scored):
    facts = {'n_total': int(len(df_scored))}
    n = facts['n_total']
    if n == 0:
        return facts
    labels = df_scored.get('risk_label', pd.Series([], dtype=str))
    facts['n_high'] = int((labels == 'HIGH').sum())
    facts['n_medium'] = int((labels == 'MEDIUM').sum())
    facts['n_low'] = int((labels == 'LOW').sum())
    facts['pct_high'] = round(facts['n_high'] / n * 100, 1)
    facts['pct_medium'] = round(facts['n_medium'] / n * 100, 1)
    facts['pct_low'] = round(facts['n_low'] / n * 100, 1)
    if 'risk_score' in df_scored.columns:
        facts['mean_risk_pct'] = round(
            pd.to_numeric(df_scored['risk_score'], errors='coerce').mean() * 100, 1)
    else:
        facts['mean_risk_pct'] = None

    def _pct(col):
        if col in df_scored.columns:
            v = pd.to_numeric(df_scored[col], errors='coerce')
            return round((v > 0.5).mean() * 100, 1)
        return None
    facts['pct_interruption'] = _pct('had_interruption')
    facts['pct_side_effects'] = _pct('side_effects')
    facts['pct_tb'] = _pct('tb_positive')
    facts['pct_opp_infection'] = _pct('opp_infection')

    if 'stage_start_num' in df_scored.columns:
        stage = pd.to_numeric(df_scored['stage_start_num'], errors='coerce')
        facts['pct_advanced_stage'] = round((stage >= 3).mean() * 100, 1)
    else:
        facts['pct_advanced_stage'] = None
    if 'MostRecentCd4Count' in df_scored.columns:
        cd4 = pd.to_numeric(df_scored['MostRecentCd4Count'], errors='coerce')
        facts['pct_low_cd4'] = round((cd4 < 200).mean() * 100, 1)
    else:
        facts['pct_low_cd4'] = None
    if 'sex_female' in df_scored.columns and 'risk_score' in df_scored.columns:
        sf = pd.to_numeric(df_scored['sex_female'], errors='coerce')
        mr = df_scored.loc[sf < 0.5, 'risk_score'].mean()
        fr = df_scored.loc[sf >= 0.5, 'risk_score'].mean()
        if pd.notna(mr) and pd.notna(fr):
            facts['male_risk_pct'] = round(mr * 100, 1)
            facts['female_risk_pct'] = round(fr * 100, 1)
            facts['sex_gap_pp'] = round((mr - fr) * 100, 1)
        else:
            facts['sex_gap_pp'] = None
    else:
        facts['sex_gap_pp'] = None
    return facts


def _nb_build_template(facts, impact=None, params=None):
    n = facts.get('n_total', 0)
    if n == 0:
        return {'full_text': 'No patient data available to summarise.'}
    pct_high = facts.get('pct_high', 0)
    n_high = facts.get('n_high', 0)
    headline = (f"Of {n:,} patients in this cohort, {n_high:,} ({pct_high}%) are at "
                f"HIGH risk of treatment interruption and warrant proactive outreach.")
    situation = (
        f"This cohort of {n:,} patients comprises {facts.get('n_high',0):,} HIGH-risk "
        f"({pct_high}%), {facts.get('n_medium',0):,} MEDIUM-risk "
        f"({facts.get('pct_medium',0)}%), and {facts.get('n_low',0):,} LOW-risk "
        f"({facts.get('pct_low',0)}%) patients.")
    if facts.get('mean_risk_pct') is not None:
        situation += (f" The mean predicted interruption risk across the cohort is "
                      f"{facts['mean_risk_pct']}%.")

    clauses = []
    if facts.get('pct_interruption') is not None and facts['pct_interruption'] > 15:
        clauses.append(f"{facts['pct_interruption']}% have a documented prior ART "
                       f"interruption -- the strongest single predictor of future "
                       f"disengagement")
    if facts.get('pct_advanced_stage') is not None and facts['pct_advanced_stage'] > 20:
        clauses.append(f"{facts['pct_advanced_stage']}% presented at WHO Stage 3-4, "
                       f"indicating late presentation")
    if facts.get('pct_low_cd4') is not None and facts['pct_low_cd4'] > 25:
        clauses.append(f"{facts['pct_low_cd4']}% have a most-recent CD4 below 200 "
                       f"cells/microlitre")
    if facts.get('pct_tb') is not None and facts['pct_tb'] > 10:
        clauses.append(f"{facts['pct_tb']}% are TB-positive, requiring TB/HIV "
                       f"co-management")
    if facts.get('pct_side_effects') is not None and facts['pct_side_effects'] > 20:
        clauses.append(f"{facts['pct_side_effects']}% report treatment side effects "
                       f"that may be addressable through regimen review")
    if clauses:
        drivers = "Key risk drivers in this cohort: " + "; ".join(clauses) + "."
    else:
        drivers = ("No single risk driver dominates this cohort; risk is distributed "
                   "across multiple factors. Review individual patient profiles for "
                   "specifics.")
    if facts.get('sex_gap_pp') is not None and facts['sex_gap_pp'] >= 3:
        drivers += (f" Male patients average {facts['male_risk_pct']}% risk versus "
                    f"{facts['female_risk_pct']}% for female patients (a "
                    f"{facts['sex_gap_pp']} percentage-point gap), suggesting "
                    f"male-targeted retention efforts may be warranted.")

    if impact and impact.get('n_planned', 0) > 0:
        nw = params.get('n_workers', '-') if params else '-'
        dd = params.get('days', '-') if params else '-'
        action = (f"With your stated capacity of {nw} outreach worker(s) over {dd} "
                  f"day(s), SmartDaaS recommends contacting {impact['n_planned']} "
                  f"patients this week ({impact.get('n_high',0)} HIGH-risk, "
                  f"{impact.get('n_medium',0)} MEDIUM-risk). This covers "
                  f"{impact.get('coverage_pct',0)}% of all HIGH-risk patients within "
                  f"{impact.get('total_hrs_planned','-')} hours of outreach time.")
        if impact.get('interruptions_prevented') is not None:
            action += (f" If all planned contacts are completed, the illustrative "
                       f"estimate is {impact['interruptions_prevented']} interruptions "
                       f"prevented next month, corresponding to approximately "
                       f"${impact.get('cost_savings_usd',0):,.0f} in avoidable "
                       f"programme costs (planning estimate only -- not for funder "
                       f"reporting without local validation).")
    else:
        action = (f"Recommended next step: prioritise the {n_high:,} HIGH-risk patients "
                  f"for proactive outreach. Use the capacity inputs above to fit this "
                  f"list to your available staff and generate a ranked weekly plan.")

    full = (f"{headline}\n\nSITUATION. {situation}\n\nRISK DRIVERS. {drivers}\n\n"
            f"RECOMMENDED ACTION. {action}")
    return {'full_text': full}


def _nb_verify_numbers(source, enhanced):
    import re
    def nums(t):
        return {r.replace(',', '') for r in re.findall(r'\d[\d,]*\.?\d*', t)}
    return len(nums(source) - nums(enhanced)) == 0


def _nb_enhance_with_api(full_text):
    """Optional. Returns (text, was_enhanced). Silent no-op if unavailable."""
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return full_text, False
    try:
        import anthropic
    except Exception:
        return full_text, False
    sys_prompt = (
        "You are rephrasing a pre-computed HIV programme brief. Rephrase into smooth "
        "professional prose. STRICT RULES: (1) do not change, add, remove, or round ANY "
        "number; every figure must appear exactly as given. (2) add no clinical claim or "
        "recommendation not already present. (3) preserve all caveats. (4) max 4 short "
        "paragraphs. (5) no greeting or signature. If unsure, reproduce verbatim.")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1000,
            system=sys_prompt,
            messages=[{"role": "user",
                       "content": "Rephrase this brief:\n\n" + full_text}])
        out = "".join(b.text for b in msg.content
                      if getattr(b, "type", None) == "text").strip()
        if out and _nb_verify_numbers(full_text, out):
            return out, True
        return full_text, False
    except Exception:
        return full_text, False


def _nb_format_html(text):
    import html
    safe = html.escape(text)
    for label in ['SITUATION.', 'RISK DRIVERS.', 'RECOMMENDED ACTION.']:
        safe = safe.replace(label, f'<strong style="color:#00e5ff">{label}</strong>')
    return safe


def _nb_pdf_brief(narrative_text, plan_df, impact, params, org_name=""):
    """Build the brief + action plan as a PDF using fpdf2 (already installed)."""
    from fpdf import FPDF

    def _s(t):
        if not isinstance(t, str):
            t = str(t)
        return (t.replace('\u2014', '-').replace('\u2013', '-')
                 .replace('\u2192', '->').replace('\u00b7', '.')
                 .replace('\u2019', "'").replace('\u2018', "'")
                 .replace('\u201c', '"').replace('\u201d', '"')
                 .replace('\u2026', '...').replace('\u00b5', 'u')
                 .replace('\u00d7', 'x')
                 .encode('latin-1', errors='replace').decode('latin-1'))

    TEAL = (10, 125, 140)
    DARK = (34, 34, 34)
    GREY = (120, 120, 120)

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_left_margin(18)
    pdf.set_right_margin(18)

    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 10, _s("SmartDaaS - Programme Brief"), ln=True)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GREY)
    import datetime as _dt
    sub = (f"{org_name} . " if org_name else "") + \
          _dt.datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')
    pdf.cell(0, 6, _s(sub), ln=True)
    pdf.ln(3)

    blocks = narrative_text.replace('\r\n', '\n').split('\n\n')
    label_map = {'SITUATION.': 'Situation', 'RISK DRIVERS.': 'Risk drivers',
                 'RECOMMENDED ACTION.': 'Recommended action'}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        matched = False
        for raw, nice in label_map.items():
            if block.startswith(raw):
                pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(*TEAL)
                pdf.cell(0, 7, _s(nice), ln=True)
                pdf.set_font('Helvetica', '', 10)
                pdf.set_text_color(*DARK)
                pdf.multi_cell(0, 5.5, _s(block[len(raw):].strip()))
                pdf.ln(1.5)
                matched = True
                break
        if not matched:
            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 6, _s(block))
            pdf.ln(2)

    if plan_df is not None and len(plan_df) > 0:
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 7, _s("Weekly action plan - patient contact list"), ln=True)

        widths = [10, 26, 14, 18, 44, 62]
        headers = ['#', 'Patient ID', 'Risk%', 'Tier', 'Contact type', 'Primary focus']
        pdf.set_font('Helvetica', 'B', 7.5)
        pdf.set_fill_color(*TEAL)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(widths, headers):
            pdf.cell(w, 6, _s(h), border=0, fill=True)
        pdf.ln()

        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(*DARK)
        max_rows = 40
        for i, (_, r) in enumerate(plan_df.head(max_rows).iterrows()):
            fill = (i % 2 == 1)
            if fill:
                pdf.set_fill_color(242, 246, 247)
            cells = [
                str(r['rank']), str(r['patient_id']),
                f"{r['risk_pct']:.0f}", str(r['risk_label']),
                str(r['contact_type']).split(' (')[0],
                str(r['top_modifiable_factor']),
            ]
            for w, c in zip(widths, cells):
                pdf.cell(w, 5.5, _s(c[:40]), border=0, fill=fill)
            pdf.ln()
        if len(plan_df) > max_rows:
            pdf.ln(1)
            pdf.set_font('Helvetica', 'I', 7)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 4, _s(
                f"Showing first {max_rows} of {len(plan_df)} planned contacts. "
                f"Full list available in the CSV export."))

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4, _s(
        "Projected impact figures are illustrative planning estimates only and are not "
        "for funder reporting without local validation. SmartDaaS is a decision-support "
        "tool; all outputs require review by qualified programme and clinical staff "
        "before operational use. Patient data is processed in-session only and is not "
        "stored or transmitted."))

    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)


def render_narrative_block(df_scored, plan_df=None, impact=None,
                           params=None, supabase=None, org_name=""):
    """Embedded programme brief — top of the Outreach Optimiser results."""
    if df_scored is None or len(df_scored) == 0:
        return
    facts = _nb_extract_facts(df_scored)
    template = _nb_build_template(facts, impact, params)
    display_text = template['full_text']
    enhanced = False

    if get_secret("ANTHROPIC_API_KEY") and st.session_state.get('oo_enhance_brief'):
        display_text, enhanced = _nb_enhance_with_api(template['full_text'])

    st.markdown(f"""
<div style="background:#111820;border:1px solid #00e5ff55;border-radius:10px;
    padding:1.5rem 1.75rem;margin:0 0 1rem 0;line-height:1.75;font-size:0.96rem;
    color:#e2eaf3">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;
        color:#00e5ff;text-transform:uppercase;letter-spacing:2.5px;
        margin-bottom:0.75rem">
        Programme Brief - auto-generated - every figure traceable to your data
    </div>
    <div style="white-space:pre-wrap">{_nb_format_html(display_text)}</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        import streamlit.components.v1 as components
        safe_js = (display_text.replace('\\', '\\\\').replace('`', '\\`')
                   .replace('$', '\\$'))
        components.html(f"""
            <button id="cp" style="width:100%;padding:8px 12px;font-size:0.85rem;
                background:#1e2530;color:#00e5ff;border:1px solid #00e5ff55;
                border-radius:6px;cursor:pointer;font-family:sans-serif">
                Copy brief</button>
            <script>
            const b=document.getElementById('cp');
            b.onclick=()=>{{navigator.clipboard.writeText(`{safe_js}`).then(()=>{{
                b.textContent='Copied';setTimeout(()=>b.textContent='Copy brief',1800);}});}};
            </script>""", height=48)
    with c2:
        try:
            pdf_bytes = _nb_pdf_brief(display_text, plan_df, impact or {},
                                      params or {}, org_name)
            st.download_button(
                "Download PDF brief", data=pdf_bytes,
                file_name=f"smartdaas_brief_{datetime.date.today()}.pdf",
                mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.button("PDF unavailable", disabled=True, use_container_width=True,
                      help=f"{type(e).__name__}")
    with c3:
        if get_secret("ANTHROPIC_API_KEY"):
            st.toggle("Smoother wording (AI - figures verified)",
                      key="oo_enhance_brief",
                      help="Rephrases the brief; numbers are verified so the AI "
                           "cannot change any figure.")
        if enhanced:
            st.caption("AI-enhanced wording. All figures verified against the template.")

    if supabase is not None:
        try:
            supabase.table("audit_log").insert({
                "event_at": datetime.datetime.utcnow().isoformat(),
                "event_type": "narrative_brief",
                "n_patients": int(facts.get('n_total', 0)),
                "report_type": "ai_enhanced" if enhanced else "template",
            }).execute()
        except Exception:
            pass


# ── Demo cohort for when no data is scored yet ───────────────────────────
def _oo_demo_cohort(n=200, seed=42):
    rng = np.random.RandomState(seed)
    n_high = int(n * 0.22); n_medium = int(n * 0.35); n_low = n - n_high - n_medium

    def _block(size, rmean, rstd, stage_high=False):
        risk = np.clip(rng.normal(rmean, rstd, size), 0.01, 0.99)
        return pd.DataFrame({
            'patient_id': [f'PT-{i:04d}' for i in range(size)],
            'risk_score': risk, 'risk_pct': (risk * 100).round(1),
            'Age': rng.randint(20, 60, size).astype(float),
            'sex_female': rng.randint(0, 2, size).astype(float),
            'Cd4AtStart': rng.randint(50, 600, size).astype(float),
            'MostRecentCd4Count': rng.randint(50, 700, size).astype(float),
            'CD4_improvement': rng.randint(-200, 400, size).astype(float),
            'stage_start_num': rng.choice([1,2,3,4] if stage_high else [1,2],
                                          size).astype(float),
            'WeightAtStart': rng.randint(45, 90, size).astype(float),
            'weight_change': rng.randint(-10, 10, size).astype(float),
            'BMI_start': rng.uniform(16, 32, size).round(1),
            'days_to_ART': rng.randint(0, 400, size).astype(float),
            'had_interruption': rng.choice([0,1], size, p=[0.6,0.4]).astype(float),
            'opp_infection': rng.choice([0,1], size, p=[0.75,0.25]).astype(float),
            'side_effects': rng.choice([0,1], size, p=[0.65,0.35]).astype(float),
            'tb_positive': rng.choice([0,1], size, p=[0.82,0.18]).astype(float),
            'stage_worsened': rng.choice([0,1], size, p=[0.78,0.22]).astype(float),
        })
    parts = [_block(n_high, 0.82, 0.08, True), _block(n_medium, 0.52, 0.07),
             _block(n_low, 0.18, 0.09)]
    idx = 0
    for p in parts:
        p['patient_id'] = [f'PT-{idx+i:04d}' for i in range(len(p))]; idx += len(p)
    df = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    df['risk_label'] = pd.cut(df['risk_score'], bins=[-0.001, 0.4, 0.7, 1.001],
                              labels=['LOW', 'MEDIUM', 'HIGH']).astype(str)
    return df


def render_outreach_optimiser(supabase=None):
    """Outreach Optimiser page — capacity-constrained weekly action plan."""
    st.markdown("""
<div style="background:#111820;border:1px solid #00e5ff33;border-radius:10px;
    padding:1.5rem 2rem;margin-bottom:1.25rem">
    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#00e5ff;
        text-transform:uppercase;letter-spacing:3px;margin:0 0 0.4rem 0">
        SmartDaaS . New</p>
    <h2 style="color:#e6edf3;font-size:1.4rem;font-weight:700;margin:0 0 0.5rem 0">
        Outreach Optimiser</h2>
    <p style="color:#cdd9e5;font-size:0.95rem;margin:0;line-height:1.6">
        Converts patient risk scores into a <strong>capacity-constrained weekly action
        plan</strong> for your outreach team. Not just who is high-risk -
        <em>exactly who to contact this week, in what order, and why</em>,
        fitted to your actual staff capacity.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#0c2014;border-left:3px solid #30d158;
        padding:0.75rem 1rem;border-radius:0 6px 6px 0;font-size:0.9rem;
        color:#30d158;margin-bottom:1rem">
        <strong>What makes this different:</strong>
        Every other HIV analytics platform stops at "these 500 patients are high risk."
        SmartDaaS asks: <em>how many outreach workers do you actually have, and how many
        hours this week?</em> Then it builds the plan around your real capacity.
    </div>""", unsafe_allow_html=True)

    df_scored = st.session_state.get('df_scored', None)
    using_demo = False
    if df_scored is None or not isinstance(df_scored, pd.DataFrame) or len(df_scored) == 0:
        st.markdown("""<div style="background:#1c1a10;border:1px solid #f0a50066;
            border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem">
            <p style="color:#f0a500;font-weight:700;font-size:0.95rem;margin:0 0 0.5rem 0">
            No programme data loaded yet</p>
            <p style="color:#cdd9e5;font-size:0.9rem;margin:0 0 0.75rem 0;line-height:1.6">
            To build an action plan for your patients, go to <strong>Patient Risk</strong>
            in the sidebar, upload your programme CSV, and run the risk scoring. Then come
            back here - the Outreach Optimiser will automatically use your cohort.</p>
            <p style="color:#8b949e;font-size:0.85rem;margin:0">
            In the meantime, the plan below runs on a <strong>synthetic demo cohort of
            200 patients</strong> so you can explore how the tool works.</p>
            </div>""", unsafe_allow_html=True)
        df_scored = _oo_demo_cohort(200, 42)
        using_demo = True

    if not {'risk_score', 'risk_label'} <= set(df_scored.columns):
        st.markdown("""<div style="background:#2a1010;border:1px solid #ff453a66;
            border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem">
            <p style="color:#ff453a;font-weight:700;font-size:0.95rem;margin:0 0 0.5rem 0">
            Risk scoring has not been run on this data yet</p>
            <p style="color:#cdd9e5;font-size:0.9rem;margin:0;line-height:1.6">
            Your data was uploaded but risk scores have not been calculated. Please go to
            <strong>Patient Risk</strong>, run the scoring, then return here.</p>
            </div>""", unsafe_allow_html=True)
        return

    if 'patient_id' not in df_scored.columns:
        df_scored = df_scored.copy()
        df_scored['patient_id'] = [f'PT-{i:04d}' for i in range(len(df_scored))]
    if 'risk_pct' not in df_scored.columns:
        df_scored = df_scored.copy()
        df_scored['risk_pct'] = (df_scored['risk_score'] * 100).round(1)

    n_total = len(df_scored)
    n_high = int((df_scored['risk_label'] == 'HIGH').sum())
    n_medium = int((df_scored['risk_label'] == 'MEDIUM').sum())
    n_low = int((df_scored['risk_label'] == 'LOW').sum())

    st.markdown('<p class="section-hdr">Cohort Snapshot</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, val, lbl, color in [
        (cols[0], f'{n_total:,}', 'Total patients', '#cdd9e5'),
        (cols[1], f'{n_high:,}', 'HIGH risk', _OO_COL_HIGH),
        (cols[2], f'{n_medium:,}', 'MEDIUM risk', _OO_COL_MEDIUM),
        (cols[3], f'{n_low:,}', 'LOW risk', _OO_COL_LOW)]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" '
                        f'style="color:{color}">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Your Outreach Capacity This Week</p>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">Enter your <em>actual</em> available outreach
        staff and time. SmartDaaS fits the action plan to this capacity.</div>""",
                unsafe_allow_html=True)
    cc = st.columns(4)
    with cc[0]:
        n_workers = st.number_input('Outreach workers available', 1, 50, 2,
                                    help='CHWs, peer navigators, adherence counsellors')
    with cc[1]:
        days_avail = st.number_input('Working days in window', 1, 30, 5,
                                     help='Days until next reporting deadline')
    with cc[2]:
        hrs_per_day = st.number_input('Outreach hours per worker per day',
                                      0.5, 8.0, 3.0, step=0.5)
    with cc[3]:
        include_medium = st.checkbox('Include MEDIUM-risk\n(after HIGH)', value=True)

    cap_hrs = n_workers * days_avail * hrs_per_day
    st.markdown(f'<div style="background:#1e2530;border:1px solid #00e5ff33;'
                f'border-radius:6px;padding:8px 14px;font-size:0.85rem;color:#00e5ff;'
                f'margin:4px 0 12px 0">Total outreach capacity: <strong>{cap_hrs:.1f} '
                f'hours</strong> ({cap_hrs*60:.0f} minutes) across {n_workers} worker(s) '
                f'over {days_avail} day(s)</div>', unsafe_allow_html=True)

    if st.button("Build This Week's Action Plan", type='primary',
                 use_container_width=True):
        st.session_state['oo_run'] = True
        st.session_state['oo_params'] = {
            'n_workers': int(n_workers), 'days': int(days_avail),
            'hrs': float(hrs_per_day), 'include_medium': bool(include_medium)}

    if not st.session_state.get('oo_run'):
        st.markdown('<div style="text-align:center;padding:2rem;color:#484f58;'
                    'font-size:0.9rem">Set your staff capacity above and click Build.'
                    '</div>', unsafe_allow_html=True)
        return

    params = st.session_state.get('oo_params', {
        'n_workers': int(n_workers), 'days': int(days_avail),
        'hrs': float(hrs_per_day), 'include_medium': bool(include_medium)})

    with st.spinner('Building capacity-constrained action plan...'):
        plan_df, used_mins, total_mins = _oo_build_action_plan(
            df_scored, params['n_workers'], params['days'], params['hrs'],
            params['include_medium'])
        impact = _oo_impact_estimates(plan_df, df_scored)

    if plan_df.empty:
        st.warning("No HIGH or MEDIUM risk patients found, or capacity is too low to "
                   "schedule any contacts. Try increasing available hours.")
        return

    # Programme brief at the top of results
    try:
        render_narrative_block(df_scored, plan_df, impact, params, supabase)
    except Exception as _e:
        st.caption(f"Programme brief unavailable ({type(_e).__name__}).")

    st.markdown('<p class="section-hdr">This Week\'s Plan - At a Glance</p>',
                unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#111820;border:1px solid #00e5ff44;
        border-radius:10px;padding:1.25rem 1.75rem;margin-bottom:1rem">
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#00e5ff;
        margin:0 0 0.75rem 0">SmartDaaS recommends contacting
        <strong style="font-size:1.4rem">{impact['n_planned']}</strong> patients this week
        ({impact['n_high']} HIGH . {impact['n_medium']} MEDIUM)</p>
        <p style="color:#cdd9e5;font-size:0.9rem;margin:0;line-height:1.7">
        Out of <strong>{impact['n_full_high']}</strong> total HIGH-risk patients, this plan
        covers <strong>{impact['coverage_pct']}%</strong> within your available capacity of
        <strong>{impact['total_hrs_planned']} hours</strong>. Estimated interruptions
        prevented if all contacts are completed:
        <strong style="color:{_OO_COL_LOW}">{impact['interruptions_prevented']}</strong>
        (illustrative).</p></div>""", unsafe_allow_html=True)

    mc = st.columns(4)
    for col, val, lbl, color in [
        (mc[0], str(impact['n_planned']), 'Patients to contact', _OO_COL_ACCENT),
        (mc[1], f"{impact['coverage_pct']}%", 'HIGH-risk coverage', _OO_COL_MEDIUM),
        (mc[2], str(impact['interruptions_prevented']),
         'Est. interruptions prevented', _OO_COL_LOW),
        (mc[3], f"${impact['cost_savings_usd']:,.0f}", 'Est. cost saved', '#21d4fd')]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" '
                        f'style="color:{color}">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)
    st.caption("Illustrative estimates only. Based on a conservative 23% interruption "
               "reduction per contacted HIGH-risk patient (PEPFAR retention literature) "
               "and $1,850/poor outcome (Menzies et al. 2011). Not for funder reporting "
               "without local validation.")

    # Capacity gauge
    st.markdown('<p class="section-hdr">Capacity Utilisation</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 1.2), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    pct = min(used_mins / total_mins, 1.0) if total_mins > 0 else 0
    ax.barh(0, 1.0, height=0.5, color='#30363d')
    ax.barh(0, pct, height=0.5,
            color=_OO_COL_HIGH if pct > 0.9 else _OO_COL_MEDIUM if pct > 0.7 else _OO_COL_LOW)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8, color=_OO_COL_MUTED)
    ax.set_title(f'Outreach capacity used: {pct*100:.0f}% ({used_mins/60:.1f} of '
                 f'{total_mins/60:.1f} hrs)', color='#e6edf3', fontsize=9, pad=6)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Priority scatter
    st.markdown('<p class="section-hdr">Priority Score Distribution</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="font-size:0.85rem">Priority = risk '
                'probability x intervention leverage (modifiable factors) x clinical '
                'urgency (CD4, WHO stage). Higher = greater outreach ROI.</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    for tier, color in {'HIGH': _OO_COL_HIGH, 'MEDIUM': _OO_COL_MEDIUM}.items():
        sub = plan_df[plan_df['risk_label'] == tier]
        if len(sub) > 0:
            ax.scatter(sub['rank'], sub['priority_score'], c=color, label=tier,
                       s=40, alpha=0.85, zorder=3)
    ax.set_xlabel('Outreach rank', color=_OO_COL_MUTED, fontsize=9)
    ax.set_ylabel('Priority score (0-100)', color=_OO_COL_MUTED, fontsize=9)
    ax.set_title("Patient priority scores - this week's plan", color='#e6edf3',
                 fontsize=10, pad=8)
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor=_OO_COL_TEXT)
    ax.tick_params(colors=_OO_COL_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linewidth=0.5, alpha=0.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Modifiable factors
    st.markdown('<p class="section-hdr">Where to Focus Interventions</p>',
                unsafe_allow_html=True)
    counts = plan_df['top_modifiable_factor'].value_counts().head(6)
    fig, ax = plt.subplots(figsize=(9, 3), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=_OO_COL_ACCENT,
                   alpha=0.8, height=0.55)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=8, color=_OO_COL_TEXT)
    ax.set_xlabel('Patients in plan', color=_OO_COL_MUTED, fontsize=9)
    ax.set_title('Top intervention focus areas this week', color='#e6edf3',
                 fontsize=10, pad=8)
    ax.tick_params(colors=_OO_COL_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.set_xlim(0, counts.values.max() * 1.25)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Action plan table
    st.markdown('<p class="section-hdr">Weekly Action Plan - Patient List</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="font-size:0.85rem">Patients ranked by '
                'priority - highest outreach ROI first. Work down this list in order.'
                '</div>', unsafe_allow_html=True)
    disp = plan_df[['rank', 'patient_id', 'risk_pct', 'risk_label', 'priority_score',
                    'contact_type', 'top_modifiable_factor', 'estimated_minutes']].copy()
    disp['risk_pct'] = disp['risk_pct'].apply(lambda x: f'{x:.1f}')
    disp['priority_score'] = disp['priority_score'].apply(lambda x: f'{x:.1f}')
    disp.columns = ['Rank', 'Patient ID', 'Risk %', 'Tier', 'Priority Score',
                    'Contact Type', 'Primary Focus', 'Est. Minutes']

    def _tier_color(val):
        c = {'HIGH': _OO_COL_HIGH, 'MEDIUM': _OO_COL_MEDIUM,
             'LOW': _OO_COL_LOW}.get(val, _OO_COL_TEXT)
        return f'color: {c}; font-weight: bold'
    try:
        styled = disp.style.map(_tier_color, subset=['Tier'])
    except Exception:
        styled = disp.style.applymap(_tier_color, subset=['Tier'])
    st.dataframe(styled, use_container_width=True, height=420)

    # Exports
    st.markdown('<p class="section-hdr">Export Action Plan</p>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        export = plan_df[['rank', 'patient_id', 'risk_pct', 'risk_label',
                          'priority_score', 'contact_type', 'top_modifiable_factor',
                          'estimated_minutes',
                          'projected_interruption_reduction']].copy()
        export.columns = ['Priority Rank', 'Patient ID', 'Risk Score (%)', 'Risk Tier',
                          'Priority Score', 'Contact Type', 'Primary Intervention Focus',
                          'Est. Time (min)', 'Projected Risk Reduction']
        st.download_button("Download Action Plan (CSV)",
                           data=export.to_csv(index=False).encode(),
                           file_name=f"smartdaas_outreach_plan_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("For CHW team use.")
    with ec2:
        summary = pd.DataFrame([
            ['Generated', datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')],
            ['Workers', params['n_workers']], ['Days', params['days']],
            ['Hrs/worker/day', params['hrs']],
            ['Total capacity (hrs)', impact['total_hrs_planned']],
            ['Patients in plan', impact['n_planned']],
            ['HIGH risk', impact['n_high']], ['MEDIUM risk', impact['n_medium']],
            ['HIGH-risk coverage', f"{impact['coverage_pct']}%"],
            ['Est. interruptions prevented', impact['interruptions_prevented']],
            ['Est. cost savings', f"${impact['cost_savings_usd']:,.0f}"],
        ], columns=['Metric', 'Value'])
        st.download_button("Download Summary (CSV)",
                           data=summary.to_csv(index=False).encode(),
                           file_name=f"smartdaas_outreach_summary_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("For programme director / donor reporting.")

    # Audit log
    if supabase is not None:
        try:
            supabase.table("audit_log").insert({
                "event_at": datetime.datetime.utcnow().isoformat(),
                "event_type": "outreach_plan",
                "n_patients": int(n_total),
                "n_high_risk": int(impact['n_high']),
                "report_type": "demo" if using_demo else "real",
            }).execute()
        except Exception:
            pass

    if using_demo:
        st.markdown("""<div style="background:#231a00;border-left:3px solid #ffb300;
            padding:0.75rem 1rem;border-radius:0 6px 6px 0;font-size:0.85rem;
            color:#ffb300;margin-top:1rem">Demo mode: based on synthetic data. Upload
            your programme data on the Patient Risk page for a real action plan.</div>""",
            unsafe_allow_html=True)


