"""
SmartDaaS — ActionLens Module
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

Self-contained demo module. Converts risk/explanation output into a
frontline action workflow: explanation -> localized action prompt ->
outreach logging -> audit trail.

SYNTHETIC DEMO DATA ONLY. Does not call, import, or modify model.py,
pipeline.py, or outreach.py. Safe to drop into the app as an isolated
page with no effect on existing prediction/validation logic.

Exports:
    render_action_lens()   — single entry point, call from app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

MODEL_VERSION = "SmartDaaS-RF-v0.1-demo"

LANGUAGES = ["English", "French", "Swahili", "Hausa", "Yoruba"]

LANG_NOTE = (
    "Demo language options selected for global health pilot relevance. "
    "Additional languages can be added based on partner site and local "
    "workflow needs."
)

LOCALIZATION_NOTE = (
    "Demo localization only. Final language templates should be reviewed "
    "by local clinical/program teams before field use."
)

EXPLANATION_NOTE = "These are model explanation factors, not causal claims."

DISCLAIMER = (
    "SmartDaaS ActionLens is a decision-support module for pilot evaluation. "
    "It does not provide diagnosis or treatment recommendations. All outreach "
    "decisions remain under the responsibility of qualified clinical/program staff."
)

# ─────────────────────────────────────────────────────────────
# DEMO TRANSLATIONS
# Best-effort placeholder text for a working demo only.
# Not reviewed by native speakers or local clinical teams.
# Must be verified with local partner staff before any real use.
# ─────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "risk_labels": {"High": "High", "Medium": "Medium", "Low": "Low"},
        "reason": "Recently missed a scheduled visit and may have a medication refill gap.",
        "action": "Phone call within 7 days. CHW follow-up if unreachable.",
        "script": ("Hello, we noticed you may be due for care. Do you need help "
                    "rescheduling your appointment or getting your medication refill?"),
        "log": {
            "attempted": "Outreach attempted", "yes": "Yes", "no": "No",
            "method": "Outreach method",
            "methods": ["Phone", "SMS", "CHW visit", "Other"],
            "reached": "Patient reached",
            "outcome": "Outcome",
            "outcomes": ["Appointment rescheduled", "Medication refill arranged",
                         "Counseling needed", "Transport support needed",
                         "Not reachable", "Other"],
            "notes": "Notes", "save": "Save",
        },
    },
    "French": {
        "risk_labels": {"High": "Élevé", "Medium": "Moyen", "Low": "Faible"},
        "reason": "A récemment manqué un rendez-vous prévu et pourrait avoir besoin d'un renouvellement de médicaments.",
        "action": "Appel téléphonique dans les 7 jours. Suivi par un agent de santé communautaire si injoignable.",
        "script": ("Bonjour, nous avons remarqué que vous pourriez avoir besoin de soins. "
                    "Avez-vous besoin d'aide pour reprogrammer votre rendez-vous ou renouveler vos médicaments ?"),
        "log": {
            "attempted": "Contact tenté", "yes": "Oui", "no": "Non",
            "method": "Méthode de contact",
            "methods": ["Téléphone", "SMS", "Visite ASC", "Autre"],
            "reached": "Patient atteint",
            "outcome": "Résultat",
            "outcomes": ["Rendez-vous reprogrammé", "Renouvellement de médicaments organisé",
                         "Besoin de conseil", "Besoin de soutien au transport",
                         "Inaccessible", "Autre"],
            "notes": "Notes", "save": "Enregistrer",
        },
    },
    "Swahili": {
        "risk_labels": {"High": "Juu", "Medium": "Wastani", "Low": "Chini"},
        "reason": "Amekosa ziara iliyopangwa hivi karibuni na huenda akahitaji kujaza dawa.",
        "action": "Piga simu ndani ya siku 7. Mfuatiliaji wa jamii (CHW) afuatilie kama hapatikani.",
        "script": ("Habari, tumegundua kuwa unaweza kuhitaji huduma. Je, unahitaji msaada wa "
                    "kupanga upya miadi yako au kupata dawa zako?"),
        "log": {
            "attempted": "Mawasiliano yamefanyika", "yes": "Ndiyo", "no": "Hapana",
            "method": "Njia ya mawasiliano",
            "methods": ["Simu", "SMS", "Ziara ya CHW", "Nyingine"],
            "reached": "Mgonjwa amefikiwa",
            "outcome": "Matokeo",
            "outcomes": ["Miadi imepangwa upya", "Dawa zimepatikana",
                         "Anahitaji ushauri", "Anahitaji msaada wa usafiri",
                         "Hawezi kufikiwa", "Nyingine"],
            "notes": "Maelezo", "save": "Hifadhi",
        },
    },
    "Hausa": {
        "risk_labels": {"High": "Babba", "Medium": "Matsakaici", "Low": "Karami"},
        "reason": "Ya rasa ziyarar da aka tsara kwanan nan kuma yana iya buƙatar sake cika magani.",
        "action": "Kira ta waya cikin kwanaki 7. Mai bin diddigi na al'umma (CHW) ya bi idan ba a samu shi ba.",
        "script": ("Sannu, mun lura kana iya buƙatar kulawa. Kana buƙatar taimako wajen sake "
                    "tsara alƙawarinka ko samun maganinka?"),
        "log": {
            "attempted": "An yi ƙoƙarin tuntuɓa", "yes": "Eh", "no": "A'a",
            "method": "Hanyar tuntuɓa",
            "methods": ["Waya", "SMS", "Ziyarar CHW", "Wani"],
            "reached": "An kai ga majiyyaci",
            "outcome": "Sakamako",
            "outcomes": ["An sake tsara alƙawari", "An shirya sake cika magani",
                         "Ana buƙatar shawara", "Ana buƙatar taimakon sufuri",
                         "Ba a iya kaiwa gare shi", "Wani"],
            "notes": "Bayanai", "save": "Ajiye",
        },
    },
    "Yoruba": {
        "risk_labels": {"High": "Gíga", "Medium": "Àárín", "Low": "Kéékèèké"},
        "reason": "Ó padánu ìpàdé tí a ṣètò láìpẹ́ ó sì lè nílò àtúnkún oògùn.",
        "action": "Pe lórí fóònù láàrín ọjọ́ 7. Olùrànlọ́wọ́ ìlera àgbègbè (CHW) yóò tẹ̀lé bí kò bá ṣe é rí.",
        "script": ("Ẹ kú àbọ̀, a ṣàkíyèsí pé o lè nílò ìtọ́jú. Ṣe o nílò ìrànlọ́wọ́ láti tún ìpàdé "
                    "rẹ ṣètò tàbí láti gba oògùn rẹ?"),
        "log": {
            "attempted": "Ìgbìyànjú ìfọwọ́kàn", "yes": "Bẹ́ẹ̀ni", "no": "Rárá",
            "method": "Ọ̀nà ìfọwọ́kàn",
            "methods": ["Fóònù", "SMS", "Ìbẹ̀wò CHW", "Mìíràn"],
            "reached": "Aláìsàn dé",
            "outcome": "Àbájáde",
            "outcomes": ["Ìpàdé tún ṣètò", "Ìpèsè oògùn ṣe",
                         "Nílò ìmọ̀ràn", "Nílò ìrànlọ́wọ́ ìrìnnà",
                         "Kò ṣe é rí", "Mìíràn"],
            "notes": "Àkọsílẹ̀", "save": "Fi pamọ́",
        },
    },
}

DRIVER_POOL = [
    "Recent missed appointment pattern",
    "Possible medication refill gap",
    "Prior treatment interruption history",
    "Unstable visit attendance",
    "Long time since last clinical contact",
    "Declining CD4 trend over recent visits",
    "Reported side effects at last visit",
]


# ─────────────────────────────────────────────────────────────
# REAL DATA (from Patient Risk upload) — falls back to synthetic
# ─────────────────────────────────────────────────────────────
def _derive_drivers_from_row(row):
    """Best-effort plain-language drivers from real scored patient features.
    Falls back to a generic driver if expected feature columns aren't present."""
    drivers = []
    try:
        if row.get("had_interruption", 0) == 1:
            drivers.append("Prior treatment interruption history")
        if row.get("stage_worsened", 0) == 1:
            drivers.append("WHO clinical stage worsened")
        if row.get("CD4_improvement", 0) is not None and row.get("CD4_improvement", 0) < 0:
            drivers.append("Declining CD4 trend over recent visits")
        if row.get("opp_infection", 0) == 1:
            drivers.append("Recent opportunistic infection")
        if row.get("side_effects", 0) == 1:
            drivers.append("Reported side effects at last visit")
        if row.get("tb_positive", 0) == 1:
            drivers.append("Active TB co-infection")
        if row.get("days_to_ART", 0) is not None and row.get("days_to_ART", 0) > 90:
            drivers.append("Delayed ART initiation")
    except Exception:
        pass
    if not drivers:
        drivers = ["Elevated model risk score — see SHAP Explainability for full driver breakdown"]
    return drivers[:4]


def _has_real_data():
    df = st.session_state.get("df_scored", None)
    return df is not None and len(df) > 0 and "risk_pct" in df.columns


def _get_real_patients():
    """Build the ActionLens patient list from the real scored cohort (df_scored),
    set by the Patient Risk page after upload. Returns None if no real data exists yet."""
    df = st.session_state.get("df_scored", None)
    if df is None or len(df) == 0 or "risk_pct" not in df.columns:
        return None

    df = df.copy().sort_values("risk_pct", ascending=False).reset_index(drop=True)
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), start=1):
        score = float(r.get("risk_pct", 0))
        level = r.get("risk_label", "LOW")
        level = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}.get(str(level).upper(), "Medium")
        rows.append({
            "patient_id": r.get("patient_id", f"PT-{rank:04d}"),
            "risk_level": level,
            "risk_score": round(score),
            "drivers": _derive_drivers_from_row(r),
            "priority_rank": rank,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA (fallback only — used when no real upload exists yet)
# ─────────────────────────────────────────────────────────────
def _generate_synthetic_patients(n=15, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(1, n + 1):
        score = rng.randint(15, 96)
        level = "High" if score >= 65 else ("Medium" if score >= 35 else "Low")
        n_drivers = 4 if level == "High" else (3 if level == "Medium" else 2)
        drivers = list(rng.choice(DRIVER_POOL, size=n_drivers, replace=False))
        rows.append({
            "patient_id": f"SYN-{1000 + i}",
            "risk_level": level,
            "risk_score": score,
            "drivers": drivers,
        })
    rows.sort(key=lambda r: -r["risk_score"])
    for rank, r in enumerate(rows, start=1):
        r["priority_rank"] = rank
    return rows


def _get_patients():
    """Returns real uploaded+scored patients if available, else a cached synthetic demo set."""
    real = _get_real_patients()
    if real is not None:
        return real
    if "al_patients" not in st.session_state:
        st.session_state["al_patients"] = _generate_synthetic_patients()
    return st.session_state["al_patients"]


def _get_audit_log():
    if "al_audit_log" not in st.session_state:
        st.session_state["al_audit_log"] = []
    return st.session_state["al_audit_log"]


def _get_status(patient_id):
    log = _get_audit_log()
    entries = [e for e in log if e["patient_id"] == patient_id]
    if not entries:
        return "Not yet contacted"
    last = entries[-1]
    return last["outcome"] if last["outreach_attempted"] == "Yes" else "Attempted, not reached"


# ─────────────────────────────────────────────────────────────
# BADGES / SHARED UI
# ─────────────────────────────────────────────────────────────
def _badges():
    real = _has_real_data()
    data_badge = ('<span class="version-tag" style="border-color:#30d15855;color:#30d158">'
                  'Live uploaded cohort</span>' if real else
                  '<span class="version-tag">Synthetic demo data</span>')
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 14px 0">'
        f'{data_badge}'
        '<span class="version-tag">Decision support only</span>'
        '<span class="version-tag">Human review required</span>'
        '<span class="version-tag">Pilot-ready workflow</span>'
        '</div>', unsafe_allow_html=True)


def _risk_css_class(level):
    return {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}[level]


# ─────────────────────────────────────────────────────────────
# LIST VIEW
# ─────────────────────────────────────────────────────────────
def _render_list():
    st.markdown('<p class="section-hdr">ActionLens — Weekly Outreach List</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">ActionLens helps translate explainable risk into '
        'frontline outreach action.</div>', unsafe_allow_html=True)
    _badges()

    patients = _get_patients()

    hdr = st.columns([1.1, 0.8, 0.9, 0.9, 2.6, 1.8, 1.5, 1.1])
    for col, label in zip(hdr, ["Patient ID", "Rank", "Risk", "Score",
                                 "Main drivers", "Suggested action",
                                 "Status", ""]):
        col.markdown(f"**{label}**")

    for p in patients:
        c = st.columns([1.1, 0.8, 0.9, 0.9, 2.6, 1.8, 1.5, 1.1])
        c[0].write(p["patient_id"])
        c[1].write(p["priority_rank"])
        c[2].markdown(f'<span class="{_risk_css_class(p["risk_level"])}" '
                       f'style="padding:2px 8px;border-radius:6px;font-size:0.8rem">'
                       f'{p["risk_level"]}</span>', unsafe_allow_html=True)
        c[3].write(f'{p["risk_score"]}%')
        c[4].write(", ".join(p["drivers"][:2]) + ("…" if len(p["drivers"]) > 2 else ""))
        c[5].write("Phone call within 7 days")
        c[6].write(_get_status(p["patient_id"]))
        if c[7].button("Open", key=f"open_{p['patient_id']}", use_container_width=True):
            st.session_state["al_open_patient"] = p["patient_id"]
            st.rerun()

    st.markdown(f'<div style="font-size:0.7rem;color:#6e7b8a;margin-top:10px">{DISCLAIMER}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DETAIL VIEW
# ─────────────────────────────────────────────────────────────
def _render_detail(patient_id):
    patients = _get_patients()
    p = next(pt for pt in patients if pt["patient_id"] == patient_id)

    if st.button("← Back to outreach list"):
        st.session_state["al_open_patient"] = None
        st.rerun()

    st.markdown('<p class="section-hdr">ActionLens — Patient Detail</p>', unsafe_allow_html=True)
    _badges()

    lang_key = f"al_lang_{patient_id}"
    if lang_key not in st.session_state:
        st.session_state[lang_key] = "English"

    # ── Card 1: Risk Summary ──────────────────────────────────
    st.markdown("#### Risk Summary")
    mc = st.columns(4)
    mc[0].markdown(f'<div class="metric-box"><span class="metric-val">{p["patient_id"]}</span>'
                    f'<span class="metric-lbl">Patient ID</span></div>', unsafe_allow_html=True)
    mc[1].markdown(f'<div class="metric-box"><span class="metric-val">{p["risk_level"]}</span>'
                    f'<span class="metric-lbl">Risk level</span></div>', unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-box"><span class="metric-val">{p["risk_score"]}%</span>'
                    f'<span class="metric-lbl">Risk score</span></div>', unsafe_allow_html=True)
    mc[3].markdown(f'<div class="metric-box"><span class="metric-val">#{p["priority_rank"]}</span>'
                    f'<span class="metric-lbl">Priority rank</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.75rem;color:#6e7b8a;margin-top:4px">'
                f'Model version: {MODEL_VERSION}</div>', unsafe_allow_html=True)

    # ── Card 2: Explanation ───────────────────────────────────
    st.markdown("#### Explanation")
    for d in p["drivers"]:
        st.markdown(f"- {d}")
    st.markdown(f'<div class="info-box">{EXPLANATION_NOTE}</div>', unsafe_allow_html=True)

    # ── Card 3: Frontline Action Prompt ───────────────────────
    st.markdown("#### Frontline Action Prompt")
    driver_text = "; ".join(d.lower() for d in p["drivers"][:2])
    st.markdown(
        f'<div class="warn-box">This patient may need follow-up because of: '
        f'{driver_text}. Please check whether they need help with appointment '
        f'rescheduling, medication refill, transport, or counseling support.</div>',
        unsafe_allow_html=True)

    # ── Card 4: Suggested Outreach ────────────────────────────
    st.markdown("#### Suggested Outreach")
    st.markdown("- **Recommended next step:** Phone call within 7 days")
    st.markdown("- **Backup action:** CHW follow-up if unreachable")
    st.markdown("- **What to check:** appointment, refill, transport, counseling need")

    # ── Card 5: Localization ──────────────────────────────────
    st.markdown("#### Localization")
    lang = st.selectbox("Language", LANGUAGES, key=lang_key)
    t = TRANSLATIONS[lang]
    st.markdown(f'<div class="template-box">'
                f'<strong>{t["risk_labels"][p["risk_level"]]}</strong><br><br>'
                f'{t["reason"]}<br><br>'
                f'<em>{t["action"]}</em><br><br>'
                f'<span style="color:#00e5ff">"{t["script"]}"</span>'
                f'</div>', unsafe_allow_html=True)
    st.caption(LANG_NOTE)
    st.caption(LOCALIZATION_NOTE)

    # ── Outcome logging ────────────────────────────────────────
    st.markdown("#### Log Outreach Outcome")
    log_t = t["log"]
    with st.form(key=f"log_form_{patient_id}"):
        attempted = st.radio(log_t["attempted"], [log_t["yes"], log_t["no"]], horizontal=True)
        method = st.selectbox(log_t["method"], log_t["methods"])
        reached = st.radio(log_t["reached"], [log_t["yes"], log_t["no"]], horizontal=True)
        outcome = st.selectbox(log_t["outcome"], log_t["outcomes"])
        notes = st.text_area(log_t["notes"])
        submitted = st.form_submit_button(log_t["save"])

    if submitted:
        entry = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "user_role": "Outreach worker (demo)",
            "patient_id": patient_id,
            "risk_category_shown": p["risk_level"],
            "recommended_action_shown": "Phone call within 7 days",
            "language": lang,
            "outreach_attempted": "Yes" if attempted == log_t["yes"] else "No",
            "outcome": outcome if reached == log_t["yes"] else "Not reachable",
            "notes": notes,
            "model_version": MODEL_VERSION,
        }
        _get_audit_log().append(entry)
        st.success("Outcome saved.")
        st.rerun()

    # ── Audit trail for this patient ───────────────────────────
    entries = [e for e in _get_audit_log() if e["patient_id"] == patient_id]
    if entries:
        st.markdown("#### Audit Trail")
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

    st.markdown(f'<div style="font-size:0.7rem;color:#6e7b8a;margin-top:14px">{DISCLAIMER}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def render_action_lens():
    """Single entry point. Call this from app.py's ActionLens page route."""
    st.markdown(
        '<div class="info-box">SmartDaaS does not stop at risk prediction. '
        'ActionLens helps care teams understand why a patient was flagged, '
        'what to check next, and how to document the follow-up.</div>',
        unsafe_allow_html=True)

    if _has_real_data():
        st.markdown(
            '<div class="success-box">Showing your uploaded cohort from Patient Risk, '
            'ranked by risk score. Outreach logging below writes to this session only.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warn-box">No cohort uploaded yet — showing a synthetic demo cohort. '
            'Upload a file on the Patient Risk page to see ActionLens work on your own data.</div>',
            unsafe_allow_html=True)

    if "al_open_patient" not in st.session_state:
        st.session_state["al_open_patient"] = None

    if st.session_state["al_open_patient"]:
        _render_detail(st.session_state["al_open_patient"])
    else:
        _render_list()
