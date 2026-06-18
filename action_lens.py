"""
SmartDaaS — ActionLens Module
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

ActionLens converts model risk output into frontline outreach action:
risk prediction -> explanation -> localized action prompt -> outreach
logging -> audit trail -> feedback loop.

Design notes (read before editing):
  - Fully self-contained. Uses synthetic demo patients only. Does NOT
    read st.session_state['df_scored'] or any other page's state, and
    does NOT import model.py / call the trained model or SHAP. This is
    deliberate: ActionLens must keep working even if the trained model
    pickle files are missing or fail to load elsewhere in the app.
  - Does NOT call any external translation API. All localized strings
    below are hardcoded demo text, clearly flagged as such in the UI.
  - Reuses existing platform CSS classes from styles.py (section-hdr,
    risk-high/medium/low, metric-box, info-box/warn-box/success-box,
    template-box, version-tag) instead of adding new CSS, so there is
    nothing here that can visually clash with the rest of the app.
  - Audit logging mirrors the pattern already used by reports.py:
    receives `log_event` as a callable passed in from app.py (avoids a
    circular import), with a no-op fallback if it's not provided.

Exports:
    render_action_lens(supabase=None, log_event=None)
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st

MODEL_VERSION_LABEL = "SmartDaaS-RF-v0.1-demo"

DEMO_LANGUAGES = ["English", "French", "Swahili", "Hausa", "Yoruba", "Igbo", "Nigerian Pidgin"]

LANGUAGE_NOTE = (
    "Demo languages are illustrative examples selected to show country/site-level "
    "localization. Final language options should be determined with implementation "
    "partners and frontline teams."
)

LOCALIZATION_DISCLAIMER = (
    "Demo localization only. Final language templates should be reviewed "
    "by local clinical/program teams before field use."
)

ACTIONLENS_DISCLAIMER = (
    "SmartDaaS ActionLens is a decision-support module for pilot "
    "evaluation. It does not provide diagnosis or treatment "
    "recommendations. All outreach decisions remain under the "
    "responsibility of qualified clinical/program staff."
)

POSITIONING_LINE = (
    "ActionLens helps translate explainable risk into frontline outreach action."
)

POSITIONING_STRONG = (
    "SmartDaaS does not stop at risk prediction. ActionLens helps care teams "
    "understand why a patient was flagged, what to check next, and how to "
    "document the follow-up."
)

# ─────────────────────────────────────────────────────────────
# EXPLANATION FACTOR POOL (English only — not localized; see
# constants.py FEAT_LABELS for the underlying model vocabulary these
# loosely echo). These are plain-language translations of risk drivers,
# not raw SHAP output.
# ─────────────────────────────────────────────────────────────
DRIVER_POOL = [
    "Recent missed appointment pattern",
    "Possible medication refill gap",
    "Prior treatment interruption history",
    "Unstable visit attendance",
    "Long time since last clinical contact",
    "Limited weight gain since starting treatment",
    "WHO clinical stage worsened since enrollment",
    "Reported side effects affecting adherence",
    "Transport barrier flagged at last visit",
    "Slow CD4 recovery trend since ART start",
]

# ─────────────────────────────────────────────────────────────
# LOCALIZED CONTENT — hardcoded demo strings only.
# Localizes ONLY the frontline-facing layer: risk word, plain-language
# reason, suggested next step / backup / script, outcome-logging labels.
# Does NOT localize: explanation factors, model version, patient ID.
# ─────────────────────────────────────────────────────────────
L10N = {
    "English": {
        "risk_word": {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"},
        "reason": {
            "HIGH": "This patient may need follow-up because of a recent missed visit and a possible gap in their medication refill.",
            "MEDIUM": "This patient has shown some inconsistency in visit attendance and may benefit from a check-in call.",
            "LOW": "This patient is currently on a stable care pathway. Routine follow-up at the next scheduled visit is sufficient.",
        },
        "next_step": {
            "HIGH": "Phone call within 7 days",
            "MEDIUM": "SMS reminder, phone call within 2 weeks",
            "LOW": "Routine follow-up at next visit",
        },
        "backup": {
            "HIGH": "CHW follow-up if unreachable by phone",
            "MEDIUM": "CHW follow-up if no response to SMS",
            "LOW": "None required",
        },
        "script": {
            "HIGH": "Hello, we noticed you may be due for care. Do you need help rescheduling your appointment or getting your medication refill?",
            "MEDIUM": "Hello, just checking in - is there anything you need help with for your next appointment or your medication?",
            "LOW": "Hello, this is a routine check-in. We look forward to seeing you at your next scheduled visit.",
        },
        "outcome": {
            "attempted_q": "Outreach attempted?", "method_q": "Outreach method",
            "reached_q": "Patient reached?", "outcome_q": "Outcome",
            "notes_q": "Notes", "save_btn": "Save outreach record",
            "saved_msg": "Outreach record saved.",
            "yes": "Yes", "no": "No",
            "methods": ["Phone", "SMS", "CHW visit", "Other"],
            "outcomes": ["Appointment rescheduled", "Medication refill arranged",
                         "Counseling needed", "Transport support needed",
                         "Not reachable", "Other"],
        },
    },
    "French": {
        "risk_word": {"HIGH": "Élevé", "MEDIUM": "Moyen", "LOW": "Faible"},
        "reason": {
            "HIGH": "Ce patient pourrait avoir besoin d'un suivi en raison d'une visite manquée récemment et d'un possible retard de renouvellement de son traitement.",
            "MEDIUM": "Ce patient a montré une certaine irrégularité dans la fréquentation des visites et pourrait bénéficier d'un appel de suivi.",
            "LOW": "Ce patient suit actuellement un parcours de soins stable. Un suivi de routine lors de la prochaine visite prévue est suffisant.",
        },
        "next_step": {
            "HIGH": "Appel téléphonique dans les 7 jours",
            "MEDIUM": "Rappel par SMS, appel téléphonique dans les 2 semaines",
            "LOW": "Suivi de routine à la prochaine visite",
        },
        "backup": {
            "HIGH": "Visite d'un agent de santé communautaire si injoignable",
            "MEDIUM": "Visite d'un agent de santé communautaire en l'absence de réponse",
            "LOW": "Aucune action supplémentaire requise",
        },
        "script": {
            "HIGH": "Bonjour, nous avons remarqué que vous pourriez avoir besoin de soins. Avez-vous besoin d'aide pour reprogrammer votre rendez-vous ou renouveler votre traitement ?",
            "MEDIUM": "Bonjour, nous prenons simplement de vos nouvelles - avez-vous besoin d'aide pour votre prochain rendez-vous ou votre traitement ?",
            "LOW": "Bonjour, ceci est un suivi de routine. Nous nous réjouissons de vous voir à votre prochaine visite prévue.",
        },
        "outcome": {
            "attempted_q": "Contact tenté ?", "method_q": "Méthode de contact",
            "reached_q": "Patient atteint ?", "outcome_q": "Résultat",
            "notes_q": "Remarques", "save_btn": "Enregistrer le suivi",
            "saved_msg": "Suivi enregistré.",
            "yes": "Oui", "no": "Non",
            "methods": ["Téléphone", "SMS", "Visite ASC", "Autre"],
            "outcomes": ["Rendez-vous reprogrammé", "Renouvellement organisé",
                         "Counseling nécessaire", "Aide au transport nécessaire",
                         "Injoignable", "Autre"],
        },
    },
    "Swahili": {
        "risk_word": {"HIGH": "Juu", "MEDIUM": "Wastani", "LOW": "Chini"},
        "reason": {
            "HIGH": "Mgonjwa huyu anaweza kuhitaji ufuatiliaji kwa sababu ya kukosa ziara ya hivi karibuni na uwezekano wa pengo katika kuongeza dawa.",
            "MEDIUM": "Mgonjwa huyu ameonyesha mwenendo usio thabiti wa kuhudhuria ziara na anaweza kufaidika na simu ya kuangalia hali yake.",
            "LOW": "Mgonjwa huyu kwa sasa anaendelea vizuri na matibabu. Ufuatiliaji wa kawaida kwenye ziara inayofuata unatosha.",
        },
        "next_step": {
            "HIGH": "Piga simu ndani ya siku 7",
            "MEDIUM": "Tuma SMS, piga simu ndani ya wiki 2",
            "LOW": "Ufuatiliaji wa kawaida kwenye ziara inayofuata",
        },
        "backup": {
            "HIGH": "Ziara ya mhudumu wa afya ngazi ya jamii kama hapatikani kwa simu",
            "MEDIUM": "Ziara ya mhudumu wa afya ngazi ya jamii kama hajajibu SMS",
            "LOW": "Hakuna hatua zaidi inayohitajika",
        },
        "script": {
            "HIGH": "Habari, tumegundua unaweza kuwa unahitaji huduma. Je, unahitaji msaada wa kupanga upya miadi yako au kupata dawa zako?",
            "MEDIUM": "Habari, tunaangalia hali yako tu - unahitaji msaada wowote kwa miadi yako inayofuata au dawa zako?",
            "LOW": "Habari, huu ni ufuatiliaji wa kawaida. Tunatarajia kukuona kwenye ziara yako inayofuata.",
        },
        "outcome": {
            "attempted_q": "Je, mawasiliano yalifanyika?", "method_q": "Njia ya mawasiliano",
            "reached_q": "Mgonjwa amefikiwa?", "outcome_q": "Matokeo",
            "notes_q": "Maelezo", "save_btn": "Hifadhi rekodi ya ufuatiliaji",
            "saved_msg": "Rekodi ya ufuatiliaji imehifadhiwa.",
            "yes": "Ndiyo", "no": "Hapana",
            "methods": ["Simu", "SMS", "Ziara ya mhudumu wa afya", "Nyingine"],
            "outcomes": ["Miadi imepangwa upya", "Dawa zimepangiwa",
                         "Ushauri unahitajika", "Msaada wa usafiri unahitajika",
                         "Hapatikani", "Nyingine"],
        },
    },
    "Hausa": {
        "risk_word": {"HIGH": "Babba", "MEDIUM": "Matsakaici", "LOW": "Kasa"},
        "reason": {
            "HIGH": "Wannan majiyyaci na iya bukatar bibiya saboda ya rasa ziyarar da aka tsara masa kwanan nan kuma akwai yiwuwar gibi a sake samun magani.",
            "MEDIUM": "Wannan majiyyaci ya nuna rashin daidaito wajen zuwa ziyarori kuma zai amfana da kira na duba lafiyarsa.",
            "LOW": "Wannan majiyyaci yana kan turbar kulawa mai kwanciyar hankali a yanzu. Bibiya ta yau da kullum a ziyarar da ke tafe ta isa.",
        },
        "next_step": {
            "HIGH": "Kira ta waya cikin kwanaki 7",
            "MEDIUM": "Tunatarwa ta SMS, kira ta waya cikin makonni 2",
            "LOW": "Bibiya ta yau da kullum a ziyarar da ke tafe",
        },
        "backup": {
            "HIGH": "Ziyarar ma'aikacin lafiya na al'umma idan ba a samu shi ta waya ba",
            "MEDIUM": "Ziyarar ma'aikacin lafiya na al'umma idan babu amsa ga SMS",
            "LOW": "Babu wani mataki da ake bukata",
        },
        "script": {
            "HIGH": "Sannu, mun lura kana iya bukatar kulawa. Kana bukatar taimako wajen sake tsara ziyararka ko samun maganinka?",
            "MEDIUM": "Sannu, muna duba lafiyarka ne kawai - akwai wani taimako da kake bukata game da ziyararka ko maganinka?",
            "LOW": "Sannu, wannan bibiya ce ta yau da kullum. Muna jiran ganin ka a ziyararka mai zuwa.",
        },
        "outcome": {
            "attempted_q": "An yi ƙoƙarin tuntuɓa?", "method_q": "Hanyar tuntuɓa",
            "reached_q": "An samu majiyyaci?", "outcome_q": "Sakamako",
            "notes_q": "Bayani", "save_btn": "Adana rikodin bibiya",
            "saved_msg": "An adana rikodin bibiya.",
            "yes": "Eh", "no": "A'a",
            "methods": ["Waya", "SMS", "Ziyarar ma'aikacin lafiya", "Wani"],
            "outcomes": ["An sake tsara ziyara", "An shirya samun magani",
                         "Ana bukatar shawara", "Ana bukatar taimakon sufuri",
                         "Ba a samu shi ba", "Wani"],
        },
    },
    "Yoruba": {
        "risk_word": {"HIGH": "Gíga", "MEDIUM": "Àárín", "LOW": "Kéré"},
        "reason": {
            "HIGH": "Aláìsàn yìí lè nílò ìtọ́jú nítorí pé kò lọ sí ìpàdé tó kẹ́yìn àti pé ó lè ní àlàfo nínú gbígba òògùn rẹ̀.",
            "MEDIUM": "Aláìsàn yìí ti fi àìdúróṣinṣin hàn nínú dídé sí ìpàdé, ó sì lè jàǹfààní nínú pípè é láti bá a sọ̀rọ̀.",
            "LOW": "Aláìsàn yìí wà lórí ìtọ́jú tó dúró ṣinṣin lọ́wọ́lọ́wọ́. Ìtọ́jú déédéé ní ìpàdé tó kàn yóò tó.",
        },
        "next_step": {
            "HIGH": "Pe é lórí fóònù láàrin ọjọ́ 7",
            "MEDIUM": "Ránṣẹ́ sí i nípa SMS, pe é lórí fóònù láàrin ọ̀sẹ̀ méjì",
            "LOW": "Ìtọ́jú déédéé ní ìpàdé tó kàn",
        },
        "backup": {
            "HIGH": "Kí òṣìṣẹ́ ìlera àgbègbè lọ bá a bí kò bá ṣe é dé lórí fóònù",
            "MEDIUM": "Kí òṣìṣẹ́ ìlera àgbègbè lọ bá a bí kò bá dáhùn sí SMS",
            "LOW": "Kò sí ìgbésẹ̀ míràn tí a nílò",
        },
        "script": {
            "HIGH": "Báwo, a ṣe àkíyèsí pé ó lè nílò ìtọ́jú. Ṣé o nílò ìrànlọ́wọ́ láti tún ìpàdé rẹ ṣe tàbí láti gba òògùn rẹ?",
            "MEDIUM": "Báwo, à ń bèèrè ìròyìn rẹ nìkan - ṣé o nílò ìrànlọ́wọ́ kankan fún ìpàdé rẹ tó kàn tàbí òògùn rẹ?",
            "LOW": "Báwo, èyí jẹ́ ìbéwò déédéé. À ó fẹ́ràn láti rí ọ ní ìpàdé rẹ tó kàn.",
        },
        "outcome": {
            "attempted_q": "Ṣé a gbìyànjú láti kàn sí i?", "method_q": "Ọ̀nà ìbánisọ̀rọ̀",
            "reached_q": "Ṣé a kàn sí aláìsàn?", "outcome_q": "Àbájáde",
            "notes_q": "Àkọsílẹ̀", "save_btn": "Fi àkọsílẹ̀ ìtọ́jú pamọ́",
            "saved_msg": "Àkọsílẹ̀ ìtọ́jú ti fi pamọ́.",
            "yes": "Bẹ́ẹ̀ni", "no": "Rárá",
            "methods": ["Fóònù", "SMS", "Ìbẹ̀wò òṣìṣẹ́ ìlera", "Òmíràn"],
            "outcomes": ["Ìpàdé ti tún ṣe", "Òògùn ti ṣètò",
                         "Ìmọ̀ràn nílò", "Ìrànlọ́wọ́ ìrìnnà nílò",
                         "Kò ṣe é dé", "Òmíràn"],
        },
    },
    "Igbo": {
        "risk_word": {"HIGH": "Elu", "MEDIUM": "Etiti", "LOW": "Ala"},
        "reason": {
            "HIGH": "Onye ọrịa a nwere ike ịchọ nleba anya n'ihi na ọ hapụrụ nleta n'oge na enwere ohere na o nwere oghere n'ịnweta ọgwụ ya ọzọ.",
            "MEDIUM": "Onye ọrịa a egosila enweghị ukwu n'ịbịa nleta ya, ọ ga-aba uru ma a kpọọ ya oku iji lelee ya.",
            "LOW": "Onye ọrịa a nọ n'ụzọ nlekọta ahụike kwụ ọtọ ugbu a. Nleba anya nkịtị na nleta ọzọ ga-ezuru ya.",
        },
        "next_step": {
            "HIGH": "Kpọọ ya oku n'ime ụbọchị asaa",
            "MEDIUM": "Ozi SMS na ịkpọ ya oku n'ime izu abụọ",
            "LOW": "Nleba anya nkịtị na nleta ọzọ",
        },
        "backup": {
            "HIGH": "Ka onye na-elekọta ahụike obodo gaa leta ya ma a pụghị ịkpọ ya oku",
            "MEDIUM": "Ka onye na-elekọta ahụike obodo gaa leta ya ma ọ zaghị SMS",
            "LOW": "Ọ dịghị ihe ọzọ achọrọ",
        },
        "script": {
            "HIGH": "Ndewo, anyị chọpụtara na ị nwere ike chọọ nleba anya. Ị chọrọ enyemaka iji hazigharị oge nleta gị ma ọ bụ inweta ọgwụ gị ọzọ?",
            "MEDIUM": "Ndewo, anyị na-ajụ ka ahụ́ dị gị - ị chọrọ enyemaka ọ bụla maka nleta gị ọzọ ma ọ bụ ọgwụ gị?",
            "LOW": "Ndewo, nke a bụ nleba anya nkịtị. Anyị ga-anabata ịhụ gị na nleta gị ọzọ.",
        },
        "outcome": {
            "attempted_q": "A gbalịrị ịkpọtụrụ ya?", "method_q": "Ụzọ nkwukọrịta",
            "reached_q": "Eruru onye ọrịa?", "outcome_q": "Nsonaazụ",
            "notes_q": "Ndetu", "save_btn": "Chekwaa ndekọ nleba anya",
            "saved_msg": "Edebe ndekọ nleba anya.",
            "yes": "Ee", "no": "Mba",
            "methods": ["Ekwentị", "SMS", "Nleta onye na-elekọta ahụike obodo", "Ọzọ"],
            "outcomes": ["E megharịrị oge nleta", "E hazi inweta ọgwụ ọzọ",
                         "Achọrọ ndụmọdụ", "Achọrọ enyemaka njem",
                         "Apụghị iru ya", "Ọzọ"],
        },
    },
    "Nigerian Pidgin": {
        "risk_word": {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"},
        "reason": {
            "HIGH": "Dis patient fit need follow-up because e miss one appointment recently and e fit dey get gap for im medication refill.",
            "MEDIUM": "Dis patient don show small inconsistency for coming to im appointments, e fit helep if dem call am check up.",
            "LOW": "Dis patient dey waka well well for im treatment now. Normal follow-up for di next appointment go do.",
        },
        "next_step": {
            "HIGH": "Call am within 7 days",
            "MEDIUM": "Send SMS reminder, call am within 2 weeks",
            "LOW": "Normal follow-up for next appointment",
        },
        "backup": {
            "HIGH": "CHW go visit am if dem no fit reach am for phone",
            "MEDIUM": "CHW go visit am if e no reply SMS",
            "LOW": "No extra action needed",
        },
        "script": {
            "HIGH": "Hello, we notice say you fit need care. You need helep to rearrange your appointment or to get your medication refill?",
            "MEDIUM": "Hello, we just dey check up on you - anything wey you need helep with for your next appointment or your medication?",
            "LOW": "Hello, dis na normal check-in. We dey look forward to see you for your next appointment.",
        },
        "outcome": {
            "attempted_q": "Dem try reach am?", "method_q": "Method wey dem use",
            "reached_q": "Dem reach di patient?", "outcome_q": "Wetin happen",
            "notes_q": "Notes", "save_btn": "Save di outreach record",
            "saved_msg": "Outreach record don save.",
            "yes": "Yes", "no": "No",
            "methods": ["Phone", "SMS", "CHW visit", "Other"],
            "outcomes": ["Dem rearrange appointment", "Dem arrange medication refill",
                         "Counseling needed", "Transport support needed",
                         "Dem no fit reach am", "Other"],
        },
    },
}


# ─────────────────────────────────────────────────────────────
# SYNTHETIC COHORT GENERATOR
# Deterministic, self-contained. Independent of any data uploaded
# elsewhere in the platform — ActionLens never reads df_scored or any
# other page's session state.
# ─────────────────────────────────────────────────────────────
def _generate_synthetic_cohort(n=16, seed=7):
    rng = np.random.RandomState(seed)

    n_high = max(1, round(n * 0.35))
    n_medium = max(1, round(n * 0.40))
    n_low = max(1, n - n_high - n_medium)
    tiers = (["HIGH"] * n_high) + (["MEDIUM"] * n_medium) + (["LOW"] * n_low)
    rng.shuffle(tiers)

    syn_ids = sorted(rng.choice(np.arange(1000, 2000), size=n, replace=False))
    rank_high = sorted(rng.choice(np.arange(1, 60), size=n_high, replace=False))
    rank_med = sorted(rng.choice(np.arange(60, 180), size=n_medium, replace=False))
    rank_low = sorted(rng.choice(np.arange(180, 320), size=n_low, replace=False))
    rank_cursors = {"HIGH": list(rank_high), "MEDIUM": list(rank_med), "LOW": list(rank_low)}

    rows = []
    for i, tier in enumerate(tiers):
        if tier == "HIGH":
            score = rng.uniform(0.70, 0.95)
            n_drivers = rng.randint(3, 6)
        elif tier == "MEDIUM":
            score = rng.uniform(0.40, 0.69)
            n_drivers = rng.randint(2, 4)
        else:
            score = rng.uniform(0.05, 0.39)
            n_drivers = rng.randint(1, 3)
        driver_idx = rng.choice(len(DRIVER_POOL), size=n_drivers, replace=False)
        drivers = [DRIVER_POOL[j] for j in driver_idx]
        rows.append({
            "patient_id": f"SYN-{syn_ids[i]}",
            "risk_label": tier,
            "risk_score": score,
            "risk_pct": round(score * 100, 1),
            "priority_rank": rank_cursors[tier].pop(0),
            "drivers": drivers,
            "last_outreach_status": "Not yet contacted",
        })
    df = pd.DataFrame(rows).sort_values("priority_rank").reset_index(drop=True)
    return df


def _badge(text):
    return f'<span class="version-tag" style="margin-right:6px;">{text}</span>'


def _risk_box(risk_label, risk_pct, lang="English"):
    css_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[risk_label]
    word = L10N.get(lang, L10N["English"])["risk_word"][risk_label]
    return (f'<div class="{css_class}"><span class="risk-number">{risk_pct:.0f}%</span>'
            f'<br><span class="risk-label">{word}</span></div>')


# ─────────────────────────────────────────────────────────────
# WEEKLY OUTREACH LIST
# ─────────────────────────────────────────────────────────────
def _render_weekly_list(df):
    st.markdown('<p class="section-hdr">ActionLens — Weekly Outreach List</p>',
                unsafe_allow_html=True)
    st.markdown(
        _badge("Synthetic demo data") + _badge("Decision support only") +
        _badge("Human review required") + _badge("Pilot-ready workflow"),
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="info-box">{POSITIONING_LINE}</div>', unsafe_allow_html=True)
    st.caption(f"{len(df)} synthetic demo patients · sorted by priority rank")

    header_cols = st.columns([1.2, 0.7, 0.9, 0.8, 2.6, 1.8, 1.4, 0.8])
    for c, h in zip(header_cols, ["Patient ID", "Rank", "Risk", "Score",
                                   "Main drivers", "Suggested action",
                                   "Last status", ""]):
        c.markdown(f"**{h}**")

    for _, row in df.sort_values("priority_rank").iterrows():
        cols = st.columns([1.2, 0.7, 0.9, 0.8, 2.6, 1.8, 1.4, 0.8])
        cols[0].markdown(row["patient_id"])
        cols[1].markdown(str(row["priority_rank"]))
        cols[2].markdown(L10N["English"]["risk_word"][row["risk_label"]])
        cols[3].markdown(f'{row["risk_pct"]:.0f}%')
        drivers = row["drivers"]
        summary = ", ".join(drivers[:2]) + ("…" if len(drivers) > 2 else "")
        cols[4].markdown(summary)
        cols[5].markdown(L10N["English"]["next_step"][row["risk_label"]])
        cols[6].markdown(row["last_outreach_status"])
        if cols[7].button("Open", key=f"al_open_{row['patient_id']}"):
            st.session_state["al_selected_patient"] = row["patient_id"]
            st.session_state["al_view"] = "detail"
            st.rerun()

    st.markdown(f'<div class="warn-box">{ACTIONLENS_DISCLAIMER}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ACTIONLENS DETAIL VIEW
# ─────────────────────────────────────────────────────────────
def _render_detail_view(df, patient_id, supabase, log_event):
    match = df[df["patient_id"] == patient_id]
    if match.empty:
        st.warning("Patient not found in this demo cohort.")
        if st.button("← Back to weekly outreach list"):
            st.session_state["al_view"] = "list"
            st.rerun()
        return
    row = match.iloc[0]

    if st.button("← Back to weekly outreach list", key="al_back_btn"):
        st.session_state["al_view"] = "list"
        st.rerun()

    st.markdown('<p class="section-hdr">ActionLens</p>', unsafe_allow_html=True)
    st.markdown(POSITIONING_STRONG)

    lang = st.selectbox(
        "Language", DEMO_LANGUAGES,
        index=DEMO_LANGUAGES.index(st.session_state.get("al_language", "English")),
        key="al_language_select",
    )
    st.session_state["al_language"] = lang
    t = L10N[lang]

    # 1. Risk Summary
    st.markdown('<p class="section-hdr">1 · Risk Summary</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_risk_box(row["risk_label"], row["risk_pct"], lang), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><span class="metric-val">{row["priority_rank"]}</span>'
                     f'<span class="metric-lbl">Priority Rank</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><span class="metric-val">{row["patient_id"]}</span>'
                     f'<span class="metric-lbl">Patient ID</span></div>', unsafe_allow_html=True)
    st.markdown(_badge("Decision support only") + _badge("Synthetic demo data"),
                unsafe_allow_html=True)
    st.caption(f"Model version: {MODEL_VERSION_LABEL}")

    # 2. Explanation (NOT localized)
    st.markdown('<p class="section-hdr">2 · Explanation</p>', unsafe_allow_html=True)
    for d in row["drivers"]:
        st.markdown(f"- {d}")
    st.markdown('<div class="info-box">These are model explanation factors, not causal claims.</div>',
                unsafe_allow_html=True)

    # 3. Frontline Action Prompt (localized)
    st.markdown('<p class="section-hdr">3 · Frontline Action Prompt</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="template-box">{t["reason"][row["risk_label"]]}</div>',
                unsafe_allow_html=True)

    # 4. Suggested Outreach (localized)
    st.markdown('<p class="section-hdr">4 · Suggested Outreach</p>', unsafe_allow_html=True)
    st.markdown(f"**Recommended next step:** {t['next_step'][row['risk_label']]}")
    st.markdown(f"**Backup action:** {t['backup'][row['risk_label']]}")
    st.markdown("**What to check:** Appointment · Medication refill · Transport · Counseling need")
    st.markdown(f'<div class="template-box"><em>Suggested script:</em><br>{t["script"][row["risk_label"]]}</div>',
                unsafe_allow_html=True)

    # 5. Localization
    st.markdown('<p class="section-hdr">5 · Localization</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="warn-box">{LOCALIZATION_DISCLAIMER}</div>', unsafe_allow_html=True)
    st.caption(LANGUAGE_NOTE)

    # Outcome logging
    st.markdown('<p class="section-hdr">Outcome Logging</p>', unsafe_allow_html=True)
    with st.form(key=f"al_outcome_form_{patient_id}"):
        attempted = st.radio(t["outcome"]["attempted_q"],
                              [t["outcome"]["yes"], t["outcome"]["no"]], horizontal=True)
        method = st.selectbox(t["outcome"]["method_q"], t["outcome"]["methods"])
        reached = st.radio(t["outcome"]["reached_q"],
                            [t["outcome"]["yes"], t["outcome"]["no"]], horizontal=True)
        outcome = st.selectbox(t["outcome"]["outcome_q"], t["outcome"]["outcomes"])
        notes = st.text_area(t["outcome"]["notes_q"])
        submitted = st.form_submit_button(t["outcome"]["save_btn"])

    if submitted:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "user_role": "Programme / CHW (demo)",
            "patient_id": patient_id,
            "risk_category": row["risk_label"],
            "recommended_action": t["next_step"][row["risk_label"]],
            "selected_language": lang,
            "outreach_attempted": attempted,
            "outreach_method": method,
            "patient_reached": reached,
            "outcome_recorded": outcome,
            "notes": notes,
            "model_version": MODEL_VERSION_LABEL,
        }
        st.session_state["al_audit_log"].append(record)
        df.loc[df["patient_id"] == patient_id, "last_outreach_status"] = outcome
        try:
            log_event(supabase, "action_lens_outreach", record)
        except Exception:
            pass  # demo mode / no supabase configured — never block the UI on this
        st.markdown(f'<div class="success-box">{t["outcome"]["saved_msg"]}</div>',
                    unsafe_allow_html=True)

    # Audit trail
    st.markdown('<p class="section-hdr">Audit Trail</p>', unsafe_allow_html=True)
    entries = [e for e in st.session_state["al_audit_log"] if e["patient_id"] == patient_id]
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.caption("No outreach recorded yet for this patient.")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def render_action_lens(supabase=None, log_event=None):
    """Top-level ActionLens page. Called from app.py the same way every
    other page module is called: render_action_lens(supabase=supabase,
    log_event=log_event)."""
    if log_event is None:
        def log_event(*args, **kwargs):
            pass

    st.session_state.setdefault("al_cohort", _generate_synthetic_cohort())
    st.session_state.setdefault("al_audit_log", [])
    st.session_state.setdefault("al_view", "list")
    st.session_state.setdefault("al_selected_patient", None)
    st.session_state.setdefault("al_language", "English")

    df = st.session_state["al_cohort"]

    if st.session_state["al_view"] == "detail" and st.session_state["al_selected_patient"]:
        _render_detail_view(df, st.session_state["al_selected_patient"], supabase, log_event)
    else:
        _render_weekly_list(df)
