"""
SmartDaaS — FrontlineLens Module
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

FrontlineLens converts model risk output into frontline outreach action:
risk prediction -> explanation -> localized action prompt -> outreach
logging -> audit trail -> feedback loop.

Design notes (read before editing):
  - Fully self-contained. Uses synthetic demo patients only. Does NOT
    read st.session_state['df_scored'] or any other page's state, and
    does NOT import model.py / call the trained model or SHAP. This is
    deliberate: FrontlineLens must keep working even if the trained model
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
  - Brand name note: the page is named "FrontlineLens" in the nav and
    the weekly-list title/subtitle. Internal identifiers (module file
    name, function names, session_state keys, the Supabase event-type
    string "action_lens_outreach") deliberately keep the old "action_lens"
    naming — nobody but a developer ever sees those, and changing them
    would touch more surface area (e.g. app.py's import line, any
    existing Supabase rows) for zero user-visible benefit.

Exports:
    render_action_lens(supabase=None, log_event=None)
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st

MODEL_VERSION_LABEL = "SmartDaaS-RF-v0.1-demo"

DEMO_LANGUAGES = ["English", "French", "Swahili", "Hausa", "Yoruba", "Igbo", "Kinyarwanda"]

ACTIONLENS_DISCLAIMER = (
    "SmartDaaS FrontlineLens is a decision-support module for pilot "
    "evaluation. It does not provide diagnosis or treatment "
    "recommendations. All outreach decisions remain under the "
    "responsibility of qualified clinical/program staff."
)

PAGE_TITLE = "FrontlineLens: From Risk to Outreach"
PAGE_SUBTITLE = "Explainable risk translated into frontline action"

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
# Split localization boundary:
#   Localized: risk word, plain-language reason, suggested next step /
#   backup / script, outcome-logging form, and all "human-facing UI
#   chrome" (section headers, field labels, safety badges, the two
#   localization disclaimers) — so the whole page visibly changes.
#   NOT localized: explanation factor list, "not causal claims" note,
#   model version. These are the model/technical-output layer and stay
#   in English by design — see the "ui"."boundary_note" string shown
#   in the Localization card, which makes this split explicit to the
#   person using the page.
#   Audit records always store CANONICAL ENGLISH values regardless of
#   which language the form was filled out in (see _canonical() below),
#   so the audit trail stays comparable across patients/languages.
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
        "ui": {
            "language_label": "Language",
            "back_btn": "← Back to weekly outreach list",
            "positioning": "SmartDaaS does not stop at risk prediction. FrontlineLens helps care teams understand why a patient was flagged, what to check next, and how to document the follow-up.",
            "section_risk_summary": "Risk Summary", "section_explanation": "Explanation",
            "section_frontline_prompt": "Frontline Action Prompt",
            "section_suggested_outreach": "Suggested Outreach",
            "section_localization": "Localization",
            "section_outcome_logging": "Outcome Logging", "section_audit_trail": "Audit Trail",
            "label_priority_rank": "Priority Rank", "label_patient_id": "Patient ID",
            "label_next_step": "Recommended next step:", "label_backup": "Backup action:",
            "label_what_to_check": "What to check:",
            "what_to_check_value": "Appointment · Medication refill · Transport · Counseling need",
            "label_script": "Suggested script:",
            "badge_decision_support": "Decision support only",
            "badge_synthetic_demo": "Synthetic demo data",
            "localization_disclaimer": "Demo localization only. Final language templates should be reviewed by local clinical/program teams before field use.",
            "language_note": "Demo languages are illustrative examples selected to show country/site-level localization. Final language options should be determined with implementation partners and frontline teams.",
            "boundary_note": "Technical and model-derived content (explanation drivers, model version, audit records) stays in English. Frontline action content above is localized.",
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
        "ui": {
            "language_label": "Langue",
            "back_btn": "← Retour à la liste de suivi hebdomadaire",
            "positioning": "SmartDaaS ne se limite pas à la prédiction du risque. FrontlineLens aide les équipes de soins à comprendre pourquoi un patient a été signalé, ce qu'il faut vérifier ensuite, et comment documenter le suivi.",
            "section_risk_summary": "Résumé du risque", "section_explanation": "Explication",
            "section_frontline_prompt": "Message d'action de première ligne",
            "section_suggested_outreach": "Suivi suggéré",
            "section_localization": "Localisation",
            "section_outcome_logging": "Enregistrement du résultat", "section_audit_trail": "Journal d'audit",
            "label_priority_rank": "Rang de priorité", "label_patient_id": "ID du patient",
            "label_next_step": "Prochaine étape recommandée :", "label_backup": "Action de secours :",
            "label_what_to_check": "À vérifier :",
            "what_to_check_value": "Rendez-vous · Renouvellement de traitement · Transport · Besoin de counseling",
            "label_script": "Script suggéré :",
            "badge_decision_support": "Aide à la décision uniquement",
            "badge_synthetic_demo": "Données de démonstration synthétiques",
            "localization_disclaimer": "Localisation de démonstration uniquement. Les modèles linguistiques finaux doivent être révisés par les équipes cliniques/programmatiques locales avant utilisation sur le terrain.",
            "language_note": "Les langues de démonstration sont des exemples illustratifs choisis pour montrer la localisation au niveau du pays/site. Les options linguistiques définitives devraient être déterminées avec les partenaires de mise en œuvre et les équipes de première ligne.",
            "boundary_note": "Le contenu technique et issu du modèle (facteurs d'explication, version du modèle, registres d'audit) reste en anglais. Le contenu d'action de première ligne ci-dessus est localisé.",
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
        "ui": {
            "language_label": "Lugha",
            "back_btn": "← Rudi kwenye orodha ya ufuatiliaji wa wiki",
            "positioning": "SmartDaaS haisimami kwenye utabiri wa hatari. FrontlineLens husaidia timu za huduma kuelewa kwa nini mgonjwa aliwekwa alama, kinachohitaji kuangaliwa baadaye, na jinsi ya kuandika ufuatiliaji.",
            "section_risk_summary": "Muhtasari wa Hatari", "section_explanation": "Maelezo",
            "section_frontline_prompt": "Ujumbe wa Kitendo cha Mstari wa Mbele",
            "section_suggested_outreach": "Ufuatiliaji Uliopendekezwa",
            "section_localization": "Ujanibishaji",
            "section_outcome_logging": "Kurekodi Matokeo", "section_audit_trail": "Rekodi ya Ukaguzi",
            "label_priority_rank": "Kipaumbele cha Nafasi", "label_patient_id": "Kitambulisho cha Mgonjwa",
            "label_next_step": "Hatua inayopendekezwa:", "label_backup": "Hatua ya akiba:",
            "label_what_to_check": "Cha kuangalia:",
            "what_to_check_value": "Miadi · Kuongeza dawa · Usafiri · Uhitaji wa ushauri",
            "label_script": "Maandishi yaliyopendekezwa:",
            "badge_decision_support": "Msaada wa maamuzi tu",
            "badge_synthetic_demo": "Data ya onyesho ya bandia",
            "localization_disclaimer": "Ujanibishaji wa onyesho tu. Violezo vya lugha vya mwisho vinapaswa kukaguliwa na timu za kimatibabu/kiprogramu za eneo kabla ya matumizi shambani.",
            "language_note": "Lugha za onyesho ni mifano ya kielelezo iliyochaguliwa kuonyesha ujanibishaji wa kiwango cha nchi/eneo. Chaguo za lugha za mwisho zinapaswa kuamuliwa pamoja na washirika wa utekelezaji na timu za mstari wa mbele.",
            "boundary_note": "Maudhui ya kiufundi na yanayotokana na modeli (vichocheo vya maelezo, toleo la modeli, rekodi za ukaguzi) yanabaki kwa Kiingereza. Maudhui ya kitendo cha mstari wa mbele hapo juu yamejanibishwa.",
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
        "ui": {
            "language_label": "Harshe",
            "back_btn": "← Koma zuwa jerin bibiya na mako",
            "positioning": "SmartDaaS baya tsayawa kawai akan hasashen haɗari. FrontlineLens na taimaka wa ƴan kungiyoyin kulawa su fahimci dalilin da majiyyaci ya samu alama, abin da za a duba na gaba, da kuma yadda za a rubuta bibiya.",
            "section_risk_summary": "Taƙaitaccen Haɗari", "section_explanation": "Bayani",
            "section_frontline_prompt": "Saƙon Mataki na Gaba",
            "section_suggested_outreach": "Bibiya da Aka Bayar da Shawara",
            "section_localization": "Fassara zuwa Harshen Gida",
            "section_outcome_logging": "Rikodin Sakamako", "section_audit_trail": "Rikodin Bincike",
            "label_priority_rank": "Matsayin Fifiko", "label_patient_id": "Lambar Majiyyaci",
            "label_next_step": "Matakin da aka bada shawara:", "label_backup": "Madadin mataki:",
            "label_what_to_check": "Abin da za a duba:",
            "what_to_check_value": "Ziyara · Sake samun magani · Sufuri · Bukatar shawara",
            "label_script": "Rubutun da aka bada shawara:",
            "badge_decision_support": "Taimakon yanke shawara kawai",
            "badge_synthetic_demo": "Bayanan nuni na ƙirƙira",
            "localization_disclaimer": "Fassarar nuni kawai. Tsarukan harshe na ƙarshe ya kamata su sake dubawa daga ƴan ƙungiyoyin asibiti/shirye-shirye na gida kafin amfani da su a fage.",
            "language_note": "Harsunan nuni misalai ne na bayyana da aka zaɓa don nuna fassarar harshe a matakin ƙasa/wuri. Ya kamata a tantance zaɓin harsuna na ƙarshe tare da abokan aiwatarwa da ƴan ƙungiyoyin gaba.",
            "boundary_note": "Abubuwan fasaha da na model (abubuwan bayani, sigar model, rikodin bincike) suna kasance da Turanci. Abubuwan mataki na gaba a sama an fassara su.",
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
        "ui": {
            "language_label": "Èdè",
            "back_btn": "← Padà sí àtòjọ ìtọ́jú ọ̀sẹ̀",
            "positioning": "SmartDaaS kò dúró ní àsọtẹ́lẹ̀ ewu nìkan. FrontlineLens ń ràn àwọn ẹgbẹ́ ìtọ́jú lọ́wọ́ láti mọ ìdí tí a fi ṣàmì sí aláìsàn, ohun tí ó yẹ kí a ṣàyẹ̀wò lẹ́yìn náà, àti bí a ṣe lè ṣàkọsílẹ̀ ìtọ́jú náà.",
            "section_risk_summary": "Àkópọ̀ Ewu", "section_explanation": "Àlàyé",
            "section_frontline_prompt": "Ìpèníjà Ìṣe Iwájú",
            "section_suggested_outreach": "Ìtọ́jú Tí A Dámọ̀ràn",
            "section_localization": "Ìṣàtúmọ̀ Sí Èdè Ìbílẹ̀",
            "section_outcome_logging": "Àkọsílẹ̀ Àbájáde", "section_audit_trail": "Àkọsílẹ̀ Àyẹ̀wò",
            "label_priority_rank": "Ipò Pàtàkì", "label_patient_id": "Ìdánimọ̀ Aláìsàn",
            "label_next_step": "Ìgbésẹ̀ tí a dámọ̀ràn:", "label_backup": "Ìgbésẹ̀ àfikún:",
            "label_what_to_check": "Ohun tí o yẹ kí a ṣàyẹ̀wò:",
            "what_to_check_value": "Ìpàdé · Àtúnṣe òògùn · Ìrìnnà · Ìmọ̀ràn tí a nílò",
            "label_script": "Ọ̀rọ̀ tí a dámọ̀ràn:",
            "badge_decision_support": "Ìrànlọ́wọ́ ìpinnu nìkan",
            "badge_synthetic_demo": "Dátà àpẹẹrẹ irọ́",
            "localization_disclaimer": "Ìṣàtúmọ̀ àpẹẹrẹ nìkan. Àwọn àpẹẹrẹ èdè tí ó kẹyìn yẹ kí a fi àwọn ẹgbẹ́ ìtọ́jú/ètò agbègbè yẹ̀wò kí wọ́n tó lò wọ́n níta gidi.",
            "language_note": "Àwọn èdè àpẹẹrẹ jẹ́ àpẹẹrẹ tí a yàn láti fi ìṣàtúmọ̀ ìpele orílẹ̀-èdè/agbègbè hàn. Yóò yẹ kí a pinnu àwọn èdè tí ó kẹyìn pẹ̀lú àwọn alábàákẹ́gbẹ́ àti àwọn ẹgbẹ́ iwájú.",
            "boundary_note": "Àwọn àkóónú onímọ̀ ẹ̀rọ àti tí ó ti inú àpẹẹrẹ wá (àwọn okùnfà àlàyé, ẹ̀yà àpẹẹrẹ, àkọsílẹ̀ àyẹ̀wò) yóò máa wà ní èdè Gẹ̀ẹ́sì. Àkóónú ìṣe iwájú lókè ni a ti ṣàtúmọ̀.",
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
        "ui": {
            "language_label": "Asụsụ",
            "back_btn": "← Laghachi na ndepụta nleba anya kwa izu",
            "positioning": "SmartDaaS akwụsịghị na ịkọ amụma ihe egwu. FrontlineLens na-enyere ndị otu nlekọta aka ịghọta ihe kpatara e ji mara onye ọrịa, ihe a ga-elele ọzọ, na otu e si edekọ nleba anya.",
            "section_risk_summary": "Nchịkọta Ihe Egwu", "section_explanation": "Nkọwa",
            "section_frontline_prompt": "Ozi Omume Ihu Mmadụ",
            "section_suggested_outreach": "Nleba Anya A Tụrụ Aro",
            "section_localization": "Ntụgharị n'Asụsụ Mpaghara",
            "section_outcome_logging": "Ndekọ Nsonaazụ", "section_audit_trail": "Ndekọ Nyocha",
            "label_priority_rank": "Ọnọdụ Mkpa", "label_patient_id": "Nchọpụta Onye Ọrịa",
            "label_next_step": "Usoro a tụrụ aro:", "label_backup": "Usoro nkwado:",
            "label_what_to_check": "Ihe a ga-elele:",
            "what_to_check_value": "Nleta · Ọgwụ ọzọ · Njem · Mkpa ndụmọdụ",
            "label_script": "Okwu a tụrụ aro:",
            "badge_decision_support": "Enyemaka mkpebi naanị",
            "badge_synthetic_demo": "Data ngosi nke a kpụrụ akpụ",
            "localization_disclaimer": "Ntụgharị ngosi naanị. Ndị otu ahụike/mmemme mpaghara kwesịrị ilele ụkpụrụ asụsụ ikpeazụ tupu e jiri ya rụọ ọrụ n'ọrụ.",
            "language_note": "Asụsụ ngosi bụ ihe atụ doro anya nke e họọrọ iji gosipụta ntụgharị n'ọkwa obodo/ebe. Ekwesịrị ikpebi nhọrọ asụsụ ikpeazụ ya na ndị mmekọ mmejuputa na ndị otu ihu mmadụ.",
            "boundary_note": "Ọdịnaya teknụzụ na nke sitere na modeli (ihe kpatara nkọwa, ụdị modeli, ndekọ nyocha) ga-anọgide na Bekee. Ọdịnaya omume ihu mmadụ n'elu a sụgharịrị ya.",
        },
    },
    "Kinyarwanda": {
        "risk_word": {"HIGH": "Kinini", "MEDIUM": "Hagati", "LOW": "Gito"},
        "reason": {
            "HIGH": "Uyu murwayi ashobora gukenera gukurikiranwa kubera ko atagiye ku gahunda ye iheruka kandi hashobora kubaho icyuho mu kongera imiti ye.",
            "MEDIUM": "Uyu murwayi ntiyajya buri gihe mu gahunda ze, kandi yafasha guhamagarwa kugira ngo barebe uko ameze.",
            "LOW": "Uyu murwayi ari mu nzira nziza y'ubuvuzi. Gukurikirana gakurikira mu gahunda izaza birahagije.",
        },
        "next_step": {
            "HIGH": "Hamagara mu minsi 7",
            "MEDIUM": "Ohereza ubutumwa SMS, hamagara mu byumweru 2",
            "LOW": "Gukurikirana gakurikira mu gahunda izaza",
        },
        "backup": {
            "HIGH": "Umujyanama w'ubuzima w'umuryango asure niba adashoboka kumuhamagara",
            "MEDIUM": "Umujyanama w'ubuzima w'umuryango asure niba ntasubije SMS",
            "LOW": "Nta kindi gikenewe",
        },
        "script": {
            "HIGH": "Muraho, twabonye ko ushobora gukenera ubuvuzi. Ukeneye ubufasha mu guhindura itariki y'igihe cyo kubonana cyangwa kubona imiti yawe?",
            "MEDIUM": "Muraho, turagusuzuma gusa - hari icyo ukeneye gufashwa kijyanye n'igihe cyo kubonana cyawe gitaha cyangwa imiti yawe?",
            "LOW": "Muraho, iki ni igenzura risanzwe. Turifuza kukubona mu gahunda yawe itaha.",
        },
        "outcome": {
            "attempted_q": "Hageragejwe kuvugana?", "method_q": "Uburyo bwakoreshejwe",
            "reached_q": "Umurwayi yagezweho?", "outcome_q": "Igisubizo",
            "notes_q": "Inyandiko", "save_btn": "Bika inyandiko y'ikurikirana",
            "saved_msg": "Inyandiko y'ikurikirana yabitswe.",
            "yes": "Yego", "no": "Oya",
            "methods": ["Telefone", "SMS", "Gusura n'umujyanama w'ubuzima", "Ikindi"],
            "outcomes": ["Itariki yo kubonana yahinduwe", "Imiti yateguwe",
                         "Ubujyanama burakenewe", "Ubufasha bw'ubwikorezi burakenewe",
                         "Ntiyashoboye kugerwaho", "Ikindi"],
        },
        "ui": {
            "language_label": "Ururimi",
            "back_btn": "← Garuka ku rutonde rw'ikurikirana ry'icyumweru",
            "positioning": "SmartDaaS ntigarukira ku guteganya ibyago gusa. FrontlineLens ifasha itsinda ry'ubuvuzi gusobanukirwa impamvu umurwayi yamenyekanye, icyagombye gusuzumwa, n'uburyo bwo kwandika ikurikirana.",
            "section_risk_summary": "Incamake y'Ibyago", "section_explanation": "Ibisobanuro",
            "section_frontline_prompt": "Ubutumwa bw'Igikorwa cy'Imbere",
            "section_suggested_outreach": "Ikurikirana Ryasabwe",
            "section_localization": "Guhindura mu Rurimi",
            "section_outcome_logging": "Kwandika Igisubizo", "section_audit_trail": "Inyandiko y'Igenzura",
            "label_priority_rank": "Urutonde rw'Ibanze", "label_patient_id": "Indangamuntu y'Umurwayi",
            "label_next_step": "Intambwe yasabwe:", "label_backup": "Igikorwa cy'inyongera:",
            "label_what_to_check": "Ibyo gusuzuma:",
            "what_to_check_value": "Igihe cyo kubonana · Kongera imiti · Ubwikorezi · Ubujyanama bukenewe",
            "label_script": "Amagambo yasabwe:",
            "badge_decision_support": "Ubufasha mu kufata ibyemezo gusa",
            "badge_synthetic_demo": "Amakuru y'ikinamico",
            "localization_disclaimer": "Guhindura mu rurimi by'ikinamico gusa. Inyandiko z'ururimi za nyuma zigomba gusuzumwa n'itsinda ry'ubuvuzi/porogaramu byo mu karere mbere yo gukoreshwa mu murima.",
            "language_note": "Indimi z'ikinamico ni urugero rugaragaza guhindura mu rurimi ku rwego rw'igihugu/aho biherereye. Amahitamo y'indimi za nyuma agomba gufatwa hamwe n'abafatanyabikorwa b'ishyirwa mu bikorwa n'itsinda ry'imbere.",
            "boundary_note": "Ibikubiye mu buhanga no biva muri model (impamvu z'ibisobanuro, verisiyo ya model, inyandiko z'igenzura) bizagumana mu Cyongereza. Ibikubiye mu gikorwa cy'imbere byavuzwe haruguru byahinduwe mu rurimi.",
        },
    },
}


# ─────────────────────────────────────────────────────────────
# SYNTHETIC COHORT GENERATOR
# Deterministic, self-contained. Independent of any data uploaded
# elsewhere in the platform — FrontlineLens never reads df_scored or any
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


def _canonical_outcome_value(value, lang, field):
    """Map a localized form selection back to its canonical English
    equivalent before it's written to the audit record. This keeps the
    audit trail comparable across patients even when different CHWs
    filled out the form in different languages — without this, the
    'outcome_recorded' column would mix French/Swahili/etc. text for
    the same underlying outcome, which defeats the point of an audit
    trail. Relies on every language's outcome lists being in the same
    fixed order as English (true by construction in L10N above).
    """
    if field == "yesno":
        return "Yes" if value == L10N[lang]["outcome"]["yes"] else "No"
    try:
        idx = L10N[lang]["outcome"][field].index(value)
        return L10N["English"]["outcome"][field][idx]
    except (ValueError, KeyError):
        return value  # fallback — should not happen given fixed lists


# ─────────────────────────────────────────────────────────────
# WEEKLY OUTREACH LIST
# ─────────────────────────────────────────────────────────────
def _render_weekly_list(df):
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace; font-size:1.6rem; '
        f'font-weight:600; color:#00e5ff; margin-bottom:0.15rem;">{PAGE_TITLE}</p>'
        f'<p style="font-size:1.0rem; color:#ffffff; opacity:0.85; margin-top:0; '
        f'margin-bottom:1rem; font-weight:400;">{PAGE_SUBTITLE}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _badge("Synthetic demo data") + _badge("Decision support only") +
        _badge("Human review required") + _badge("Pilot-ready workflow"),
        unsafe_allow_html=True,
    )
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

    lang = st.session_state.get("al_language_select", st.session_state.get("al_language", "English"))
    t = L10N[lang]

    if st.button(t["ui"]["back_btn"], key="al_back_btn"):
        st.session_state["al_view"] = "list"
        st.rerun()

    st.markdown('<p class="section-hdr">FrontlineLens</p>', unsafe_allow_html=True)
    st.markdown(t["ui"]["positioning"])

    lang = st.selectbox(
        t["ui"]["language_label"], DEMO_LANGUAGES,
        index=DEMO_LANGUAGES.index(lang),
        key="al_language_select",
    )
    st.session_state["al_language"] = lang
    t = L10N[lang]  # re-fetch in case the selection changed this run

    # 1. Risk Summary — UI chrome localized; model version stays English
    st.markdown(f'<p class="section-hdr">1 · {t["ui"]["section_risk_summary"]}</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_risk_box(row["risk_label"], row["risk_pct"], lang), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><span class="metric-val">{row["priority_rank"]}</span>'
                     f'<span class="metric-lbl">{t["ui"]["label_priority_rank"]}</span></div>',
                     unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><span class="metric-val">{row["patient_id"]}</span>'
                     f'<span class="metric-lbl">{t["ui"]["label_patient_id"]}</span></div>',
                     unsafe_allow_html=True)
    st.markdown(_badge(t["ui"]["badge_decision_support"]) + _badge(t["ui"]["badge_synthetic_demo"]),
                unsafe_allow_html=True)
    st.caption(f"Model version: {MODEL_VERSION_LABEL}")  # technical/model output — always English

    # 2. Explanation — header localized; driver list + causal-claims note stay English (model-adjacent)
    st.markdown(f'<p class="section-hdr">2 · {t["ui"]["section_explanation"]}</p>',
                unsafe_allow_html=True)
    for d in row["drivers"]:
        st.markdown(f"- {d}")
    st.markdown('<div class="info-box">These are model explanation factors, not causal claims.</div>',
                unsafe_allow_html=True)

    # 3. Frontline Action Prompt — fully localized
    st.markdown(f'<p class="section-hdr">3 · {t["ui"]["section_frontline_prompt"]}</p>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="template-box">{t["reason"][row["risk_label"]]}</div>',
                unsafe_allow_html=True)

    # 4. Suggested Outreach — fully localized, including labels and the checklist line
    st.markdown(f'<p class="section-hdr">4 · {t["ui"]["section_suggested_outreach"]}</p>',
                unsafe_allow_html=True)
    st.markdown(f"**{t['ui']['label_next_step']}** {t['next_step'][row['risk_label']]}")
    st.markdown(f"**{t['ui']['label_backup']}** {t['backup'][row['risk_label']]}")
    st.markdown(f"**{t['ui']['label_what_to_check']}** {t['ui']['what_to_check_value']}")
    st.markdown(f'<div class="template-box"><em>{t["ui"]["label_script"]}</em><br>'
                f'{t["script"][row["risk_label"]]}</div>', unsafe_allow_html=True)

    # 5. Localization — disclaimers + the boundary note explaining the split, all localized
    st.markdown(f'<p class="section-hdr">5 · {t["ui"]["section_localization"]}</p>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="warn-box">{t["ui"]["localization_disclaimer"]}</div>',
                unsafe_allow_html=True)
    st.caption(t["ui"]["language_note"])
    st.caption(t["ui"]["boundary_note"])

    # Outcome logging — form is localized; what gets STORED is canonicalized to English below
    st.markdown(f'<p class="section-hdr">{t["ui"]["section_outcome_logging"]}</p>',
                unsafe_allow_html=True)
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
        # Canonicalize every fixed-vocabulary field to English before storing, so the
        # audit trail stays comparable across patients regardless of which language
        # the form was filled out in. Free-text notes can't be canonicalized and are
        # stored as typed; everything else comes from a fixed, language-parallel list.
        canon_outcome = _canonical_outcome_value(outcome, lang, "outcomes")
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "user_role": "Programme / CHW (demo)",
            "patient_id": patient_id,
            "risk_category": row["risk_label"],
            "recommended_action": L10N["English"]["next_step"][row["risk_label"]],
            "selected_language": lang,
            "outreach_attempted": _canonical_outcome_value(attempted, lang, "yesno"),
            "outreach_method": _canonical_outcome_value(method, lang, "methods"),
            "patient_reached": _canonical_outcome_value(reached, lang, "yesno"),
            "outcome_recorded": canon_outcome,
            "notes": notes,
            "model_version": MODEL_VERSION_LABEL,
        }
        st.session_state["al_audit_log"].append(record)
        df.loc[df["patient_id"] == patient_id, "last_outreach_status"] = canon_outcome
        try:
            log_event(supabase, "action_lens_outreach", record)
        except Exception:
            pass  # demo mode / no supabase configured — never block the UI on this
        st.markdown(f'<div class="success-box">{t["outcome"]["saved_msg"]}</div>',
                    unsafe_allow_html=True)

    # Audit trail — header localized; the data inside is always canonical English
    st.markdown(f'<p class="section-hdr">{t["ui"]["section_audit_trail"]}</p>',
                unsafe_allow_html=True)
    entries = [e for e in st.session_state["al_audit_log"] if e["patient_id"] == patient_id]
    if entries:
        display_cols = {
            "timestamp": "Timestamp", "user_role": "User Role", "risk_category": "Risk Category",
            "recommended_action": "Recommended Action", "selected_language": "Language Used",
            "outreach_attempted": "Attempted", "outreach_method": "Method",
            "patient_reached": "Reached", "outcome_recorded": "Outcome",
            "notes": "Notes", "model_version": "Model Version",
        }
        audit_df = pd.DataFrame(entries).drop(columns=["patient_id"]).rename(columns=display_cols)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No outreach recorded yet for this patient.")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def render_action_lens(supabase=None, log_event=None):
    """Top-level FrontlineLens page. Called from app.py the same way every
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
