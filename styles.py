"""
SmartDaaS v1.0 — Styles
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

All platform CSS in one place.
Injected into the app via inject_css() called once at startup.
"""

import streamlit as st

SMARTDAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e2eaf3; }
.smartdaas-header {
    background: #111820;
    border: 1px solid #00e5ff33; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}
.brand-name { font-family:'IBM Plex Mono',monospace; font-size:3.6rem;
    font-weight:600; color:#00e5ff; letter-spacing:-2px; margin:0; line-height:1.1; }
.brand-sub { font-size:1.05rem; color:#ffffff; margin-top:0.5rem; font-weight:400; opacity:0.88; }
.version-tag { display:inline-block; background:#00e5ff18; border:1px solid #00e5ff55;
    color:#00e5ff; font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    padding:2px 8px; border-radius:4px; margin-top:0.5rem; }
.metric-box { background:#1e2530; border:1px solid #2e3a4a; border-radius:8px;
    padding:1rem; text-align:center; margin-bottom:0.5rem; min-width:0; overflow:hidden; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
    font-weight:600; color:#ffb300; white-space:nowrap; overflow:hidden;
    text-overflow:ellipsis; display:block; }
.metric-lbl { font-size:0.75rem; color:#ffffff; text-transform:uppercase;
    letter-spacing:0.5px; white-space:normal; word-break:break-word; opacity:0.75; }
.risk-high { background:#2a1010; border:1px solid #ff453a77; border-radius:10px;
    padding:1rem; text-align:center; color:#ff453a; }
/* Navigation — professional clinical styling */
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 1px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 5px 10px 5px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    border-left: 3px solid transparent !important;
    transition: background 0.1s !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1e2530 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.95rem !important;
    color: #c8d8e8 !important;
    margin: 0 !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] {
    background: #0a1e2e !important;
    border-left: 3px solid #00e5ff !important;
    padding-left: 9px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Group break lines before 3rd, 6th, 10th items */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(3),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(6),
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(10) {
    margin-top: 8px !important;
    border-top: 1px solid #1e2530 !important;
    padding-top: 8px !important;
}
/* Hide separator items in nav radio */
div[data-testid="stRadio"] label:has(> div > p:is([data-value="— Patient Analytics —"],
    [data-value="— Programme Intelligence —"],
    [data-value="— Validation & Model —"])) {
    display: none !important;
}
.risk-medium { background:#231a00; border:1px solid #ffb30077; border-radius:10px;
    padding:1rem; text-align:center; color:#ffb300; }
.risk-low { background:#0c2014; border:1px solid #30d15877; border-radius:10px;
    padding:1rem; text-align:center; color:#30d158; }
.risk-number { font-family:'IBM Plex Mono',monospace; font-size:2.5rem; font-weight:600; }
.risk-label { font-size:0.8rem; opacity:0.9; text-transform:uppercase; letter-spacing:1px; }
.section-hdr { font-family:'IBM Plex Mono',monospace; font-size:0.95rem; color:#00e5ff;
    text-transform:uppercase; letter-spacing:2px; border-bottom:1px solid #00e5ff44;
    padding-bottom:0.5rem; margin-bottom:1rem; margin-top:1.5rem; }
.info-box { background:#1e2530; border-left:3px solid #00e5ff; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.95rem; color:#cdd9e5; margin:0.75rem 0; }
.warn-box { background:#231a00; border-left:3px solid #ffb300; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.95rem; color:#ffb300; margin:0.75rem 0; }
.success-box { background:#0c2014; border-left:3px solid #30d158; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.95rem; color:#30d158; margin:0.75rem 0; }
.template-box { background:#1e2530; border:1px solid #00e5ff44; border-radius:8px;
    padding:1.5rem; margin:1rem 0; }
[data-testid="stSidebar"] { background-color:#0c0f14; border-right:1px solid #1e2530; }
[data-testid="stSidebar"] * { color: #dde8f0 !important; }
[data-testid="stSidebar"] .section-hdr { color: #00e5ff !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
/* Hide desktop header bar but keep mobile sidebar toggle visible */
@media (min-width: 768px) { header {visibility:hidden;} }
@media (max-width: 767px) {
    header { visibility: visible !important; background: #0c0f14 !important; }
    header [data-testid="stToolbar"] { visibility: visible !important; }
    /* Hide everything in the header EXCEPT the sidebar toggle button */
    header button[kind="header"] ~ * { display: none !important; }
}

/* ── Mobile bottom navigation bar ── */
@media (max-width: 767px) {
    .mobile-nav-bar {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 9999;
        background: #0c0f14; border-top: 1px solid #1e2530;
        display: flex; justify-content: space-around; align-items: center;
        padding: 6px 0 max(6px, env(safe-area-inset-bottom));
    }
    .mobile-nav-item {
        display: flex; flex-direction: column; align-items: center;
        font-size: 0.6rem; color: #6e7b8a; cursor: pointer;
        padding: 4px 8px; border-radius: 8px; text-decoration: none;
        -webkit-tap-highlight-color: transparent; min-width: 52px;
    }
    .mobile-nav-item .nav-icon { font-size: 1.3rem; line-height: 1; }
    .mobile-nav-item.active { color: #00e5ff; }
    .mobile-nav-item.active .nav-icon { filter: drop-shadow(0 0 4px #00e5ff88); }
    /* Push page content above the bottom bar */
    .main .block-container { padding-bottom: 90px !important; }
}
@media (min-width: 768px) { .mobile-nav-bar { display: none !important; } }
</style>
"""


def inject_css():
    """Inject SmartDaaS platform CSS. Call once at app startup."""
    st.markdown(SMARTDAAS_CSS, unsafe_allow_html=True)
