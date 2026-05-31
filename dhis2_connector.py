"""
SmartDaaS DHIS2 Connector
─────────────────────────────────────────────────────────────────────────────
Pulls patient-level ART programme data directly from any DHIS2 instance
via the DHIS2 Web API. Feeds into the existing SmartDaaS normalize_columns()
pipeline — no changes needed to the core model or analysis code.

Usage (add to app.py sidebar alongside the CSV upload option):
    from dhis2_connector import render_dhis2_connector
    df = render_dhis2_connector()
    if df is not None:
        # feed df into normalize_columns() exactly as you would a CSV upload

Security:
    - Credentials never stored — held in st.session_state for session only
    - Zero patient data written to disk or logged
    - All API calls made server-side via requests library
    - SHA-256 session hashing preserved (same as CSV upload path)

Author: SmartDaaS LLC
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import hashlib
import datetime
from typing import Optional, Tuple, Dict, List

# ─────────────────────────────────────────────────────────────
# DHIS2 API CLIENT
# ─────────────────────────────────────────────────────────────

class DHIS2Client:
    """
    Lightweight DHIS2 Web API client.
    Handles authentication, pagination, and data extraction.
    Stateless — no data persisted between calls.
    """

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def ping(self) -> Tuple[bool, str]:
        """Test connection and credentials."""
        try:
            r = self.session.get(
                f"{self.base_url}/api/system/info",
                timeout=10
            )
            if r.status_code == 200:
                info = r.json()
                version = info.get('version', 'unknown')
                instance = info.get('instanceBaseUrl', self.base_url)
                return True, f"Connected — DHIS2 v{version}"
            elif r.status_code == 401:
                return False, "Authentication failed — check username and password"
            elif r.status_code == 404:
                return False, "DHIS2 API not found — check the base URL"
            else:
                return False, f"Connection error — HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot reach server — check the URL and your network"
        except requests.exceptions.Timeout:
            return False, "Connection timed out — server may be slow or unavailable"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def get_programs(self) -> List[Dict]:
        """List available tracker programs (ART/HIV programmes)."""
        try:
            r = self.session.get(
                f"{self.base_url}/api/programs",
                params={
                    'fields': 'id,displayName,programType,trackedEntityType[displayName]',
                    'paging': 'false',
                    'programType': 'WITH_REGISTRATION'
                },
                timeout=15
            )
            if r.status_code == 200:
                return r.json().get('programs', [])
            return []
        except Exception:
            return []

    def get_org_units(self, level: int = 3) -> List[Dict]:
        """List organisation units at a given level (default 3 = facility)."""
        try:
            r = self.session.get(
                f"{self.base_url}/api/organisationUnits",
                params={
                    'fields': 'id,displayName,level',
                    'level': level,
                    'paging': 'false'
                },
                timeout=15
            )
            if r.status_code == 200:
                return r.json().get('organisationUnits', [])
            return []
        except Exception:
            return []

    def get_tracked_entities(
        self,
        program_id: str,
        org_unit_id: str,
        page_size: int = 250,
        max_records: int = 5000
    ) -> Tuple[pd.DataFrame, str]:
        """
        Pull tracked entity instances for a given programme and org unit.
        Returns (DataFrame of raw attributes, status_message).
        Paginates automatically up to max_records.
        """
        all_rows = []
        page = 1
        total_fetched = 0

        while total_fetched < max_records:
            try:
                r = self.session.get(
                    f"{self.base_url}/api/trackedEntityInstances",
                    params={
                        'program': program_id,
                        'ou': org_unit_id,
                        'ouMode': 'DESCENDANTS',
                        'fields': 'trackedEntityInstance,attributes[attribute,value],'
                                  'enrollments[enrollmentDate,status,events['
                                  'programStage,eventDate,dataValues[dataElement,value]]]',
                        'pageSize': page_size,
                        'page': page,
                        'totalPages': 'false'
                    },
                    timeout=30
                )

                if r.status_code != 200:
                    return pd.DataFrame(), f"API error on page {page}: HTTP {r.status_code}"

                data = r.json()
                instances = data.get('trackedEntityInstances', [])

                if not instances:
                    break  # no more pages

                for tei in instances:
                    row = {'patient_id': tei.get('trackedEntityInstance', '')}

                    # Extract tracked entity attributes (demographics, clinical)
                    for attr in tei.get('attributes', []):
                        key = attr.get('attribute', '').lower()
                        row[key] = attr.get('value', '')

                    # Extract latest enrollment event data values
                    enrollments = tei.get('enrollments', [])
                    if enrollments:
                        latest = sorted(
                            enrollments,
                            key=lambda e: e.get('enrollmentDate', ''),
                            reverse=True
                        )[0]
                        row['enrollment_date'] = latest.get('enrollmentDate', '')
                        row['enrollment_status'] = latest.get('status', '')

                        # Flatten all event data values
                        events = latest.get('events', [])
                        for event in events:
                            for dv in event.get('dataValues', []):
                                de_key = dv.get('dataElement', '').lower()
                                # Don't overwrite if already set from earlier event
                                if de_key not in row:
                                    row[de_key] = dv.get('value', '')

                    all_rows.append(row)
                    total_fetched += 1
                    if total_fetched >= max_records:
                        break

                page += 1

            except requests.exceptions.Timeout:
                return pd.DataFrame(all_rows), (
                    f"Timeout on page {page} — returning {total_fetched} records fetched so far"
                )
            except Exception as e:
                return pd.DataFrame(all_rows), f"Error on page {page}: {str(e)}"

        if not all_rows:
            return pd.DataFrame(), "No tracked entity instances found for this programme and org unit"

        df = pd.DataFrame(all_rows)
        return df, f"Successfully pulled {len(df):,} patient records from DHIS2"

    def get_data_elements(self, program_id: str) -> Dict[str, str]:
        """
        Return mapping of data element UIDs to display names for a programme.
        Used to make raw column names human-readable.
        """
        try:
            r = self.session.get(
                f"{self.base_url}/api/programs/{program_id}",
                params={
                    'fields': 'programStages[programStageDataElements['
                              'dataElement[id,displayName,code]]]'
                },
                timeout=15
            )
            if r.status_code == 200:
                mapping = {}
                stages = r.json().get('programStages', [])
                for stage in stages:
                    for psde in stage.get('programStageDataElements', []):
                        de = psde.get('dataElement', {})
                        uid = de.get('id', '').lower()
                        name = de.get('code', '') or de.get('displayName', '')
                        if uid and name:
                            mapping[uid] = name.lower().replace(' ', '_')
                return mapping
            return {}
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────
# DHIS2 → SMARTDAAS FIELD MAPPER
# ─────────────────────────────────────────────────────────────

# Maps common DHIS2 data element display names / codes
# to SmartDaaS COLUMN_ALIASES-compatible names.
# These cover Uganda DHIS2, Kenya NASCOP DATIM, and Malawi HMIS
# standard data element naming conventions.

DHIS2_FIELD_MAP = {
    # Age
    'age': 'age',
    'age_years': 'age',
    'age_at_art_initiation': 'age',
    'client_age': 'age',
    'patient_age': 'age',

    # Sex
    'sex': 'sex',
    'gender': 'gender',
    'client_sex': 'sex',
    'patient_sex': 'sex',

    # CD4
    'cd4_count': 'cd4_at_start',
    'cd4_at_art_initiation': 'cd4_at_start',
    'cd4_count_at_art_start': 'cd4_at_start',
    'baseline_cd4_count': 'cd4_at_start',
    'cd4_enrolment': 'cd4_at_start',
    'most_recent_cd4': 'most_recent_cd4',
    'cd4_last_result': 'most_recent_cd4',
    'current_cd4_count': 'most_recent_cd4',
    'last_cd4_count': 'most_recent_cd4',

    # WHO Stage
    'who_clinical_stage': 'who_stage',
    'clinical_stage_at_art': 'who_stage',
    'who_stage_at_initiation': 'who_stage',
    'who_clinical_stage_at_art_start': 'who_stage',

    # Weight
    'weight_at_art_initiation': 'weight_at_start',
    'weight_kg': 'weight_at_start',
    'body_weight_kg': 'weight_at_start',
    'current_weight': 'weight_at_start',

    # BMI
    'bmi': 'bmi_start',
    'body_mass_index': 'bmi_start',

    # ART timing
    'date_of_art_initiation': 'art_start_date',
    'art_start_date': 'art_start_date',
    'date_enrolled_on_art': 'art_start_date',

    # Date of HIV diagnosis (for days_to_ART calculation)
    'date_of_hiv_diagnosis': 'date_hiv_diagnosis',
    'hiv_diagnosis_date': 'date_hiv_diagnosis',
    'date_first_hiv_positive': 'date_hiv_diagnosis',

    # Treatment interruption
    'treatment_interruption': 'had_interruption',
    'art_interruption': 'had_interruption',
    'ever_defaulted': 'had_interruption',
    'lost_to_followup': 'had_interruption',
    'ltfu': 'had_interruption',

    # OI
    'opportunistic_infection': 'opp_infection',
    'oi_diagnosis': 'opp_infection',
    'oi_present': 'opp_infection',

    # Side effects
    'side_effects': 'side_effects',
    'adverse_drug_reaction': 'side_effects',
    'art_side_effects': 'side_effects',

    # TB
    'tb_status': 'tb_positive',
    'tb_diagnosis': 'tb_positive',
    'tb_co_infection': 'tb_positive',
    'tuberculosis': 'tb_positive',
    'tb_screen_result': 'tb_positive',

    # Stage worsening
    'clinical_stage_worsened': 'stage_worsened',
    'who_stage_worsened': 'stage_worsened',
    'stage_progression': 'stage_worsened',

    # Uganda DHIS2 specific codes (UgandaEMR / METS)
    'pe_age': 'age',
    'pe_sex': 'sex',
    'cd4_1': 'cd4_at_start',
    'cd4_2': 'most_recent_cd4',
    'who_1': 'who_stage',
    'wt_1': 'weight_at_start',
    'tb_1': 'tb_positive',
    'oi_1': 'opp_infection',
    'se_1': 'side_effects',
    'int_1': 'had_interruption',

    # Kenya NASCOP / DATIM codes
    'hts_tstcd4': 'cd4_at_start',
    'tx_new_age': 'age',
    'tx_new_sex': 'sex',
    'tx_ml_iit': 'had_interruption',
    'tx_curr_cd4': 'most_recent_cd4',

    # Malawi HMIS codes
    'pre_art_cd4': 'cd4_at_start',
    'art_cd4_latest': 'most_recent_cd4',
    'art_who_stage': 'who_stage',
    'art_weight': 'weight_at_start',
    'tb_art': 'tb_positive',
}


def map_dhis2_columns(df: pd.DataFrame, de_name_map: Dict[str, str]) -> pd.DataFrame:
    """
    Renames raw DHIS2 column names (UIDs or display names) to
    SmartDaaS-compatible names using DHIS2_FIELD_MAP.
    Also applies the de_name_map (UID -> display name) from the API first.
    """
    df = df.copy()

    # Step 1: Replace UID column names with display names where available
    if de_name_map:
        uid_rename = {}
        for col in df.columns:
            if col.lower() in de_name_map:
                uid_rename[col] = de_name_map[col.lower()]
        if uid_rename:
            df = df.rename(columns=uid_rename)

    # Step 2: Map display names to SmartDaaS aliases
    field_rename = {}
    for col in df.columns:
        col_clean = col.lower().strip().replace(' ', '_').replace('-', '_')
        if col_clean in DHIS2_FIELD_MAP:
            field_rename[col] = DHIS2_FIELD_MAP[col_clean]

    if field_rename:
        df = df.rename(columns=field_rename)

    # Step 3: Derive days_to_ART from date fields if present
    if 'art_start_date' in df.columns and 'date_hiv_diagnosis' in df.columns:
        try:
            art_dates = pd.to_datetime(df['art_start_date'], errors='coerce')
            hiv_dates = pd.to_datetime(df['date_hiv_diagnosis'], errors='coerce')
            df['days_to_art'] = (art_dates - hiv_dates).dt.days.clip(lower=0)
        except Exception:
            pass

    # Step 4: Recode TB / interruption binary fields
    # DHIS2 often stores Yes/No or Positive/Negative as strings
    for col in ['had_interruption', 'tb_positive', 'opp_infection',
                'side_effects', 'stage_worsened']:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            df[col] = s.map({
                'yes': 1.0, 'true': 1.0, '1': 1.0, 'positive': 1.0,
                'confirmed': 1.0, 'worsened': 1.0, 'defaulted': 1.0,
                'no': 0.0, 'false': 0.0, '0': 0.0, 'negative': 0.0,
                'stable': 0.0, 'improved': 0.0, 'not defaulted': 0.0,
            }).combine_first(pd.to_numeric(df[col], errors='coerce'))

    return df


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI COMPONENT
# ─────────────────────────────────────────────────────────────

def render_dhis2_connector() -> Optional[pd.DataFrame]:
    """
    Renders the DHIS2 connection UI inside a Streamlit expander.
    Returns a DataFrame ready to pass into normalize_columns(),
    or None if not connected / no data pulled yet.

    Add this to the Patient Risk page in app.py, alongside the
    existing st.file_uploader, like this:

        # Existing CSV upload
        uploaded_file = st.file_uploader(...)

        # New DHIS2 direct connection
        st.markdown("**Or connect directly to your DHIS2 instance:**")
        df_dhis2 = render_dhis2_connector()
        if df_dhis2 is not None:
            df = df_dhis2
            # continue with normalize_columns(df) as normal
    """

    with st.expander("🔗 Connect directly to DHIS2 instance", expanded=False):

        st.markdown("""
        <div style="background:#0d1f2d;border:1px solid #00e5ff33;border-radius:8px;
            padding:14px 18px;margin-bottom:16px;font-size:0.88rem;color:#adbac7">
        <strong style="color:#00e5ff">Direct DHIS2 connection</strong> — pull patient data
        straight from your DHIS2 instance without downloading a CSV.
        Credentials are held in this browser session only and never stored.
        Zero patient data is written to disk.
        </div>
        """, unsafe_allow_html=True)

        # ── Connection credentials ────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            base_url = st.text_input(
                "DHIS2 Base URL",
                placeholder="https://hmis.moh.gov.ng",
                help="Your DHIS2 instance URL — no trailing slash",
                key="dhis2_base_url"
            )
        with col2:
            username = st.text_input(
                "Username",
                placeholder="your.username",
                key="dhis2_username"
            )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Your DHIS2 password",
            key="dhis2_password"
        )

        # ── Test connection ───────────────────────────────────────────────────
        if st.button("Test connection", key="dhis2_test_btn"):
            if not all([base_url, username, password]):
                st.error("Please enter URL, username, and password")
            else:
                with st.spinner("Connecting to DHIS2..."):
                    client = DHIS2Client(base_url, username, password)
                    ok, msg = client.ping()
                    if ok:
                        st.success(f"✅ {msg}")
                        st.session_state['dhis2_connected'] = True
                        st.session_state['dhis2_client_params'] = {
                            'base_url': base_url,
                            'username': username,
                            'password': password
                        }
                        # Load programmes and org units
                        programs = client.get_programs()
                        st.session_state['dhis2_programs'] = programs
                        org_units = client.get_org_units(level=3)
                        st.session_state['dhis2_org_units'] = org_units
                    else:
                        st.error(f"❌ {msg}")
                        st.session_state['dhis2_connected'] = False

        # ── Programme and org unit selection ─────────────────────────────────
        if st.session_state.get('dhis2_connected'):

            programs = st.session_state.get('dhis2_programs', [])
            org_units = st.session_state.get('dhis2_org_units', [])

            if programs:
                prog_options = {p['displayName']: p['id'] for p in programs}
                selected_prog_name = st.selectbox(
                    "Select ART programme",
                    options=list(prog_options.keys()),
                    key="dhis2_program_select"
                )
                selected_prog_id = prog_options.get(selected_prog_name, '')
            else:
                st.warning("No tracker programmes found on this instance.")
                selected_prog_id = st.text_input(
                    "Enter programme UID manually",
                    placeholder="e.g. IpHINAT79UW",
                    key="dhis2_prog_uid"
                )

            if org_units:
                ou_options = {o['displayName']: o['id'] for o in org_units}
                ou_options = {'All facilities (descendants)': 'ROOT', **ou_options}
                selected_ou_name = st.selectbox(
                    "Select organisation unit / facility",
                    options=list(ou_options.keys()),
                    key="dhis2_ou_select"
                )
                selected_ou_id = ou_options.get(selected_ou_name, '')
                if selected_ou_id == 'ROOT':
                    # Use the root org unit
                    selected_ou_id = st.text_input(
                        "Enter root org unit UID",
                        placeholder="e.g. ImspTQPwCqd",
                        key="dhis2_root_ou"
                    )
            else:
                selected_ou_id = st.text_input(
                    "Enter org unit UID manually",
                    placeholder="e.g. DiszpKrYNg8",
                    key="dhis2_ou_uid"
                )

            max_records = st.slider(
                "Maximum patients to pull",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="dhis2_max_records",
                help="Larger pulls take longer. Start with 500-1000 to test."
            )

            # ── Pull data ─────────────────────────────────────────────────────
            if st.button(
                "Pull patient data from DHIS2",
                key="dhis2_pull_btn",
                type="primary"
            ):
                if not selected_prog_id or not selected_ou_id:
                    st.error("Please select a programme and organisation unit")
                else:
                    params = st.session_state.get('dhis2_client_params', {})
                    client = DHIS2Client(
                        params['base_url'],
                        params['username'],
                        params['password']
                    )

                    with st.spinner(
                        f"Pulling up to {max_records:,} patient records from DHIS2..."
                    ):
                        # Get data element name map for this programme
                        de_map = client.get_data_elements(selected_prog_id)

                        # Pull tracked entities
                        df_raw, msg = client.get_tracked_entities(
                            program_id=selected_prog_id,
                            org_unit_id=selected_ou_id,
                            max_records=max_records
                        )

                    if df_raw.empty:
                        st.error(f"No data returned: {msg}")
                    else:
                        # Map DHIS2 columns to SmartDaaS aliases
                        df_mapped = map_dhis2_columns(df_raw, de_map)

                        st.success(f"✅ {msg}")
                        st.session_state['dhis2_df'] = df_mapped

                        # Show what was pulled and mapped
                        st.markdown(
                            f'<div style="background:#0c2014;border:1px solid #30d15877;'
                            f'border-radius:6px;padding:10px 14px;font-size:0.85rem;'
                            f'color:#30d158;margin:8px 0">'
                            f'✅ {len(df_mapped):,} patients pulled · '
                            f'{len(df_mapped.columns):,} fields extracted · '
                            f'Ready for SmartDaaS analysis</div>',
                            unsafe_allow_html=True
                        )

                        # Show column mapping preview
                        from dhis2_connector import DHIS2_FIELD_MAP
                        smartdaas_cols = [
                            c for c in df_mapped.columns
                            if c in [
                                'age', 'sex', 'gender', 'cd4_at_start',
                                'most_recent_cd4', 'who_stage', 'weight_at_start',
                                'had_interruption', 'tb_positive', 'opp_infection',
                                'side_effects', 'stage_worsened', 'days_to_art',
                                'bmi_start', 'art_start_date'
                            ]
                        ]
                        if smartdaas_cols:
                            st.markdown(
                                f"**Mapped fields:** {', '.join(f'`{c}`' for c in smartdaas_cols)}"
                            )

                        unmapped = [
                            c for c in df_mapped.columns
                            if c not in smartdaas_cols
                            and c != 'patient_id'
                            and not c.startswith('_')
                        ]
                        if unmapped[:5]:
                            st.markdown(
                                f"**Additional fields (will be passed to alias engine):** "
                                f"{', '.join(f'`{c}`' for c in unmapped[:5])}"
                                + (f" + {len(unmapped)-5} more" if len(unmapped) > 5 else "")
                            )

        # ── Return pulled data if available ───────────────────────────────────
        if 'dhis2_df' in st.session_state:
            df_result = st.session_state['dhis2_df']
            if not df_result.empty:
                st.info(
                    f"📊 {len(df_result):,} patients from DHIS2 ready for analysis. "
                    f"Scroll up and proceed with the analysis."
                )
                return df_result

    return None


# ─────────────────────────────────────────────────────────────
# INTEGRATION INSTRUCTIONS FOR app.py
# ─────────────────────────────────────────────────────────────
"""
HOW TO INTEGRATE INTO app.py
─────────────────────────────────────────────────────────────

1. Add import at the top of app.py (after existing imports):

    from dhis2_connector import render_dhis2_connector

2. In the Patient Risk page section, find the existing file uploader:

    uploaded_file = st.file_uploader("Upload your programme CSV", ...)

   Add this ABOVE or BELOW it:

    # ── DHIS2 Direct Connection ──────────────────────────────
    df_from_dhis2 = render_dhis2_connector()

3. In the data loading logic, add a check for DHIS2 data:

    # After the existing: if uploaded_file is not None:
    if df_from_dhis2 is not None:
        df = df_from_dhis2
        source_label = "dhis2_direct"
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name

4. The rest of your pipeline is UNCHANGED:
    df_mapped, missing, mappings = normalize_columns(df)
    # ... everything continues as normal

5. Add requests to requirements.txt if not already present:
    requests>=2.31.0

That's it. The connector feeds into your existing normalize_columns()
pipeline exactly like a CSV upload would.
"""
