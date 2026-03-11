import streamlit as st
import requests
import json
import pandas as pd

# Page Configuration
st.set_page_config(page_title="HR Recruitment - HR Intelligence Platform", layout="wide")

# Custom CSS to vaguely match the original look and feel
st.markdown("""
<style>
    /* Styling adjustments to match original darkish theme where possible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 50px;
        background-color: transparent;
        border: 1px solid #334155;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-color: #6366f1;
    }
    .title-gradient {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .status-indicator {
        padding: 0.25rem 0.75rem;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border-radius: 50px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title-gradient">HR Recruitment</div>', unsafe_allow_html=True)

# API Base URL
API_BASE = "http://localhost:8000"

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Unified Recruiter", "CV Extract", "Resume Ranking"])

# --- TAB 1: Unified Recruiter ---
with tab1:
    st.subheader("HR Pipeline")
    st.write("Upload multiple resumes and provide a Job Description to get ranked candidates with AI suggestions in one click.")
    
    unified_jd = st.text_area("Job Description", placeholder="Paste the job description here...", key="unified_jd", height=150)
    unified_files = st.file_uploader("Select Resumes (PDF, Multiple allowed)", type=["pdf"], accept_multiple_files=True, key="unified_files", help="You can select multiple PDF files at once.")
    
    if st.button("Process All Candidates", type="primary"):
        if not unified_files:
            st.error("Please select at least one PDF resume.")
        elif not unified_jd.strip():
            st.error("Please paste the job description.")
        else:
            with st.spinner("Processing..."):
                try:
                    files_payload = []
                    for f in unified_files:
                        files_payload.append(("files", (f.name, f.getvalue(), "application/pdf")))
                    data_payload = {"job_description": unified_jd}
                    
                    response = requests.post(f"{API_BASE}/unified/pipeline", data=data_payload, files=files_payload)
                    
                    if response.ok:
                        data = response.json()
                        count = data.get("count", 0)
                        st.markdown(f"### Ranked Results ({count} Candidates)")
                        
                        if data.get("results"):
                            for cand in data.results:
                                score = cand.get("total_score", 0)
                                if score >= 70:
                                    color = "#10b981"
                                elif score >= 50:
                                    color = "#f59e0b"
                                else:
                                    color = "#ef4444"
                                    
                                score_html = f"<div style='font-size:1.5rem;font-weight:700;color:{color};text-align:right;'>{score}%</div>"
                                
                                st.markdown(f"""
                                <div style="margin-bottom:1.5rem;border:1px solid #334155;padding:1.25rem;border-radius:12px;background:rgba(255,255,255,0.02); display:flex; justify-content:space-between;">
                                    <div>
                                        <span style="font-size:1.25rem;font-weight:700;color:#6366f1">#{cand.get('rank')}</span>
                                        <h3 style="display:inline;margin-left:10px;">{cand.get('name', 'Unknown')}</h3>
                                        <p style="font-size:0.9rem;color:#94a3b8;margin:0;">{cand.get('email', '')}</p>
                                    </div>
                                    <div>
                                        {score_html}
                                        <div style="font-size:0.75rem;color:#94a3b8;text-align:right;">Match Score</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        elif data.get("status") == "error":
                            st.error(data.get("message", "No results returned."))
                    else:
                        st.error(f"Server error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")


# --- TAB 2: CV Extract ---
with tab2:
    st.subheader("Resume Extraction & Analysis")
    st.write("Upload a resume (PDF) to extract key information and get AI-powered insights.")
    
    cv_file = st.file_uploader("Select Resume (PDF)", type=["pdf"], key="cv_file")
    include_analysis = st.checkbox("Include AI Analysis", key="include_analysis")
    
    if st.button("Extract & Analyze", type="primary"):
        if not cv_file:
            st.error("Please select a PDF file.")
        else:
            with st.spinner("Extracting..."):
                try:
                    files_payload = {"file": (cv_file.name, cv_file.getvalue(), "application/pdf")}
                    data_payload = {"include_analysis": str(include_analysis).lower()}
                    
                    response = requests.post(f"{API_BASE}/cv_extract/extract", data=data_payload, files=files_payload)
                    
                    if response.ok:
                        data = response.json()
                        st.json(data)
                    else:
                        st.error(f"Server error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"CV Extraction Error: {str(e)}")


# --- TAB 3: Resume Ranking ---
with tab3:
    st.subheader("Resume Ranking")
    st.write("Rank multiple resumes against a specific job description using our hybrid AI ranking system.")
    
    rank_jd = st.text_area("Job Description", placeholder="Paste the job description here...", key="rank_jd", height=100)
    rank_cand_data = st.text_area("Candidate Resumes (JSON Format)", placeholder='[{"name": "John Doe", "text": "Resume text..."}]', key="rank_cand_data", height=150)
    st.caption("Or upload multiple files... (Work in Progress)")
    
    if st.button("Rank Candidates", type="primary"):
        if not rank_jd.strip():
            st.error("Please provide a job description.")
        elif not rank_cand_data.strip():
            st.error("Please provide candidate data in JSON format.")
        else:
            try:
                candidates_json = json.loads(rank_cand_data)
                with st.spinner("Ranking..."):
                    payload = {
                        "job_description": rank_jd,
                        "candidates": candidates_json
                    }
                    response = requests.post(f"{API_BASE}/resume_ranking/rank", json=payload)
                    
                    if response.ok:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        
                        if candidates:
                            for cand in candidates:
                                st.markdown(f"""
                                <div style="background:#1e293b;border:1px solid #334155;padding:1rem;border-radius:8px;display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;">
                                    <div style="font-size:1.5rem;font-weight:700;color:#6366f1;min-width:40px;">#{cand.get('rank')}</div>
                                    <div style="flex:1;">
                                        <h3 style="margin:0;">{cand.get('name', 'Unknown')}</h3>
                                        <p style="margin:0;color:#94a3b8;">{cand.get('email', '')} | {cand.get('phone', '')}</p>
                                        <small style="color:#94a3b8;">{cand.get('skills_preview', '')}</small>
                                    </div>
                                    <div style="text-align:right;font-weight:700;color:#10b981;font-size:1.2rem;">{round(cand.get('total_score', 0))}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No candidates ranked.")
                    else:
                        st.error(f"Server error: {response.status_code} - {response.text}")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Candidate Data field.")
            except Exception as e:
                st.error(f"Ranking Error: {str(e)}")

# Server Status Checker
server_status = "Server Offline"
status_color = "rgba(239, 68, 68, 0.1); color: #ef4444;"
try:
    health = requests.get(f"{API_BASE}/", timeout=2)
    if health.ok and health.json().get("status") == "running":
        server_status = "Server Online"
        status_color = "rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2);"
except:
    pass

st.markdown(f'<div style="padding: 0.25rem 0.75rem; background: {status_color.split(";")[0]}; color: {status_color.split(";")[1].split(":")[1]}; border-radius: 50px; display: inline-block; font-size: 0.85rem; margin-top: 2rem;">{server_status}</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: 10px;'>&copy; HR Recruitment</p>", unsafe_allow_html=True)
