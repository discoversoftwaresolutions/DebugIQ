from pathlib import Path

# Define a cleaned and fully structured version of debugiq_dashboard.py
dashboard_code = '''
import os
import streamlit as st
import requests
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh
from debugiq_gemini_voice import process_voice_file, process_text_command
import pandas as pd
import plotly.express as px

# --- Branding + Config ---
PROJECT_NAME = "DebugIQ"
BACKEND_URL = "https://
REPO_LINKS = {
    "GitHub (Frontend)": "https://github.com/discoversoftwaresolutions/DebugIQ-frontend",
    "GitHub (Backend)": "https://github.com/discoversoftwaresolutions/DebugIQ-backend"
}

# --- Sidebar Branding ---
st.sidebar.title(PROJECT_NAME)
st.sidebar.markdown("### 🔗 Repositories")
for name, url in REPO_LINKS.items():
    st.sidebar.markdown(f"[{name}]({url})")
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by DebugIQanalyze (GPT-4o) + DebugIQ Voice (Gemini)")
st.sidebar.markdown("Maintained by Discover Software Solutions")

# --- Main Interface ---
st.title("🧠 DebugIQ Agentic Dashboard")
st.markdown("A unified agent interface for autonomous debugging, documentation, QA, and workflow orchestration.")

# --- Tabs ---
tabs = st.tabs([
    "📄 Trace + Patch",
    "✅ QA",
    "📘 Docs",
    "📣 Issue Notices",
    "🤖 Autonomous Workflow",
    "🔍 Workflow Check",
    "📊 Metrics"
])

# -------------------------------
# 📄 Tab 1: Trace + Patch (DebugIQanalyze)
# -------------------------------
with tabs[0]:
    st.header("📄 DebugIQanalyze: Patch from Traceback")
    uploaded = st.file_uploader("Upload traceback or code file", type=["py", "txt"])
    if uploaded:
        code_text = uploaded.read().decode("utf-8")
        st.code(code_text, language="python")
        if st.button("🔧 Analyze and Patch with DebugIQanalyze"):
            with st.spinner("Calling DebugIQanalyze (GPT-4o)..."):
                res = requests.post(f"{BACKEND_URL}/suggest_patch", json={"code": code_text})
                if res.status_code == 200:
                    patch = res.json()
                    st.code(patch["diff"], language="diff")
                    st.markdown(f"💡 **Explanation:** {patch['explanation']}")
                else:
                    st.error("Patch generation failed.")
    st.subheader("🧑‍💻 Live Editor")
    st_ace(language="python", theme="twilight", height=250)

# -------------------------------
# ✅ Tab 2: QA Validation
# -------------------------------
with tabs[1]:
    st.header("✅ QA with DebugIQanalyze")
    qa_input = st.text_area("Paste updated code for validation:")
    if st.button("Run QA Validation"):
        with st.spinner("Validating with DebugIQanalyze..."):
            qa_response = requests.post(f"{BACKEND_URL}/run_qa", json={"code": qa_input})
            if qa_response.status_code == 200:
                st.json(qa_response.json())
            else:
                st.error("QA validation unavailable.")

# -------------------------------
# 📘 Tab 3: Documentation
# -------------------------------
with tabs[2]:
    st.header("📘 Auto-Documentation with DebugIQanalyze")
    doc_input = st.text_area("Paste code to generate documentation:")
    if st.button("📝 Generate Patch Doc"):
        doc_response = requests.post(f"{BACKEND_URL}/generate_doc", json={"code": doc_input})
        if doc_response.status_code == 200:
            st.markdown(doc_response.json().get("doc", "No documentation generated."))
        else:
            st.error("Doc generation failed.")

# -------------------------------
# 📣 Tab 4: Issue Notices
# -------------------------------
with tabs[3]:
    st.header("📣 Detected Issues (Autonomous Agent Summary)")
    if st.button("🔍 Fetch Notices"):
        issues = requests.get(f"{BACKEND_URL}/issues/inbox")
        if issues.status_code == 200:
            st.json(issues.json())
        else:
            st.warning("No issue data or backend error.")

# -------------------------------
# 🤖 Tab 5: Autonomous Workflow
# -------------------------------
with tabs[4]:
    st.header("🤖 DebugIQ Autonomous Agent Workflow")
    issue_id = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101")
    if st.button("▶️ Run DebugIQ Workflow"):
        if not issue_id:
            st.warning("Please enter a valid issue ID.")
        else:
            with st.spinner("Running DebugIQ agents..."):
                response = requests.post(
                    f"{BACKEND_URL}/run_autonomous_workflow",
                    json={"issue_id": issue_id}
                )
                if response.status_code == 200:
                    st.success("✅ Workflow completed")
                    st.json(response.json())
                else:
                    st.error("❌ Workflow error from backend.")

# -------------------------------
# 🔍 Tab 6: Workflow Check
# -------------------------------
with tabs[5]:
    st.header("🔍 Workflow Integrity Check")
    check = requests.get(f"{BACKEND_URL}/workflow_check")
    if check.status_code == 200:
        st.json(check.json())
    else:
        st.warning("Workflow status unavailable.")

# -------------------------------
# 📊 Tab 7: Metrics
# -------------------------------
with tabs[6]:
    st.header("📊 Agent Metrics")
    st_autorefresh(interval=30000, key="autorefresh_metrics")
    metrics = requests.get(f"{BACKEND_URL}/metrics/status")
    if metrics.status_code == 200:
        st.json(metrics.json())
    else:
        st.warning("Metrics unavailable.")

# -------------------------------
# 🧠 GPT-4o Text Command
# -------------------------------
st.markdown("---")
st.subheader("🧠 Text Command to DebugIQanalyze (GPT-4o)")
cmd = st.text_input("Enter your agent command here...")
if st.button("Send to GPT-4o"):
    with st.spinner("Processing via GPT-4o..."):
        result = process_text_command(cmd)
        if "response" in result:
            st.success("✅ Agent Response:")
            st.markdown(result["response"])
        else:
            st.error(result.get("error", "Unknown error"))

# -------------------------------
# 🎙️ Gemini Voice Command Upload
# -------------------------------
st.markdown("---")
st.subheader("🎙️ Voice Command to DebugIQ Voice Agent (Gemini)")
voice_file = st.file_uploader("Upload a .wav file", type=["wav"])
if voice_file and st.button("Send Voice to Gemini Agent"):
    with st.spinner("Processing via Gemini..."):
        audio_bytes = voice_file.read()
        result = process_voice_file(audio_bytes)
        if "response" in result:
            st.success("✅ Gemini Agent Response:")
            st.markdown(result["response"])
        else:
            st.error(result.get("error", "No response"))
'''

