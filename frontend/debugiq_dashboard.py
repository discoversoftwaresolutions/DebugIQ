# debugiq_dashboard.py
import os
import streamlit as st
import requests
from dotenv import load_dotenv
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from streamlit_autorefresh import st_autorefresh
import difflib
import pandas as pd
import plotly.express as px
from debugiq_gemini_voice import process_voice_file, process_text_command

load_dotenv()

# --- Config ---
PROJECT_NAME = "DebugIQ"
BACKEND_URL = "https://autonomous-debug.onrender.com"
REPO_LINKS = {
    "GitHub (Frontend)": "https://github.com/discoversoftwaresolutions/DebugIQ-frontend",
    "GitHub (Backend)": "https://github.com/discoversoftwaresolutions/DebugIQ-backend"
}

# --- Sidebar ---
st.sidebar.title(PROJECT_NAME)
st.sidebar.markdown("### 🔗 Repositories")
for name, url in REPO_LINKS.items():
    st.sidebar.markdown(f"[{name}]({url})")
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by DebugIQanalyze (GPT-4o) + DebugIQ Voice (Gemini)")
st.sidebar.markdown("Maintained by Discover Software Solutions")

# --- Main UI ---
st.title("🧠 DebugIQ Agentic Dashboard")
st.markdown("Autonomous debugging, QA, doc generation, and workflow orchestration.")

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

# --- Tab 0: Trace + Patch ---
with tabs[0]:
    st.header("📄 DebugIQanalyze: Patch from Traceback")
    uploaded_file = st.file_uploader("Upload traceback or .py file", type=["py", "txt"])
    if uploaded_file:
        original_code = uploaded_file.read().decode("utf-8")
        st.code(original_code, language="python")
        if st.button("🧠 Suggest Patch"):
            with st.spinner("Calling DebugIQanalyze..."):
                res = requests.post(f"{BACKEND_URL}/debugiq/suggest_patch", json={"code": original_code})
                if res.status_code == 200:
                    patch = res.json()
                    patched_code = patch.get("patched_code", "") or patch.get("suggested_patch", "")
                    html_diff = difflib.HtmlDiff().make_file(
                        original_code.splitlines(),
                        patched_code.splitlines(),
                        fromdesc="Original",
                        todesc="Patched"
                    )
                    st.components.v1.html(html_diff, height=450, scrolling=True)
                    edited_code = st_ace(value=patched_code, language="python", theme="monokai", height=300)
                    st.session_state.edited_patch = edited_code
                    st.success("✅ Patch displayed below. You can edit and pass it to QA.")
                else:
                    st.error("Failed to generate patch.")

# --- Tab 1: QA ---
with tabs[1]:
    st.header("✅ QA Validation")
    qa_input = st.text_area("Paste updated code for validation:")
    if st.button("Run QA Validation"):
        res = requests.post(f"{BACKEND_URL}/run_qa", json={"code": qa_input})
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("QA validation unavailable.")

# --- Tab 2: Docs ---
with tabs[2]:
    st.header("📘 Auto-Documentation")
    doc_input = st.text_area("Paste code to generate documentation:")
    if st.button("📝 Generate Patch Doc"):
        doc_response = requests.post(f"{BACKEND_URL}/generate_doc", json={"code": doc_input})
        if doc_response.status_code == 200:
            st.markdown(doc_response.json().get("doc", "No documentation generated."))
        else:
            st.error("Doc generation failed.")

# --- Tab 3: Issue Notices ---
with tabs[3]:
    st.header("📣 Detected Issues")
    if st.button("🔍 Fetch Notices"):
        issues = requests.get(f"{BACKEND_URL}/issues/inbox")
        if issues.status_code == 200:
            st.json(issues.json())
        else:
            st.warning("No issue data or backend error.")

# --- Tab 4: Autonomous Workflow ---
with tabs[4]:
    st.header("🤖 Autonomous Agent Workflow")
    issue_id = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101")
    if st.button("Run DebugIQ Workflow"):
        if not issue_id:
            st.warning("Please enter a valid issue ID.")
        else:
            with st.spinner("Running DebugIQ agents..."):
                response = requests.post(f"{BACKEND_URL}/run_autonomous_workflow", json={"issue_id": issue_id})
                if response.status_code == 200:
                    result = response.json()
                    st.json(result)
                else:
                    st.error("❌ Workflow error from backend.")

# --- Tab 5: Workflow Check ---
with tabs[5]:
    st.header("🔍 Workflow Integrity Check")
    check = requests.get(f"{BACKEND_URL}/workflow_check")
    if check.status_code == 200:
        st.json(check.json())
    else:
        st.warning("Workflow status unavailable.")

# --- Tab 6: Metrics ---
with tabs[6]:
    st.header("📊 Agent Metrics")
    st_autorefresh(interval=30000, key="autorefresh_metrics")
    metrics = requests.get(f"{BACKEND_URL}/metrics/status")
    if metrics.status_code == 200:
        data = pd.DataFrame(metrics.json())
        st.dataframe(data)
        fig = px.bar(data, x="agent", y="count", title="Agent Activity")
        st.plotly_chart(fig)
    else:
        st.warning("Metrics endpoint unavailable.")

# --- Text Agent (GPT-4o) ---
def gpt4o_text_agent_ui():
    st.subheader("🧠 Text Command to DebugIQanalyze (GPT-4o)")
    cmd = st.text_input("Enter your agent command here:")
    if st.button("Send to GPT-4o"):
        with st.spinner("Processing via GPT-4o..."):
            result = process_text_command(cmd)
            if "response" in result:
                st.success("✅ Agent Response:")
                st.markdown(result["response"])
            else:
                st.error(result.get("error", "Unknown error"))

# --- Voice Agent (Gemini) ---
def gemini_voice_agent_ui():
    st.subheader("🎧 Voice Command to DebugIQ Voice Agent (Gemini)")
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

# --- Render Voice + Text Agent UI ---
st.markdown("---")
gpt4o_text_agent_ui()
gemini_voice_agent_ui()
