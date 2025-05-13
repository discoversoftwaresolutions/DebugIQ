# DebugIQ Dashboard with Code Editor and Diff View

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import logging
import base64
import re

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app")
ENDPOINTS = {
    "suggest_patch": f"{BACKEND_URL}/debugiq/suggest_patch",
    "qa_validation": f"{BACKEND_URL}/qa/run_qa",
    "doc_generation": f"{BACKEND_URL}/doc/generate_doc",
    "issues_inbox": f"{BACKEND_URL}/issues_inbox",
    "workflow_run": f"{BACKEND_URL}/workflow/run",
    "workflow_status": f"{BACKEND_URL}/workflow/status",
    "system_metrics": f"{BACKEND_URL}/system_metrics",
    "voice_transcribe": f"{BACKEND_URL}/transcribe_audio",
    "gemini_chat": f"{BACKEND_URL}/gemini_chat",
}

# === Helper Functions ===
def make_api_request(method, url, payload=None):
    try:
        response = requests.request(method, url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"error": str(e)}

# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# === Sidebar for GitHub Integration ===
st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
if github_url:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", github_url)
    if match:
        owner, repo = match.groups()
        st.sidebar.success(f"**Repository:** {repo} (Owner: {owner})")
    else:
        st.sidebar.error("Invalid GitHub URL.")

# === Application Tabs ===
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Check", "📈 Metrics", "🎙️ Voice Agent"])
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics, tab_voice = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"])
    
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("Original Code", value=file_content, height=200, disabled=True)
    
        if st.button("🔬 Analyze & Suggest Patch"):
            payload = {"trace": file_content}
            response = make_api_request("POST", ENDPOINTS["suggest_patch"], payload)
            if "error" not in response:
                suggested_patch = response.get("suggested_patch", "")
                st.text_area("Suggested Patch (Read-Only)", value=suggested_patch, height=200, disabled=True)

                # Code Editor for Editing Patch
                st.markdown("### ✍️ Edit Suggested Patch")
                edited_patch = st_ace(
                    value=suggested_patch,
                    language="python",
                    theme="monokai",
                    height=300,
                    key="ace_editor_patch"
                )

                # Diff View
                st.markdown("### 🔍 Diff View (Original vs. Edited Patch)")
                if edited_patch and file_content:
                    diff_view = difflib.HtmlDiff(wrapcolumn=80).make_table(
                        fromlines=file_content.splitlines(),
                        tolines=edited_patch.splitlines(),
                        fromdesc="Original Code",
                        todesc="Edited Patch",
                        context=True
                    )
                    st.components.v1.html(diff_view, height=400, scrolling=True)
            else:
                st.error(response["error"])

# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    uploaded_patch = st.file_uploader("Upload Patch File", type=["txt", "py", "java", "js"])
    
    if uploaded_patch:
        patch_content = uploaded_patch.read().decode("utf-8")
        st.text_area("Patch Content", value=patch_content, height=200, disabled=True)
    
    if st.button("🛡️ Validate Patch"):
        payload = {"patch_code": patch_content}
        response = make_api_request("POST", ENDPOINTS["qa_validation"], payload)
        if "error" not in response:
            st.json(response)
        else:
            st.error(response["error"])

# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    uploaded_code = st.file_uploader("Upload Code File for Documentation", type=["txt", "py", "java", "js"])
    
    if uploaded_code:
        code_content = uploaded_code.read().decode("utf-8")
        st.text_area("Code Content", value=code_content, height=200, disabled=True)
    
    if st.button("📝 Generate Documentation"):
        payload = {"code": code_content}
        response = make_api_request("POST", ENDPOINTS["doc_generation"], payload)
        if "error" not in response:
            st.markdown(response.get("documentation", "No documentation generated."))
        else:
            st.error(response["error"])

# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues")
    if st.button("🔄 Refresh Issues"):
        response = make_api_request("GET", ENDPOINTS["issues_inbox"])
        if "error" not in response:
            for issue in response.get("issues", []):
                with st.expander(f"Issue {issue['id']} - {issue['title']}"):
                    st.json(issue)

# === Workflow Tab ===
with tab_workflow:
    st.header("🤖 Workflow Trigger")
    issue_id = st.text_input("Issue ID to Trigger Workflow")
    if st.button("▶️ Trigger Workflow"):
        payload = {"issue_id": issue_id}
        response = make_api_request("POST", ENDPOINTS["workflow_run"], payload)
        if "error" not in response:
            st.success(f"Workflow triggered for Issue {issue_id}.")
        else:
            st.error(response["error"])

# === Workflow Check Tab ===
with tab_status:
    st.header("🔍 Workflow Check")
    if st.button("🔄 Refresh Workflow Status"):
        response = make_api_request("GET", ENDPOINTS["workflow_status"])
        if "error" not in response:
            st.json(response)
        else:
            st.error(response["error"])

# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    if st.button("📊 Fetch Metrics"):
        response = make_api_request("GET", ENDPOINTS["system_metrics"])
        if "error" not in response:
            st.json(response)
        else:
            st.error(response["error"])

# === Voice Agent Tab ===
with tab_voice:
    st.header("🎙️ Voice Agent")
    ctx = webrtc_streamer(
        key="voice-agent",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
    )
    if ctx and ctx.audio_receiver:
        st.info("🎙️ Microphone is active. Speak now.")
