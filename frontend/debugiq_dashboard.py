# DebugIQ Dashboard

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode
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

def fetch_github_data(url):
    headers = {"Authorization": f"Bearer {os.getenv('GITHUB_OAUTH_TOKEN', '')}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"GitHub API Error: {e}")
        return []

def create_pull_request(issue_id, branch_name, code_diff, diagnosis_details, validation_results):
    logger.info(f"Creating PR for issue {issue_id}...")
    return {
        "url": f"https://github.com/fake-org/repo/pull/{issue_id}",
        "title": f"Fix for {issue_id}",
        "body": f"Branch: {branch_name}\n\nPatch Diff:\n{code_diff}\n\nDiagnosis Details:\n{diagnosis_details}\n\nValidation Results:\n{validation_results}",
    }

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

        # Fetch Branches
        branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
        branches = fetch_github_data(branches_url)
        branch_names = [branch["name"] for branch in branches]

        if branch_names:
            selected_branch = st.sidebar.selectbox("Select Branch", branch_names)
        else:
            st.sidebar.error("No branches found or access error.")

        # Directory Navigation
        if selected_branch:
            path = st.sidebar.text_input("Path", "/", help="Enter the path to navigate the repository")
            contents_url = f"https://api.github.com/repos/{owner}/{repo}/contents{path}?ref={selected_branch}"
            contents = fetch_github_data(contents_url)

            if contents:
                # Display Directories
                st.sidebar.markdown("### Directories")
                for item in contents:
                    if item["type"] == "dir":
                        if st.sidebar.button(f"📁 {item['name']}"):
                            st.experimental_set_query_params(path=item["path"])

                # Display Files
                st.sidebar.markdown("### Files")
                for item in contents:
                    if item["type"] == "file":
                        if st.sidebar.button(f"📄 {item['name']}"):
                            file_content = requests.get(item["download_url"]).text
                            st.session_state["file_content"] = file_content
                            st.session_state["file_name"] = item["name"]
            else:
                st.sidebar.error("No contents found or access error.")
    else:
        st.sidebar.error("Invalid GitHub URL.")
else:
    st.sidebar.info("Enter a GitHub repository URL to begin.")

# === Application Tabs ===
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Check", "📈 Metrics", "🎙️ Voice Agent"])
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics, tab_voice = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    trace = st.text_area("Traceback Input", height=200, placeholder="Paste traceback here...")
    source_code = st.text_area("Source Code Input", height=200, placeholder="Paste source code here...")
    if st.button("🔬 Analyze & Suggest Patch"):
        payload = {"trace": trace, "source_code": source_code}
        response = make_api_request("POST", ENDPOINTS["suggest_patch"], payload)
        if "error" not in response:
            st.text_area("Suggested Patch", value=response.get("suggested_patch", ""), height=200)
        else:
            st.error(response["error"])

# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    patch_code = st.text_area("Patch Code", height=200, placeholder="Paste patch code for QA validation...")
    if st.button("🛡️ Validate Patch"):
        payload = {"patch_code": patch_code}
        response = make_api_request("POST", ENDPOINTS["qa_validation"], payload)
        if "error" not in response:
            st.json(response)
        else:
            st.error(response["error"])

# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    code = st.text_area("Code Input", height=200, placeholder="Paste code for documentation...")
    if st.button("📝 Generate Documentation"):
        payload = {"code": code}
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
    issue_id = st.text_input("Issue ID", placeholder="Enter issue ID to trigger workflow...")
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
    
    # Display chat history
    st.markdown("### Chat History")
    for message in st.session_state.get("chat_history", []):
        role = "User" if message["role"] == "user" else "Gemini"
        st.markdown(f"**{role}:** {message['content']}")
        if "audio_base64" in message:
            try:
                audio_bytes = base64.b64decode(message["audio_base64"])
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                logger.error(f"Error decoding audio: {e}")
    
    # Initialize WebRTC streamer for voice input
    st.markdown("### 🎤 Start/Stop Microphone")
    ctx = webrtc_streamer(
        key="voice-agent-stream",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
    )
    
    if ctx and ctx.audio_receiver:
        st.info("🎙️ Microphone is active. Speak now.")
        
        # Process audio frames
        while ctx.audio_receiver:
            try:
                audio_frame = ctx.audio_receiver.get_frame()
                audio_bytes = audio_frame.to_ndarray().tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                
                # Send audio to Gemini Chat for transcription and response
                payload = {"audio_base64": audio_b64}
                response = make_api_request("POST", ENDPOINTS["voice_transcribe"], payload)
                
                if "error" not in response:
                    transcription = response.get("transcription", "Unclear speech")
                    st.session_state.chat_history.append({"role": "user", "content": transcription})
                    
                    # Send transcription to Gemini for a response
                    gemini_response = make_api_request("POST", ENDPOINTS["gemini_chat"], {"message": transcription})
                    if "error" not in gemini_response:
                        gemini_message = gemini_response.get("response", "No response from Gemini.")
                        st.session_state.chat_history.append({"role": "assistant", "content": gemini_message})
                    else:
                        st.error("Failed to get response from Gemini.")
                else:
                    st.error(response["error"])
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
    else:
        st.warning("Microphone is inactive. Click 'Start' to activate.")
