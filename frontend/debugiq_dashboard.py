# File: frontend/debugiq_dashboard.py

# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# Imports for Voice Agent section
import av  # Required for processing audio frames from streamlit-webrtc
import numpy as np  # Required for processing audio frames
import io  # Required for in-memory WAV file creation
import wave  # Required for WAV file creation
from streamlit_webrtc import webrtc_streamer, WebRtcMode  # Needed for voice
import logging  # Explicit logging import
import base64  # For voice/image encoding
import re  # For GitHub URL parsing
import threading  # For thread-safe buffers if needed
from urllib.parse import urljoin  # Construct robust URLs

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app")

ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",
    "qa_validation": "/qa/run",
    "doc_generation": "/doc/generate",
    "issues_inbox": "/issues/attention-needed",
    "workflow_run": "/workflow/run_autonomous_workflow",
    "workflow_status": "/issues/{issue_id}/status",
    "system_metrics": "/metrics/status",
    "voice_transcribe": "/voice/transcribe",
    "gemini_chat": "/gemini/chat",
    "tts": "/voice/tts"
}

# === Helper Functions ===
def make_api_request(method, endpoint_key, payload=None, return_json=True):
    """Construct API URL and make a request."""
    path_template = ENDPOINTS[endpoint_key]
    issue_id = st.session_state.get("active_issue_id") if endpoint_key == "workflow_status" else None
    try:
        path = path_template.format(issue_id=issue_id) if issue_id else path_template
        url = urljoin(BACKEND_URL, path)
        logger.info(f"Making API request: {method} {url}")
        response = requests.request(method, url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json() if return_json else response.content
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": str(e)}

# === Audio Processing Helper ===
def frames_to_wav_bytes(frames):
    """Convert audio frames to WAV bytes."""
    if not frames:
        return None
    try:
        frame_0 = frames[0]
        sample_rate = frame_0.sample_rate
        channels = frame_0.layout.channels
        sample_width_bytes = frame_0.format.bytes
        raw_data = b"".join([frame.planes[0].buffer.tobytes() for frame in frames])
    except Exception as e:
        logger.error(f"Error processing audio frames: {str(e)}")
        return None

    try:
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width_bytes)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_data)
            return wav_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error creating WAV file: {str(e)}")
        return None

# === WebRTC Audio Frame Callback ===
def audio_frame_callback(frame):
    """Callback for processing audio frames."""
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []
    if "audio_buffer_lock" not in st.session_state:
        st.session_state.audio_buffer_lock = threading.Lock()
    with st.session_state.audio_buffer_lock:
        if st.session_state.get('is_recording', False):
            st.session_state.audio_buffer.append(frame)

# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ")

# Initialize session state
for key, default in {
    'is_recording': False,
    'audio_buffer': [],
    'audio_buffer_lock': threading.Lock(),
    'recording_status': "Idle",
    'chat_history': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar for GitHub Integration
st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")
if github_url:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", github_url)
    if match:
        owner, repo = match.groups()
        st.sidebar.success(f"**Repository:** {repo} (Owner: {owner})")
    else:
        st.sidebar.error("Invalid GitHub URL.")

# Tabs
tabs = st.tabs([
    "📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation",
    "📣 Issues", "🤖 Workflow", "🔍 Workflow Status", "📈 Metrics"
])
(tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics) = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"])
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.subheader("Original Code")
        st.code(file_content, language="plaintext")
        if st.button("🔬 Analyze & Suggest Patch"):
            response = make_api_request("POST", "suggest_patch", {"code": file_content})
            if "error" not in response:
                st.code(response.get("diff", "No diff suggested."), language="diff")

# === Add other tabs similarly ===

# Placeholder for other tabs (QA Validation, Documentation, etc.)
