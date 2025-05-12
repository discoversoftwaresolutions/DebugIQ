# dashboard.py
import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import numpy as np
import av
import streamlit.components.v1 as components
import wave
import json
import logging
import base64  # For handling audio data if sent as base64
import re  # For GitHub URL parsing

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Constants ===
DEFAULT_BACKEND_URL = "https://debugiq-backend.railway.app"
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)

DEFAULT_VOICE_SAMPLE_RATE = 16000
DEFAULT_VOICE_SAMPLE_WIDTH = 2  # 16-bit audio (2 bytes)
DEFAULT_VOICE_CHANNELS = 1  # Mono
AUDIO_PROCESSING_THRESHOLD_SECONDS = 2

TRACEBACK_EXTENSION = ".txt"
SUPPORTED_SOURCE_EXTENSIONS = (
    ".py", ".js", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php",
    ".html", ".css", ".md", ".ts", ".tsx", ".json", ".yaml", ".yml",
    ".sh", ".R", ".swift", ".kt", ".scala"
)
RECOGNIZED_FILE_EXTENSIONS = SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)

# === Streamlit Page Configuration ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# === Helper Functions ===
def clear_all_github_session_state():
    """Resets all GitHub-related session state and clears loaded analysis files."""
    github_keys = [
        "github_repo_url_input", "current_github_repo_url", "github_branches",
        "github_selected_branch", "github_path_stack", "github_repo_owner", "github_repo_name"
    ]
    for key in github_keys:
        st.session_state.pop(key, None)

    # Reset analysis results
    st.session_state["analysis_results"] = {
        "trace": None,
        "source_files_content": {}
    }
    logger.info("Cleared all GitHub session state and related analysis inputs.")


def make_api_request(method, url, json_payload=None, files=None, operation_name="API Call"):
    """Makes a generic API request and handles exceptions."""
    try:
        logger.info(f"Making {method} request to {url} for {operation_name}...")
        if files:
            response = requests.request(method, url, files=files, data=json_payload, timeout=30)
        else:
            response = requests.request(method, url, json=json_payload, timeout=30)
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.warning(f"{operation_name} response not JSON. Status: {response.status_code}, Content: {response.text[:100]}")
            return {"status_code": response.status_code, "content": response.text}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"RequestException for {operation_name} to {url}: {req_err}")
        st.error(f"Communication error for {operation_name}: {req_err}")
        return {"error": str(req_err)}
    except Exception as e:
        logger.exception(f"Unexpected error during {operation_name} to {url}")
        st.error(f"Unexpected error with {operation_name}: {e}")
        return {"error": str(e)}


# === Session State Initialization ===
session_defaults = {
    "audio_sample_rate": DEFAULT_VOICE_SAMPLE_RATE,
    "audio_sample_width": DEFAULT_VOICE_SAMPLE_WIDTH,
    "audio_num_channels": DEFAULT_VOICE_CHANNELS,
    "audio_buffer": b"",
    "audio_frame_count": 0,
    "chat_history": [],
    "edited_patch": "",
    "github_repo_url_input": "",
    "current_github_repo_url": None,
    "github_branches": [],
    "github_selected_branch": None,
    "github_path_stack": [""],
    "github_repo_owner": None,
    "github_repo_name": None,
    "analysis_results": {
        "trace": None,
        "source_files_content": {},
        "patch": None,
        "explanation": None,
        "doc_summary": None,
        "patched_file_name": None,
        "original_patched_file_content": None
    },
    "qa_result": None,
    "inbox_data": None,
    "workflow_status_data": None,
    "metrics_data": None,
    "qa_code_to_validate": None
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# === Layout Components ===
st.sidebar.markdown("### 📦 GitHub Integration")
st.sidebar.text_input(
    "Public GitHub URL",
    placeholder="https://github.com/user/repo",
    key="github_repo_url_input"
)

if st.sidebar.button("Load/Reset GitHub Repo"):
    if not st.session_state.github_repo_url_input:
        clear_all_github_session_state()
    else:
        st.session_state.current_github_repo_url = None
        st.session_state.github_branches = []
        st.session_state.github_selected_branch = None
        st.session_state.github_path_stack = [""]
        st.session_state.github_repo_owner = None
        st.session_state.github_repo_name = None
        clear_all_github_session_state()

# === Main Application Logic ===
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📊 Metrics"])
with tabs[0]:
    st.header("📄 Traceback & Patch Analysis")
    # Add logic for analyzing traceback and generating patches...
