# dashboard.py

import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings # Import ClientSettings
import numpy as np
import av
import streamlit.components.v1 as components
import wave
import json
import logging
import base64
import re
from datetime import datetime # Added import for datetime used in backend snippet

# === Logging Setup ===
# Configure logging to see output in your terminal or deployment logs
# This logging is essential for debugging the rerun loop cause.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add debug logs to trace reruns and state
logger.info("--- Script Rerun Start ---")
logger.info(f"Session State Keys at start: {list(st.session_state.keys())}") # Log keys to see what's in state
logger.info(f"Initial github_repo_url_input: '{st.session_state.get('github_repo_url_input', '').strip()}'")
logger.info(f"Initial current_github_repo_url: '{st.session_state.get('current_github_repo_url')}'")
logger.info(f"Initial github_selected_branch: '{st.session_state.get('github_selected_branch')}'")
logger.info(f"Initial github_path_stack: {st.session_state.get('github_path_stack')}")


# === Constants ===
# Set the default backend URL based on your Railway deployment
DEFAULT_BACKEND_URL = "https://debugiq-backend.railway.app"
# Use environment variable BACKEND_URL if set, otherwise use the default
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)

# Define backend API endpoint URLs based on main.py routing
# Structure: f"{BACKEND_URL}/[prefix_from_main.py]/[path_in_router_file]"
# !!! IMPORTANT: Verify the exact paths within your router files (e.g., analyze.py, qa.py, etc.)
SUGGEST_PATCH_URL = f"{BACKEND_URL}/debugiq/suggest_patch" # Prefix /debugiq from main.py + assumed path /suggest_patch
QA_URL = f"{BACKEND_URL}/qa/run_qa" # Prefix /qa from main.py + assumed path /run_qa
DOC_URL = f"{BACKEND_URL}/doc/generate_doc" # Prefix /doc from main.py + assumed path /generate_doc
ISSUES_INBOX_URL = f"{BACKEND_URL}/issues_inbox" # No prefix + assumed path /issues_inbox
WORKFLOW_RUN_URL = f"{BACKEND_URL}/workflow/run" # Prefix /workflow from main.py + assumed path /run
WORKFLOW_CHECK_URL = f"{BACKEND_URL}/workflow/status" # Prefix /workflow from main.py + assumed path /status
METRICS_URL = f"{BACKEND_URL}/system_metrics" # No prefix + assumed path /system_metrics

# Voice/Chat related endpoints - verify paths in your voice/chat router files (e.g., voice_ws_router.py)
# Assuming these paths are defined directly without prefixes in the included voice router
COMMAND_URL = f"{BACKEND_URL}/command"
VOICE_TRANSCRIBE_URL = f"{BACKEND_URL}/transcribe_audio"
GEMINI_CHAT_URL = f"{BACKEND_URL}/gemini_chat"


DEFAULT_VOICE_SAMPLE_RATE = 16000
DEFAULT_VOICE_SAMPLE_WIDTH = 2  # Corrected space here
DEFAULT_VOICE_CHANNELS = 1  # Mono
AUDIO_PROCESSING_THRESHOLD_SECONDS = 2

TRACEBACK_EXTENSION = ".txt"
SUPPORTED_SOURCE_EXTENSIONS = (
    ".py", ".js", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php",
    ".html", ".css", ".md", ".ts", ".tsx", ".json", ".yaml", ".yml",
    ".sh", ".R", ".swift", ".kt", ".scala"
)
RECOGNIZED_FILE_EXTENSIONS = SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)
RECOGNIZED_FILE_EXTENSIONS = SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)

# === Helper Functions ===

def make_api_request(method, url, json_payload=None, files=None, operation_name="API Request"):
    """
    Placeholder function to make API requests to the backend.
    Replace with your actual implementation (e.g., using requests, handling auth, etc.).
    """
    logger.info(f"Attempting to make {method} request to {url} for {operation_name}")
    try:
        response = requests.request(method, url, json=json_payload, files=files, timeout=30)
        logger.info(f"{operation_name} response status code: {response.status_code}")
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Successful {method} request to {url}")
        # Attempt to return JSON, handle cases where response might be empty or not JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.warning(f"Response for {url} was not JSON. Status: {response.status_code}")
            # Return a success indicator even if no JSON, if status code is 2xx
            if 200 <= response.status_code < 300:
                return {"success": True, "message": "Request successful, no JSON response."}
            else:
                return {"error": True, "details": f"Non-JSON response, status code: {response.status_code}"}

    except requests.exceptions.Timeout:
        st.error(f"⏰ {operation_name} failed: Request timed out. URL: {url}")
        logger.error(f"{operation_name} timed out: {url}")
        return {"error": True, "details": "Request timed out."}
    except requests.exceptions.RequestException as e:
        st.error(f"❌ {operation_name} failed: {e}. URL: {url}")
        logger.error(f"{operation_name} failed for {url}: {e}")
        return {"error": True, "details": str(e)}


def clear_github_selection_state():
    """Resets only GitHub-related selection state in the sidebar."""
    logger.info("Clearing GitHub selection state.")
    # Do NOT clear github_repo_url_input here, let the text_input widget manage its value
    st.session_state.current_github_repo_url = None
    st.session_state.github_branches = []
    st.session_state.github_selected_branch = None
    st.session_state.github_path_stack = [""]
    st.session_state.github_repo_owner = None
    st.session_state.github_repo_name = None
    # Note: This does NOT clear analysis results
    # logger.info(f"GitHub selection state after clearing: {st.session_state.get('current_github_repo_url')}, {st.session_state.get('github_branches')}, {st.session_state.get('github_path_stack')}")


def clear_analysis_inputs():
    """Clears loaded traceback and source files used as analysis inputs."""
    logger.info("Clearing analysis inputs (trace and sources).")
    st.session_state.analysis_results['trace'] = None
    st.session_state.analysis_results['source_files_content'] = {}
    # logger.info(f"Analysis inputs state after clearing: {st.session_state.analysis_results.get('trace')}, {st.session_state.analysis_results.get('source_files_content')}")


def clear_analysis_outputs():
    """Clears generated patch, explanation, and other analysis outputs."""
    logger.info("Clearing analysis outputs (patch, explanation, etc.).")
    st.session_state.analysis_results.update({
        'patch': None,
        'explanation': None,
        'doc_summary': None,  # Assuming doc_summary comes from analysis results
        'patched_file_name': None,
        'original_patched_file_content': None
     })
    st.session_state.edited_patch = "" # Also clear any edited patch state
    # logger.info(f"Analysis outputs state after clearing: {st.session_state.analysis_results.get('patch')}, {st.session_state.analysis_results.get('explanation')}")


# === Session State Initialization ===
session_defaults = {
    "audio_sample_rate": DEFAULT_VOICE_SAMPLE_RATE,
    "audio_sample_width": DEFAULT_VOICE_SAMPLE_WIDTH,
    "audio_num_channels": DEFAULT_VOICE_CHANNELS,
    "audio_buffer": b"",
    "audio_frame_count": 0,
    "chat_history": [],
    "edited_patch": "",
    "github_repo_url_input": "", # Stores the value of the text input widget
    "current_github_repo_url": None, # Stores the URL that branches were successfully loaded for
    "github_branches": [],
    "github_selected_branch": None,
    "github_path_stack": [""], # Stack to manage current directory path in GitHub browser
    "github_repo_owner": None,
    "github_repo_name": None,
    "analysis_results": {
        "trace": None, # Loaded traceback content
        "source_files_content": {}, # Dictionary of {file_path: content} for source files
        "patch": None, # Generated or edited patch content
        "explanation": None, # Explanation of the patch
        "doc_summary": None, # Documentation summary (can come from analysis or separate doc gen)
        "patched_file_name": None, # Name of the file the patch applies to
        "original_patched_file_content": None # Original content of the file being patched
    },
    "qa_result": None, # Result of QA validation
    "inbox_data": None, # Data from the issues inbox
    "workflow_status_data": None, # Data from workflow status check
    "metrics_data": None, # Data from system metrics
    "qa_code_to_validate": None # Code specifically loaded for QA validation (maybe redundant with analysis_results['patch'])
}
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# === Streamlit Page Configuration ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# === GitHub Repo Integration Sidebar ===
with st.sidebar:
    st.markdown("### 📦 Load Code from GitHub")

    # Text input for GitHub URL
    # The value is automatically managed by Streamlit in st.session_state.github_repo_url_input
    github_url_input_widget_value = st.text_input(
    "Public GitHub Repository URL",
    placeholder="https://github.com/owner/repo",
    key="github_repo_url_input",
)

if st.button(f"Load/Process Repo", key=f"load_repo_button_{github_url_input_widget_value}", use_container_width=True):
    input_url = github_url_input_widget_value.strip()
    logger.info(f"Load/Process Repo button clicked. Input URL: '{input_url}'")

    if not input_url:
        # Clear GitHub state and reset analysis results if the input is empty
        logger.info("Input URL is empty. Clearing GitHub state and analysis results.")
        clear_github_selection_state()
        clear_analysis_inputs()
        clear_analysis_outputs()
        st.info("GitHub input cleared and analysis reset.")
        st.session_state.current_github_repo_url = None
    elif not input_url.startswith("PROCESSING_") and input_url != st.session_state.current_github_repo_url:
        # Set processing marker for new valid input and rerun
        logger.info(f"New valid URL detected: '{input_url}'. Preparing to process.")
        clear_analysis_inputs()
        clear_analysis_outputs()
        st.session_state.current_github_repo_url = "PROCESSING_" + input_url
        st.rerun()
    else:
        logger.warning(f"Button clicked but no valid state change detected. Current URL: {st.session_state.current_github_repo_url}")

# Check if the URL needs processing
current_loaded_url = st.session_state.get("current_github_repo_url")
if current_loaded_url and current_loaded_url.startswith("PROCESSING_"):
    # Extract the actual URL from the processing marker
    active_github_url = current_loaded_url.replace("PROCESSING_", "").strip()
    logger.info(f"Processing GitHub URL: {active_github_url}")

    # Validate and parse the URL
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", active_github_url)
    if not match:
        st.warning("Invalid GitHub URL format. Use: https://github.com/owner/repo")
        clear_github_selection_state()
        st.session_state.current_github_repo_url = None
        logger.warning("Reset current_github_repo_url to None due to invalid URL.")
    else:
        # Process the valid URL (e.g., fetch branches)
        owner, repo = match.groups()
        logger.info(f"Valid URL parsed: owner={owner}, repo={repo}")

        # Fetch branches or perform necessary processing
        try:
            api_branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
            with st.spinner(f"Loading branches for {owner}/{repo}..."):
                branches_res = requests.get(api_branches_url, timeout=10)
                branches_res.raise_for_status()
                branches_data = branches_res.json()
                st.session_state.github_branches = [b["name"] for b in branches_data]
            logger.info(f"Branch fetch successful. Found {len(st.session_state.github_branches)} branches.")

            # Update the state to mark processing complete
            st.session_state.current_github_repo_url = active_github_url
            st.success(f"Repo '{owner}/{repo}' branches loaded.")
        except Exception as e:
            logger.error(f"Error processing GitHub URL: {e}")
            st.error(f"Failed to process GitHub URL: {e}")
            clear_github_selection_state()
            st.session_state.current_github_repo_url = None    # --- GitHub URL Parsing & Branch Fetch ---
    # This logic runs on every rerun and is triggered when github_repo_url_input changes
    # and the Load/Process Repo button's logic sets current_github_repo_url to trigger it.
    active_github_url = st.session_state.get("github_repo_url_input", "").strip()
    current_loaded_url = st.session_state.get("current_github_repo_url")

    logger.info(f"Branch fetch check start. active_github_url='{active_github_url}', current_loaded_url='{current_loaded_url}'")
    # The condition to trigger branch fetching:
    # 1. There's an active URL in the input
    # 2. We haven't successfully loaded branches for this exact URL yet (current_loaded_url is None or different)
    # 3. Avoid triggering if current_loaded_url is the "PROCESSING_" marker set by the button
    condition_met = active_github_url and (current_loaded_url is None or (isinstance(current_loaded_url, str) and not current_loaded_url.startswith("PROCESSING_") and current_loaded_url != active_github_url))
    logger.info(f"Branch fetch condition met: {condition_met}")

    if condition_met:
         match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", active_github_url)
         if not match:
             logger.warning(f"Invalid GitHub URL format in fetch check: {active_github_url}")
             st.warning("Invalid GitHub URL format. Use: https://github.com/owner/repo")
             clear_github_selection_state() # Clear GitHub states on invalid format
             st.session_state.current_github_repo_url = None # Ensure loaded URL is explicitly None
             logger.info(f"Reset current_github_repo_url to None due to invalid URL format.")
         else:
             owner, repo = match.groups()
             logger.info(f"Valid URL parsed: owner={owner}, repo={repo}")

             # Update owner/repo names
             st.session_state.github_repo_owner = owner
             st.session_state.github_repo_name = repo

             api_branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
             logger.info(f"Attempting branch fetch for: {api_branches_url}")

             try:
                 with st.spinner(f"Loading branches for {owner}/{repo}..."):
                     branches_res = requests.get(api_branches_url, timeout=10)
                     branches_res.raise_for_status()
                     branches_data = branches_res.json()
                     st.session_state.github_branches = [b["name"] for b in branches_data]
                 logger.info(f"Branch fetch successful. Found {len(st.session_state.github_branches)} branches.")

                 if st.session_state.github_branches:
                     # Set branch selection to the first branch if none selected or old selection invalid
                     if st.session_state.github_selected_branch not in st.session_state.github_branches:
                          st.session_state.github_selected_branch = st.session_state.github_branches[0]
                     st.session_state.github_path_stack = [""] # Reset path on successful repo load
                     st.success(f"Repo '{owner}/{repo}' branches loaded.")
                     # --- Crucial Update for Loop Prevention ---
                     # Mark the URL as successfully loaded *only after successful API call and branches found*
                     st.session_state.current_github_repo_url = active_github_url
                     logger.info(f"Branch fetch successful. Set current_github_repo_url to: '{st.session_state.current_github_repo_url}'")
                 else:
                     st.warning("No branches found for this repository.")
                     clear_github_selection_state() # Clear GitHub states if no branches
                     st.session_state.current_github_repo_url = None # Reset on no branches found
                     logger.warning("Branch fetch found no branches. Reset current_github_repo_url to None.")


             except requests.exceptions.RequestException as e:
                 logger.error(f"❌ Branch fetch API error: {e}")
                 st.error(f"❌ Branch fetch error: {e}")
                 clear_github_selection_state() # Clear GitHub states on error
                 st.session_state.current_github_repo_url = None # Reset on API error to allow retry
                 logger.warning("Branch fetch API error. Reset current_github_repo_url to None.")

             except json.JSONDecodeError as e:
                  logger.error(f"❌ Branch JSON decode error: {e}")
                  st.error(f"❌ Branch JSON error: Could not parse response. {e}")
                  clear_github_selection_state() # Clear GitHub states on error
                  st.session_state.current_github_repo_url = None # Reset on JSON error to allow retry
                  logger.warning("Branch fetch JSON error. Reset current_github_repo_url to None.")

    # --- Branch Selection ---
    # Only show branch selector if branches were successfully loaded
    current_branches = st.session_state.get("github_branches", [])
    if current_branches:
        logger.info(f"Showing branch selector. {len(current_branches)} branches available.")
        try:
            # Find the index of the currently selected branch, default to 0 if not found
            branch_idx = current_branches.index(st.session_state.github_selected_branch) if st.session_state.github_selected_branch in current_branches else 0
        except ValueError:
            # Fallback in case of unexpected value, should be covered by above check
            branch_idx = 0
        logger.info(f"Selected branch index: {branch_idx}")

        selected_branch = st.selectbox(
            "Select Branch",
            current_branches,
            index=branch_idx,
            key="github_branch_selector",
            on_change=lambda: logger.info(f"Branch selector changed to {st.session_state.github_selected_branch}. Resetting path stack.") or st.session_state.update({"github_path_stack": [""]}) # Reset path on branch change
        )
        # Streamlit automatically updates st.session_state.github_selected_branch due to the key

    # --- File Browser Logic (if owner, repo, branch are set) ---
    gh_owner = st.session_state.get("github_repo_owner")
    gh_repo = st.session_state.get("github_repo_name")
    gh_branch = st.session_state.get("github_selected_branch")
    current_path_stack = st.session_state.get("github_path_stack", [""])
    current_path_str = "/".join([p for p in current_path_stack if p])

    if gh_owner and gh_repo and gh_branch:
        logger.info(f"File browser active for {gh_owner}/{gh_repo} @ {gh_branch} path /{current_path_str}")

        @st.cache_data(ttl=120, show_spinner="Fetching directory contents...")
        def fetch_gh_dir_content(owner_str, repo_str, path_str, branch_str):
            url = f"https://api.github.com/repos/{owner_str}/{repo_str}/contents/{path_str}?ref={branch_str}"
            logger.info(f"Fetching GitHub dir (cached): {url}")
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                st.warning(f"Dir content fetch error for '{path_str}': {e}")
                logger.error(f"Dir content fetch error for '{path_str}': {e}")
            except json.JSONDecodeError as e:
                st.warning(f"Dir content JSON error for '{path_str}': {e}")
                logger.error(f"Dir content JSON error for '{path_str}': {e}")
            return None

        entries = fetch_gh_dir_content(gh_owner, gh_repo, current_path_str, gh_branch)

        if entries is not None:
            logger.info(f"Fetched {len(entries)} items for path /{current_path_str}")
            # Display current path and "Up" button
            display_path = f"Current Path: /{current_path_str}" if current_path_str else "Current Path: / (Repo Root)"
            st.caption(display_path)
            if len(current_path_stack) > 1: # Not in root
                if st.button("⬆️ .. (Parent Directory)", key=f"gh_up_button_{current_path_str}", use_container_width=True):
                    st.session_state.github_path_stack.pop()
                    logger.info(f"Clicked 'Up' button. New path stack: {st.session_state.github_path_stack}")
                    st.rerun()

            st.markdown("###### Directories")
            # Filter and sort directories
            dirs = sorted([item for item in entries if item["type"] == "dir"], key=lambda x: x["name"])
            for item in dirs:
                 if st.button(f"📁 {item['name']}", key=f"gh_dir_{current_path_str}_{item['name']}", use_container_width=True):
                     st.session_state.github_path_stack.append(item['name'])
                     logger.info(f"Clicked directory '{item['name']}'. New path stack: {st.session_state.github_path_stack}")
                     st.rerun()

            st.markdown("###### Files")
            # Filter and sort recognized files
            files = sorted([item for item in entries if item["type"] == "file" and any(item["name"].endswith(ext) for ext in RECOGNIZED_FILE_EXTENSIONS)], key=lambda x: x["name"])
            for item in files:
                if st.button(f"📄 {item['name']}", key=f"gh_file_{current_path_str}_{item['name']}", use_container_width=True):
                    file_rel_path = f"{current_path_str}/{item['name']}".strip("/")
                    raw_url = f"https://raw.githubusercontent.com/{gh_owner}/{gh_repo}/{gh_branch}/{file_rel_path}"
                    logger.info(f"Clicked file '{item['name']}'. Attempting to fetch: {raw_url}")
                    try:
                        with st.spinner(f"Loading {item['name']}..."):
                            file_r = requests.get(raw_url, timeout=10)
                            file_r.raise_for_status()
                        file_content = file_r.text
                        logger.info(f"Successfully fetched file '{item['name']}'. Size: {len(file_content)} bytes.")

                        # Use relative path from repo root as key for analysis_results
                        key_path = os.path.join(*current_path_stack, item['name']).replace("\\","/") if current_path_str else item['name']

                        if item['name'].endswith(TRACEBACK_EXTENSION):
                            st.session_state.analysis_results['trace'] = file_content
                            st.info(f"'{key_path}' loaded as Traceback.")
                            # When loading a new traceback, clear previous analysis outputs
                            clear_analysis_outputs()
                            logger.info("Cleared analysis outputs after loading new traceback.")
                        else: # Already filtered by RECOGNIZED_FILE_EXTENSIONS
                            st.session_state.analysis_results['source_files_content'][key_path] = file_content
                            st.info(f"'{key_path}' loaded as Source File.")
                            logger.info("Added file content to analysis inputs.")


                        st.rerun() # To update "Loaded Analysis Inputs" expander and potentially other tabs
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error loading file '{item['name']}': {e}")
                        logger.error(f"Error fetching file '{item['name']}': {e}")
            if not dirs and not files:
                 st.caption("No directories or recognized files found in this path.")

        elif current_path_str: # entries is None and not in root, indicating a fetch error
             st.warning("Could not list directory contents.")
             logger.warning(f"Could not fetch directory contents for path /{current_path_str}. Entries is None.")

    st.markdown("---")
    with st.expander("📬 Loaded Analysis Inputs", expanded=True):
        trace_val = st.session_state.analysis_results.get('trace')
        sources_val = st.session_state.analysis_results.get('source_files_content', {})
        if trace_val: st.text_area("Current Traceback:", value=trace_val, height=100, disabled=True, key="sidebar_trace_loaded_final")
        else: st.caption("No traceback loaded.")
        if sources_val:
            st.write("Current Source Files:")
            for name in sources_val.keys(): st.caption(f"- {name}")
        else: st.caption("No source files loaded.")
    st.markdown("---")

# === Manual File Uploader (Main Page Area) ===
st.markdown("### ⬆️ Or, Upload Files Manually")
manual_uploaded_files_main = st.file_uploader(
    "📄 Upload traceback (.txt) and/or source files",
    type=[ext.lstrip('.') for ext in RECOGNIZED_FILE_EXTENSIONS],
    accept_multiple_files=True,
    key="manual_uploader_main_page"
)

if manual_uploaded_files_main:
    logger.info("Manual uploader detected files in this rerun.")
    manual_trace_str = None
    manual_sources_dict = {}
    manual_summary_list_main = []

    # Process uploaded files
    for file_item in manual_uploaded_files_main:
        try:
            # Read file content
            # file_item.getvalue() returns bytes, decode using utf-8
            content_bytes = file_item.getvalue()
            content_str = content_bytes.decode("utf-8", errors='ignore') # Use errors='ignore' for robustness

            if file_item.name.endswith(TRACEBACK_EXTENSION):
                manual_trace_str = content_str
                manual_summary_list_main.append(f"Traceback: {file_item.name}")
            elif any(file_item.name.endswith(ext) for ext in SUPPORTED_SOURCE_EXTENSIONS):
                manual_sources_dict[file_item.name] = content_str # Use file name as key
                manual_summary_list_main.append(f"Source: {file_item.name}")
            # Ignore files with unrecognized extensions
        except Exception as e:
            st.error(f"Error processing '{file_item.name}': {e}")
            logger.error(f"Error processing uploaded file {file_item.name}: {e}")


    # Decide how to update state based on what was uploaded
    # If files were uploaded, clear the GitHub selection state
    logger.info("Clearing GitHub selection state due to manual upload.")
    clear_github_selection_state()

    if manual_trace_str is not None:
        # If a traceback is uploaded, clear all previous analysis data (from GitHub or prior uploads)
        # and set the new traceback and any source files uploaded alongside it.
        logger.info("Traceback found in manual upload. Clearing analysis state.")
        clear_analysis_inputs()
        clear_analysis_outputs()
        st.session_state.analysis_results['trace'] = manual_trace_str
        st.session_state.analysis_results['source_files_content'] = manual_sources_dict # Only sources from this upload
        st.success(f"Manually uploaded: {', '.join(manual_summary_list_main)}. GitHub selection cleared.")
        st.rerun() # Rerun to update UI with new analysis inputs
    elif manual_sources_dict:
         # If only source files are uploaded (and no traceback in this upload),
         # merge them into the existing source files state. Keep existing traceback.
         logger.info("Source files found in manual upload (no traceback). Merging sources.")
         st.session_state.analysis_results['source_files_content'].update(manual_sources_dict)
         st.success(f"Manually added source files: {', '.join(manual_summary_list_main)}. GitHub selection cleared.")
         st.rerun() # Rerun to update UI with new source files
    else:
        logger.warning("Manual uploader files detected, but no recognized trace or sources found.")
    # If no recognized files, no rerun is triggered by this block, which is correct.


# === Main Application Tabs ===
tab_titles_main_list = [
    "📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation",
    "📣 Issue Notices", "🤖 Autonomous Workflow", "🔍 Workflow Check",
    "📊 Repo Structure Insights",
    "📈 Metrics"
]
# Assign tabs to variables
tab_patch, tab_qa, tab_doc, tab_issues, tab_workflow_trigger, \
    tab_workflow_status, tab_repo_insights, tab_metrics = st.tabs(tab_titles_main_list)

# Content for each tab - this code runs on every rerun, but the content is only displayed
# for the active tab. State accessed within these blocks reflects the current session state.

with tab_patch:
    st.header("📄 Traceback & Patch Analysis")
    loaded_trace_tab1 = st.session_state.analysis_results.get('trace')
    loaded_sources_tab1 = st.session_state.analysis_results.get('source_files_content', {})
    patched_code_val_tab1 = st.session_state.analysis_results.get('patch')
    original_content_tab1 = st.session_state.analysis_results.get('original_patched_file_content')
    explanation_val_tab1 = st.session_state.analysis_results.get('explanation')

    st.markdown("### Analysis Inputs")
    if loaded_trace_tab1:
        st.text_area("Loaded Traceback:", value=loaded_trace_tab1, height=150, disabled=True)
    else:
        st.info("Load a traceback from GitHub or manually upload to enable analysis.")

    if loaded_sources_tab1:
        with st.expander("Loaded Source Files:", expanded=False):
             st.json(loaded_sources_tab1)
    else:
         st.info("Load source files from GitHub or manually upload to provide context for analysis.")


    if st.button("🔬 Analyze Loaded Data & Suggest Patch", key="tab1_analyze_main_btn", use_container_width=True):
        logger.info("Analyze button clicked in Patch tab.")
        # This button triggers analysis based on the CURRENT inputs in session state
        # It should populate the analysis OUTPUTS state.
        if not loaded_trace_tab1 and not loaded_sources_tab1:
            st.warning("Please load data (traceback and/or source files) first from the sidebar or uploader.")
        else:
            # Prepare payload from current session state inputs
            payload = {
                "trace": loaded_trace_tab1,
                "language": "python", # Assuming language is always python for this example, adjust if needed
                "source_files": loaded_sources_tab1
                }
            # Call your backend API for patch suggestion
            response = make_api_request("POST", SUGGEST_PATCH_URL, json_payload=payload, operation_name="Patch Suggestion")

            if response and not response.get("error"):
                 # Update only the analysis OUTPUTS in session state based on the API response
                 st.session_state.analysis_results.update({
                     'patch': response.get("patched_code", ""),
                     'explanation': response.get("explanation"),
                     'patched_file_name': response.get("patched_file_name"),
                     'original_patched_file_content': response.get("original_content")
                 })
                 # Initialize the edited patch state with the suggested patch
                 st.session_state.edited_patch = st.session_state.analysis_results['patch']
                 st.success("Patch suggestion received.")
                 logger.info("Successfully received patch suggestion.")
                 st.rerun() # Rerun to update UI with results, diff, and editor
            else:
                 # If analysis fails, clear the previous outputs and show an error
                 clear_analysis_outputs()
                 st.error(f"Patch analysis failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
                 logger.error("Patch analysis failed.")


    st.markdown("---")
    st.markdown("### Analysis Results")

    if patched_code_val_tab1 is not None:
        st.markdown("### 🔍 Diff View")
        # Check if original content is available and different from the patch
        if original_content_tab1 is not None and original_content_tab1 != patched_code_val_tab1:
            html_diff_gen_tab1 = difflib.HtmlDiff(wrapcolumn=70, tabsize=4)
            diff_html = html_diff_gen_tab1.make_table(
                original_content_tab1.splitlines(keepends=True),
                patched_code_val_tab1.splitlines(keepends=True),
                "Original",
                "Patched",
                context=True,
                numlines=3
            )
            components.html(diff_html, height=450, scrolling=True)
        elif original_content_tab1 is not None and original_content_tab1 == patched_code_val_tab1:
             st.info("No changes in the suggested patch compared to the original file content.")
        elif patched_code_val_tab1 is not None:
             st.info("Original file content not available to generate a diff.")


        st.markdown("### ✍️ Edit Patch (Live)")
        # Use st_ace for live editing, its value is automatically stored in st.session_state.edited_patch
        edited_code_val_tab1 = st_ace(
            value=st.session_state.edited_patch, # Use the edited state as the source of truth for the editor
            language="python", # Set language highlighting
            theme="monokai", # Set editor theme
            height=300, # Set editor height
            key="tab1_ace_editor_main" # Key to store the value in session state
        )

        # If the user changed the code in the editor, update the 'patch' in analysis_results
        # This comparison happens on every rerun
        if edited_code_val_tab1 != st.session_state.analysis_results.get('patch'):
             logger.info("st_ace editor value changed. Updating patch in session state.")
             st.session_state.analysis_results['patch'] = edited_code_val_tab1
             st.session_state.edited_patch = edited_code_val_tab1 # Keep edited_patch in sync
             # No rerun needed here, the editor change itself triggers a rerun


        if explanation_val_tab1:
             st.markdown(f"**Explanation:** {explanation_val_tab1}")

    elif loaded_trace_tab1 or loaded_sources_tab1:
         st.info("Run analysis above to generate a patch.")
    else:
         st.info("Load data and run analysis to see results here.")


with tab_qa:
    st.header("✅ QA Validation")
    # Get the code for QA, preferably the edited patch if available, otherwise the suggested patch
    code_for_qa_tab2 = st.session_state.get('edited_patch') or st.session_state.analysis_results.get('patch')
    patched_file_name_tab2 = st.session_state.analysis_results.get('patched_file_name')

    if code_for_qa_tab2 is not None:
        st.text_area("Code for QA:", value=code_for_qa_tab2, height=200, disabled=True, key="qa_code_display_tab2_main_val")

        if st.button("🛡️ Run QA Validation", key="qa_run_validation_btn_tab2_val", use_container_width=True):
            logger.info("Run QA Validation button clicked.")
            if not code_for_qa_tab2:
                 st.warning("No code available to validate.")
            else:
                 payload = {"code": code_for_qa_tab2, "patched_file_name": patched_file_name_tab2}
                 # Call your backend API for QA validation
                 response = make_api_request("POST", QA_URL, json_payload=payload, operation_name="QA Validation")
                 if response and not response.get("error"):
                     st.session_state.qa_result = response # Store QA result in session state
                     st.success("QA Validation complete.")
                     logger.info("Successfully received QA validation result.")
                     st.rerun() # Rerun to display result
                 else:
                     st.session_state.qa_result = {"error": True, "details": response.get('details', '') if response else 'No response'} # Store error
                     st.error(f"QA Validation failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
                     logger.error("QA Validation failed.")


        # Display the QA result if available in session state
        qa_result_val_tab2 = st.session_state.get("qa_result")
        if qa_result_val_tab2:
            st.markdown("### QA Result:")
            if qa_result_val_tab2.get("error"):
                 st.error(f"QA Error: {qa_result_val_tab2.get('details', 'Unknown error.')}")
            else:
                 st.json(qa_result_val_tab2) # Display the full JSON result
        elif st.session_state.get("qa_result") is not None: # Handle case where result was explicitly set to None
             st.info("No QA result available yet. Run validation.")


    else:
        st.warning("No patched code from the 'Traceback + Patch' tab to validate.")


with tab_doc:
    st.header("📘 Documentation")
    # Display documentation summary from analysis results if available
    doc_summary_main_val = st.session_state.analysis_results.get('explanation') # Often explanation includes documentation aspects
    if doc_summary_main_val:
        st.markdown(f"**Documentation Summary from Analysis:** {doc_summary_main_val}")
        st.markdown("---")

    st.subheader("Generate Documentation for Specific Code:")
    # Text area for ad-hoc documentation generation
    doc_code_input_tab3_main_val = st.text_area("Paste code here:", key="doc_code_input_area_tab3_main_val", height=200)

    if st.button("📝 Generate Ad-hoc Documentation", key="doc_generate_btn_tab3_main_val", use_container_width=True):
        logger.info("Generate Ad-hoc Documentation button clicked.")
        if not doc_code_input_tab3_main_val:
            st.warning("Please paste code into the text area to generate documentation.")
        else:
            payload = {"code": doc_code_input_tab3_main_val}
            # Call your backend API for documentation generation
            response = make_api_request("POST", DOC_URL, json_payload=payload, operation_name="Ad-hoc Docs")

            if response and not response.get("error"):
                 # Display the generated documentation
                 generated_doc = response.get("doc", "No documentation generated.")
                 st.markdown("### Generated Documentation:")
                 st.markdown(generated_doc)
                 logger.info("Successfully received ad-hoc documentation.")
            else:
                 st.error(f"Documentation generation failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
                 logger.error("Ad-hoc documentation generation failed.")


with tab_issues:
    st.header("📣 Issue Notices & Agent Summaries")
    if st.button("🔄 Refresh Issue Notices", key="fetch_issues_tab_btn_val", use_container_width=True):
        logger.info("Refresh Issue Notices button clicked.")
        # Call your backend API to fetch issues inbox data
        response = make_api_request("GET", ISSUES_INBOX_URL, operation_name="Fetch Issues")
        if response and not response.get("error"):
            st.session_state.inbox_data = response # Store inbox data in session state
            st.success("Issue notices refreshed.")
            logger.info("Successfully fetched issue notices.")
            st.rerun() # Rerun to display new data
        else:
            st.session_state.inbox_data = {"error": True, "details": response.get('details', '') if response else 'No response'} # Store error
            st.error(f"Issue fetch failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
            logger.error("Failed to fetch issue notices.")

    # Display current inbox data from session state
    current_inbox = st.session_state.get("inbox_data")
    if current_inbox and not current_inbox.get("error"):
        issues_list = current_inbox.get("issues", [])
        if issues_list:
            st.markdown("### Current Issues:")
            for i, issue_item in enumerate(issues_list):
                with st.expander(f"Issue {issue_item.get('id','N/A')} - {issue_item.get('classification','N/A')}"):
                    st.json(issue_item)
                    # Button to trigger workflow for a specific issue
                    if st.button(f"▶️ Trigger Workflow for Issue {issue_item.get('id','N/A')}", key=f"trigger_issue_{issue_item.get('id',i)}"):
                        logger.info(f"Trigger Workflow button clicked for issue {issue_item.get('id','N/A')}.")
                        # Call your backend API to trigger workflow
                        payload = {"issue_id": issue_item.get('id')}
                        workflow_response = make_api_request("POST", WORKFLOW_RUN_URL, json_payload=payload, operation_name=f"Manual Workflow Trigger {issue_item.get('id','N/A')}")
                        if workflow_response and not workflow_response.get("error"):
                            st.success(f"Workflow for Issue {issue_item.get('id','N/A')} triggered successfully.")
                            logger.info(f"Workflow for issue {issue_item.get('id','N/A')} triggered successfully.")
                            # Optionally clear workflow status to force refresh on next tab check
                            st.session_state.workflow_status_data = None
                            st.rerun() # Rerun to update status tab potentially
                        else:
                            st.error(f"Failed to trigger workflow for Issue {issue_item.get('id','N/A')}. {workflow_response.get('details', '') if workflow_response else 'No response'}")
                            logger.error(f"Failed to trigger workflow for issue {issue_item.get('id','N/A')}.")
        else:
            st.info("No issues found in the inbox.")
    elif current_inbox and current_inbox.get("error"):
         st.error(f"Failed to load issues: {current_inbox.get('details', 'Unknown error.')}")
    else:
        st.info("Click 'Refresh Issue Notices' to fetch data.")


with tab_workflow_trigger:
    st.header("🤖 Trigger Autonomous Workflow by ID")
    workflow_issue_id_input_val = st.text_input("Enter Issue ID to trigger workflow:", key="workflow_issue_id_trigger_input_val")
    if st.button("▶️ Run Workflow by ID", key="run_workflow_by_id_btn_val", use_container_width=True):
        logger.info(f"Run Workflow by ID button clicked for ID: {workflow_issue_id_input_val}")
        if not workflow_issue_id_input_val:
            st.warning("Please enter an Issue ID to trigger a workflow.")
        else:
            payload = {"issue_id": workflow_issue_id_input_val}
            # Call your backend API to trigger workflow by ID
            response = make_api_request("POST", WORKFLOW_RUN_URL, json_payload=payload, operation_name=f"Manual Workflow Trigger {workflow_issue_id_input_val}")
            if response and not response.get("error"):
                 st.success(f"Workflow for Issue {workflow_issue_id_input_val} triggered successfully.")
                 logger.info(f"Workflow for issue {workflow_issue_id_input_val} triggered successfully.")
                 # Optionally clear workflow status to force refresh on next tab check
                 st.session_state.workflow_status_data = None
                 st.rerun() # Rerun to update status tab potentially
            else:
                 st.error(f"Failed to trigger workflow for Issue {workflow_issue_id_input_val}. {response.get('details', '') if response else 'No response'}")
                 logger.error(f"Failed to trigger workflow for issue {workflow_issue_id_input_val}.")

with tab_workflow_status:
    st.header("🔍 Workflow Status Check")
    if st.button("🔄 Refresh Workflow Status", key="refresh_workflow_status_check_btn_val", use_container_width=True):
        logger.info("Refresh Workflow Status button clicked.")
        # Call your backend API to fetch workflow status
        response = make_api_request("GET", WORKFLOW_CHECK_URL, operation_name="Workflow Status")
        if response and not response.get("error"):
            st.session_state.workflow_status_data = response # Store status data
            st.success("Workflow status refreshed.")
            logger.info("Successfully fetched workflow status.")
            st.rerun() # Rerun to display new data
        else:
            st.session_state.workflow_status_data = {"error": True, "details": response.get('details', '') if response else 'No response'} # Store error
            st.error(f"Workflow status check failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
            logger.error("Failed to fetch workflow status.")

    # Display current workflow status data from session state
    current_status = st.session_state.get("workflow_status_data")
    if current_status and not current_status.get("error"):
        st.markdown("### Current Workflow Status:")
        st.json(current_status) # Display the full JSON status
    elif current_status and current_status.get("error"):
         st.error(f"Failed to load workflow status: {current_status.get('details', 'Unknown error.')}")
    else:
        st.info("Click 'Refresh Workflow Status' to see data.")

with tab_repo_insights:
    st.header("📊 Repository Structure Insights")
    st.markdown("""
    This tab is intended to display a "digital chart" or visual representation of your loaded GitHub repository's content.

    if gh_owner_insight and gh_repo_insight:
        st.info(f"Insights for repository: {gh_owner_insight}/{gh_repo_insight} would appear here once specified.")
        # @st.cache_data # Example of fetching language data (uncomment and implement)
        # def get_repo_languages(owner, repo):
        #      lang_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        #      try:
        #          r = requests.get(lang_url, timeout=5)
        #          r.raise_for_status()
        #          return r.json() # Returns dict like {"Python": 30203, "JavaScript": 1024}
        #      except Exception as e:
        #          logger.error(f"Failed to fetch language data for {owner}/{repo}: {e}")
        #          return None

        # languages = get_repo_languages(gh_owner_insight, gh_repo_insight)
        # if languages:
        #      st.write("Language Breakdown (Example - actual chart TBD):")
        #      st.json(languages) # Replace with actual chart later using Plotly, Altair, etc.
        # else:
        #      st.warning("Could not fetch language data for the repository to display an example chart.")
    else:
        st.warning("Load a GitHub repository from the sidebar to see potential insights.")


with tab_metrics:
    st.header("📈 System Metrics")
    if st.button("📈 Fetch System Metrics", key="fetch_metrics_tab_btn_val", use_container_width=True):
        logger.info("Fetch System Metrics button clicked.")
        # Call your backend API to fetch system metrics
        response = make_api_request("GET", METRICS_URL, operation_name="Fetch Metrics")
        if response and not response.get("error"):
            st.session_state.metrics_data = response # Store metrics data
            st.success("System metrics refreshed.")
            logger.info("Successfully fetched system metrics.")
            st.rerun() # Rerun to display new data
        else:
            st.session_state.metrics_data = {"error": True, "details": response.get('details', '') if response else 'No response'} # Store error
            st.error(f"Metrics fetch failed. {response.get('details', '') if response else 'No response or unexpected format from backend.'}")
            logger.error("Failed to fetch system metrics.")

    # Display current metrics data from session state
    current_metrics = st.session_state.get("metrics_data")
    if current_metrics and not current_metrics.get("error"):
        st.markdown("### Current System Metrics:")
        st.json(current_metrics) # Display the full JSON metrics
    elif current_metrics and current_metrics.get("error"):
         st.error(f"Failed to load metrics: {current_metrics.get('details', 'Unknown error.')}")
    else:
        st.info("Click 'Fetch System Metrics' to see data.")


# === Voice Agent Section ===
# (Kept as per your last provided file - simpler version without full Gemini bi-directional audio yet)
st.markdown("---")
st.markdown("## 🎙️ DebugIQ Voice Agent")
st.caption("Speak your commands to the DebugIQ Agent.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Assuming audio_base64 is present for assistant messages when using voice output
        if "audio_base64" in message and message["role"] == "assistant" and st.session_state.get("using_gemini_voice", False):
             try:
                 # Decode the base64 audio data
                 audio_bytes = base64.b64decode(message["audio_base64"])
                 # Provide the audio data to Streamlit's audio player
                 st.audio(audio_bytes, format="audio/mp3") # Assuming backend provides mp3
             except Exception as e:
                 logger.error(f"Error decoding or playing audio for chat message: {e}")


# WebRTC streamer for capturing audio
# Added ClientSettings import and usage as it was in your original snippet
# You need to import ClientSettings from streamlit_webrtc
from streamlit_webrtc import ClientSettings # Ensure this import is present

try:
    ctx = webrtc_streamer(
        key=f"voice_agent_stream_{BACKEND_URL}", # Use backend URL in key to differentiate if needed
        mode=WebRtcMode.SENDONLY, # We only need to send audio from the browser
        client_settings=ClientSettings(
             rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Example STUN server
             media_stream_constraints={"audio": True, "video": False} # Only enable audio
        ),
        # async_processing=True # Consider if audio processing is time-consuming
    )
except Exception as e:
    st.error(f"Failed to initialize voice agent: {e}")
    logger.exception("Error initializing webrtc_streamer for Voice Agent")
    ctx = None # Set ctx to None if initialization fails

# Audio processing logic when the streamer is connected and receiving audio
if ctx and ctx.audio_receiver:
    voice_status_indicator_main_agent = st.empty() # Placeholder for status messages

    # Note: The audio capturing and processing loop using `ctx.audio_receiver.get_frame()`
    # and sending to your backend for transcription and command processing is complex
    # and requires careful handling of threads, buffering, and API calls.
    # The original code snippet for this part was commented out or incomplete.
    # Below is a simplified representation based on the structure; you will need
    # to implement the actual audio capture, buffering, and sending logic here.

    # This part of the logic needs to run in a way that doesn't block the Streamlit rerun.
    # Using a separate thread for audio processing and API calls is common in Streamlit.

    voice_status_indicator_main_agent.info("Voice agent initialized. Speak now.")

    # --- Placeholder for Audio Processing and API Call Logic ---
    # You need to implement the loop that reads audio frames from ctx.audio_receiver,
    # buffers them, detects speech (optional but recommended), sends to VOICE_TRANSCRIBE_URL,
    # gets the transcription, sends commands to COMMAND_URL or GEMINI_CHAT_URL,
    # and updates st.session_state.chat_history.
    # This often involves a dedicated thread started when ctx.state.playing becomes True.

    # Example (conceptual) - Requires a separate thread implementation:
    # if ctx.state.playing:
    #      if "audio_processing_thread" not in st.session_state:
    #          st.session_state.audio_processing_thread = start_audio_processing_thread(
    #              ctx.audio_receiver,
    #              VOICE_TRANSCRIBE_URL,
    #              COMMAND_URL, # Or GEMINI_CHAT_URL
    #              st.session_state
    #      )
    #      # Thread would handle reading frames, buffering, sending to API, updating chat_history


    # To display real-time transcription feedback or status, you might need
    # mechanisms for the processing thread to communicate back to the main Streamlit thread.
    # This is an advanced Streamlit pattern.

    # --- End Placeholder ---

elif ctx and ctx.state.state == "READY":
     voice_status_indicator_main_agent = st.empty()
     voice_status_indicator_main_agent.info("Voice agent ready. Click Start.")
elif ctx:
     voice_status_indicator_main_agent = st.empty()
     voice_status_indicator_main_agent.info(f"Voice agent state: {ctx.state.state}")
else:
     # Error message already shown in the try...except block
     pass

logger.info("--- Script Rerun End ---")
