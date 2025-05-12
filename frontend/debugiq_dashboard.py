

import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, ClientSettings, WebRtcMode
import numpy as np
import av
from difflib import HtmlDiff
import streamlit.components.v1 as components
import wave
import json
import logging # Import the logging module

# --- Basic Logging Configuration ---
# In a real production app, you might configure this more extensively
# (e.g., based on environment variables, logging to a file or service)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_SOURCE_EXTENSIONS = (".py", ".js", ".java", ".c", ".cpp", ".cs", ".go", ".rb", ".php", ".html", ".css", ".md") # Added .md
TRACEBACK_EXTENSION = ".txt"
DEFAULT_VOICE_SAMPLE_RATE = 16000
DEFAULT_VOICE_SAMPLE_WIDTH = 2 # 16-bit audio
DEFAULT_VOICE_CHANNELS = 1 # Mono
AUDIO_PROCESSING_THRESHOLD_SECONDS = 1 # Process audio every 1 second

autonomous_tab_imported = False
show_autonomous_tab_import_error = None
show_autonomous_workflow_tab = None # Explicitly define

   
except ImportError as e:
    autonomous_tab_imported = False
    show_autonomous_tab_import_error = str(e)
    show_autonomous_workflow_tab = None # Ensure it's None if import fails
    logger.error(f"Failed to import AutonomousWorkflowTab: {e}")
except Exception as e: # Catch any other potential errors during import
    autonomous_tab_imported = False
    show_autonomous_tab_import_error = f"An unexpected error occurred during import: {e}"
    show_autonomous_workflow_tab = None
    logger.error(f"Unexpected error importing AutonomousWorkflowTab: {e}")

# --- Streamlit Page Configuration ---
# set_page_config() MUST be the very first Streamlit command after imports
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")

st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# --- Display import error *after* set_page_config ---
if not autonomous_tab_imported and show_autonomous_tab_import_error:
    st.error(
        f"Could not import the Autonomous Workflow Orchestration tab: {show_autonomous_tab_import_error}. "
        f"Make sure AutonomousWorkflowTab.py is at the correct path (DebuIQ-frontend/.screens/) "
        f"and __init__.py files are in the 'frontend' and '.screens' directories."
    )

# --- Backend URL Configuration ---
# It's crucial to set BACKEND_URL in your production environment.
DEFAULT_BACKEND_URL = "https://debugiq-backend.railway.app" # Keep your default
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)
if BACKEND_URL == DEFAULT_BACKEND_URL:
    st.sidebar.caption(f"⚠️ Using default backend URL: {DEFAULT_BACKEND_URL}. Set BACKEND_URL env var for production.")
else:
    st.sidebar.caption(f"Using backend URL: {BACKEND_URL}")


@st.cache_data(show_spinner="Fetching backend configuration...")
def fetch_config(backend_url):
    """Fetches backend configuration. Errors are handled by the caller."""
    try:
        config_url = f"{backend_url}/api/config"
        logger.info(f"Fetching config from {config_url}")
        r = requests.get(config_url, timeout=10) # Added timeout
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching config from {backend_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding config JSON from {backend_url}: {e}")
        return None

# --- Fetch and Display Config ---
config = fetch_config(BACKEND_URL)

if config:
    st.sidebar.info("Backend Config Loaded.")
    # Consider showing only essential config items or a summary in production for brevity
    # st.sidebar.json(config) # Keep for debugging if needed, or make it optional
    st.sidebar.caption(f"Voice Provider: {config.get('voice_provider', 'N/A')}")
    st.sidebar.caption(f"Model: {config.get('model', 'N/A')}")
else:
    st.sidebar.warning("Backend config not loaded. Using default endpoint URLs. Check logs for errors.")

# --- Define API Endpoints ---
# Use fetched URLs, falling back to defaults constructed with BACKEND_URL
ANALYZE_URL = config.get("analyze_url", f"{BACKEND_URL}/debugiq/analyze") if config else f"{BACKEND_URL}/debugiq/analyze"
QA_URL = config.get("qa_url", f"{BACKEND_URL}/qa/") if config else f"{BACKEND_URL}/qa/"
TRANSCRIBE_URL = config.get("voice_transcribe_url", f"{BACKEND_URL}/voice/transcribe") if config else f"{BACKEND_URL}/voice/transcribe"
COMMAND_URL = config.get("voice_command_url", f"{BACKEND_URL}/voice/command") if config else f"{BACKEND_URL}/voice/command"
INBOX_URL = config.get("inbox_url", f"{BACKEND_URL}/issues/inbox") if config else f"{BACKEND_URL}/issues/inbox" # Added for consistency
WORKFLOW_RUN_URL = config.get("workflow_run_url", f"{BACKEND_URL}/workflow/run") if config else f"{BACKEND_URL}/workflow/run"
WORKFLOW_STATUS_URL = config.get("workflow_status_url", f"{BACKEND_URL}/workflow/status") if config else f"{BACKEND_URL}/workflow/status"

def initialize_session_state():
    defaults = {
        'analysis_results': {
            'trace': None,
            'patch': None,
            'explanation': None,
            'doc_summary': None,
            'patched_file_name': None,
            'original_patched_file_content': None,
            'source_files_content': {}
        },
        'qa_result': None,
        'github_repo_url_input': "", # For the text input widget
        'current_github_repo_url': None, # For tracking successfully loaded repo
        'github_branches': [],
        'github_selected_branch': None,
        'github_path_stack': [""] ,# Start at root
        'inbox_data': None,
        'workflow_status': None,
        'audio_buffer': b"",
        'audio_frame_count': 0,
        'audio_sample_rate': DEFAULT_VOICE_SAMPLE_RATE,
        'audio_sample_width': DEFAULT_VOICE_SAMPLE_WIDTH,
        'audio_num_channels': DEFAULT_VOICE_CHANNELS,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()


# === GitHub Repo Integration Sidebar ===
st.sidebar.markdown("### 📦 Load From GitHub Repo")
repo_url_input = st.sidebar.text_input(
    "Public GitHub URL",
    placeholder="https://github.com/user/repo",
    key="github_repo_url_input_widget" # Ensure keys are unique and descriptive
)

def clear_github_session_state():
    """Resets GitHub related session state."""
    st.session_state.current_github_repo_url = None
    st.session_state.github_branches = []
    st.session_state.github_selected_branch = None
    st.session_state.github_path_stack = [""] # Reset to root

if repo_url_input:
    try:
        import re
        match = re.match(r"https://github.com/([^/]+)/([^/]+)", repo_url_input.strip())
        if match:
            owner, repo = match.groups()

            # Fetch branches if repo URL changed or branches not loaded
            if st.session_state.current_github_repo_url != repo_url_input or not st.session_state.github_branches:
                st.session_state.current_github_repo_url = repo_url_input # Store attempted URL
                api_branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
                logger.info(f"Fetching branches from {api_branches_url}")
                try:
                    branches_res = requests.get(api_branches_url, timeout=10)
                    branches_res.raise_for_status()
                    st.session_state.github_branches = [b["name"] for b in branches_res.json()]
                    if st.session_state.github_branches:
                        st.session_state.github_selected_branch = st.session_state.github_branches[0]
                        st.session_state.github_path_stack = [""] # Reset path on new repo/branch list
                        st.sidebar.success(f"Repo '{owner}/{repo}' branches loaded.")
                    else:
                        st.sidebar.warning("No branches found for this repository.")
                        clear_github_session_state() # Clear relevant state
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"❌ Failed to fetch branches ({e}). Check URL or token if private.")
                    clear_github_session_state() # Clear relevant state
                except json.JSONDecodeError as e:
                    st.sidebar.error(f"❌ Error decoding branches data: {e}.")
                    clear_github_session_state()

            branches = st.session_state.get("github_branches", [])
            selected_branch = st.sidebar.selectbox(
                "Branch",
                branches,
                index=branches.index(st.session_state.github_selected_branch) if st.session_state.github_selected_branch and st.session_state.github_selected_branch in branches else 0,
                key="github_branch_select"
            )
            st.session_state.github_selected_branch = selected_branch # Update selected branch in state

            if selected_branch:
                path_stack = st.session_state.github_path_stack
                current_path = "/".join([p for p in path_stack if p]) # current_path should not start with / for GitHub API

                @st.cache_data(ttl=300, show_spinner=f"Fetching content for {current_path or 'root'}...") # Cache for 5 mins
                def fetch_github_directory_content(api_owner, api_repo, path, branch):
                    content_url = f"https://api.github.com/repos/{api_owner}/{api_repo}/contents/{path}?ref={branch}"
                    logger.info(f"Fetching GitHub content from: {content_url}")
                    try:
                        content_res = requests.get(content_url, timeout=10)
                        content_res.raise_for_status()
                        return content_res.json()
                    except requests.exceptions.RequestException as e:
                        st.sidebar.warning(f"Cannot fetch content for '{path}' ({e}).")
                        return None
                    except json.JSONDecodeError as e:
                        st.sidebar.warning(f"Error decoding content JSON for '{path}': {e}.")
                        return None

                entries = fetch_github_directory_content(owner, repo, current_path, selected_branch)

                if entries is not None:
                    dirs = sorted([e["name"] for e in entries if e["type"] == "dir"])
                    files = sorted([e["name"] for e in entries if e["type"] == "file"])

                    st.sidebar.markdown("##### 📁 Navigate")
                    if current_path: # Only show ".." if not in the root
                        if st.sidebar.button("..", key="github_up_dir", use_container_width=True):
                            st.session_state.github_path_stack.pop()
                            st.rerun()

                    for d in dirs:
                        if st.sidebar.button(f"📁 {d}", key=f"github_dir_{d.replace('.', '_')}", use_container_width=True): # Make key safer
                            st.session_state.github_path_stack.append(d)
                            st.rerun()

                    st.sidebar.markdown("##### 📄 Files")
                    for f_name in files:
                        if st.sidebar.button(f"📄 {f_name}", key=f"github_file_{f_name.replace('.', '_')}", use_container_width=True): # Make key safer
                            file_path_for_url = f"{current_path}/{f_name}".strip("/")
                            file_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{selected_branch}/{file_path_for_url}"
                            logger.info(f"Fetching file content from: {file_url}")
                            try:
                                file_content_res = requests.get(file_url, timeout=10)
                                file_content_res.raise_for_status()
                                file_content = file_content_res.text
                                st.sidebar.success(f"Loaded: {f_name}")

                                full_file_path_in_repo = os.path.join(current_path, f_name).replace("\\", "/") if current_path else f_name

                                if f_name.endswith(TRACEBACK_EXTENSION):
                                    st.session_state.analysis_results['trace'] = file_content
                                    # Optionally clear source files or the specific one if it was loaded as source
                                    st.session_state.analysis_results['source_files_content'].pop(full_file_path_in_repo, None)
                                elif f_name.endswith(SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)): # Allow .txt also as source
                                    st.session_state.analysis_results['source_files_content'][full_file_path_in_repo] = file_content
                                else:
                                    st.sidebar.warning(f"Ignoring '{f_name}': unsupported for analysis. Still loaded if needed by backend.")
                                    # Store it anyway if needed, or handle based on strictness
                                    st.session_state.analysis_results['source_files_content'][full_file_path_in_repo] = file_content

                            except requests.exceptions.RequestException as e:
                                st.sidebar.error(f"Failed to load file {f_name}: {e}")
                else:
                    st.sidebar.warning("Could not list files in this directory.")
            elif branches: # Branches exist but none selected (should not happen with current logic if branches exist)
                st.sidebar.info("Select a branch to browse files.")
        else:
            if repo_url_input: # Only show warning if input is not empty
                 st.sidebar.warning("Please enter a valid GitHub repo URL (e.g., https://github.com/user/repo).")
            clear_github_session_state() # Clear state if URL becomes invalid
    except Exception as e:
        st.sidebar.error(f"⚠️ An unexpected error occurred with GitHub integration: {e}")
        logger.exception("Unexpected GitHub integration error") # Log full traceback
        clear_github_session_state()


# --- File Uploader ---
st.markdown("---")
st.markdown("### ⬆️ Upload Files Manually")
uploaded_files = st.file_uploader(
    "📄 Upload traceback (.txt) + source files (.py, .js, etc.)",
    type=[ext.lstrip('.') for ext in SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)], # Use constants
    accept_multiple_files=True,
    key="manual_file_uploader"
)

if uploaded_files:
    trace_content_upload = None
    source_files_content_upload = {}
    files_loaded_summary = []

    for file in uploaded_files:
        try:
            content = file.getvalue().decode("utf-8")
            if file.name.endswith(TRACEBACK_EXTENSION):
                trace_content_upload = content
                files_loaded_summary.append(f"Traceback: {file.name}")
            elif file.name.endswith(SUPPORTED_SOURCE_EXTENSIONS + (TRACEBACK_EXTENSION,)): # Allow .txt as source too
                source_files_content_upload[file.name] = content # Use original filename as key
                files_loaded_summary.append(f"Source: {file.name}")
            else:
                 st.warning(f"Skipped '{file.name}': unsupported file type for manual upload here.")
        except UnicodeDecodeError:
            st.error(f"Could not decode '{file.name}'. Please ensure it's a UTF-8 encoded text file.")
        except Exception as e:
            st.error(f"Error processing uploaded file '{file.name}': {e}")
            logger.exception(f"Error processing uploaded file {file.name}")


    # Update session state if files were successfully processed
    if trace_content_upload is not None or source_files_content_upload:
        if trace_content_upload is not None:
            st.session_state.analysis_results['trace'] = trace_content_upload
            # Clear previous source files when a new trace is uploaded, this is a design choice
            # st.session_state.analysis_results['source_files_content'] = {}

        if source_files_content_upload:
            # Merge uploaded source files with existing ones, or replace
            # Current: Update/Merge. To replace: st.session_state.analysis_results['source_files_content'] = source_files_content_upload
            st.session_state.analysis_results['source_files_content'].update(source_files_content_upload)

        # Clear GitHub state as manual upload takes precedence
        st.session_state.github_repo_url_input = "" # Clear the text input
        clear_github_session_state()

        st.success(f"✅ Files uploaded and loaded: {', '.join(files_loaded_summary)}.")
        st.rerun() # Rerun to reflect changes immediately in UI, e.g. in trace display

# --- Display current Trace and Source Files (for visibility) ---
with st.sidebar.expander("📬 Loaded Analysis Inputs", expanded=False):
    if st.session_state.analysis_results.get('trace'):
        st.text_area("Current Traceback:", value=st.session_state.analysis_results['trace'], height=100, disabled=True, key="sidebar_trace_display")
    else:
        st.caption("No traceback loaded.")
    if st.session_state.analysis_results.get('source_files_content'):
        st.write("Current Source Files:")
        for name, _ in st.session_state.analysis_results['source_files_content'].items():
            st.caption(f"- {name}")
    else:
        st.caption("No source files loaded.")


# --- Define Tabs ---
tab_titles = [
    "🔧 Patch",
    "✅ QA",
    "📘 Docs",
    "📥 Issue Inbox",
    "🤖 Workflow Orchestration",
    "🔁 Workflow Status"
]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

# --- Helper function for API calls ---
def make_api_request(method, url, json_payload=None, expected_status=200, operation_name="API"):
    try:
        logger.info(f"Making {method} request to {url} for {operation_name} with payload: {json_payload if json_payload else 'No payload'}")
        response = requests.request(method, url, json=json_payload, timeout=30) # General timeout
        response.raise_for_status() # Raises HTTPError for 4xx/5xx
        if response.status_code == expected_status:
            return response.json()
        else: # Should be caught by raise_for_status, but as a fallback
            logger.error(f"{operation_name} failed with status {response.status_code}: {response.text}")
            st.error(f"{operation_name} request failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during {operation_name} to {url}: {e}. Response: {e.response.text if e.response else 'No response text'}")
        st.error(f"{operation_name} failed: {e}. Details: {e.response.text if e.response else 'Server did not provide details.'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException during {operation_name} to {url}: {e}")
        st.error(f"Error communicating with backend for {operation_name}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError during {operation_name} from {url}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        st.error(f"Could not parse {operation_name} response from server: {e}")
        return None


with tab1: # Patch Tab
    st.subheader("Traceback Analysis + Patch")
    if st.button("🧠 Run DebugIQ Analysis", key="run_analysis_button", type="primary"):
        trace_content = st.session_state.analysis_results.get('trace')
        source_files = st.session_state.analysis_results.get('source_files_content', {})

        if not trace_content and not source_files:
            st.warning("Please upload a traceback or source files first.")
        else:
            with st.spinner("🤖 Analyzing with DebugIQ Engine..."):
                payload = {
                    "trace": trace_content,
                    "language": "python", # TODO: Make this configurable if other languages are supported
                    "config": {}, # Pass relevant runtime config if any
                    "source_files": source_files
                }
                result = make_api_request("POST", ANALYZE_URL, json_payload=payload, operation_name="DebugIQ Analysis")

                if result:
                    st.session_state.analysis_results.update({
                        'patch': result.get("patch"),
                        'explanation': result.get("explanation"),
                        'doc_summary': result.get("doc_summary"),
                        'patched_file_name': result.get("patched_file_name"),
                        'original_patched_file_content': result.get("original_patched_file_content")
                    })
                    st.success("✅ Analysis complete. Patch generated.")
                    logger.info("Analysis successful, results updated in session state.")
                else:
                    # Clear previous successful results if analysis fails
                    st.session_state.analysis_results.update({
                        'patch': None, 'explanation': None, 'doc_summary': None,
                        'patched_file_name': None, 'original_patched_file_content': None
                    })
                    st.error("Analysis failed. See error message above or check logs.")

    # Display Patch Diff and Editor
    if st.session_state.analysis_results.get('patch') or st.session_state.analysis_results.get('original_patched_file_content'):
        st.markdown("### 🔍 Patch Diff")
        original_content = st.session_state.analysis_results.get('original_patched_file_content', '')
        patched_content_from_api = st.session_state.analysis_results.get('patch', '') # The one from API

        if original_content and patched_content_from_api and original_content != patched_content_from_api:
            try:
                html_diff_generator = HtmlDiff(wrapcolumn=70) # Optional: wrapcolumn
                html_diff_output = html_diff_generator.make_table(
                    original_content.splitlines(keepends=True),
                    patched_content_from_api.splitlines(keepends=True),
                    "Original", "Patched", context=True, numlines=3
                )
                components.html(html_diff_output, height=400, scrolling=True)
            except Exception as e:
                st.error(f"Could not generate diff view: {e}")
                logger.exception("Error generating HTML diff")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original Content (Fallback)", value=original_content, height=300, disabled=True, key="orig_content_fallback")
                with col2:
                    st.text_area("Patched Content (Fallback)", value=patched_content_from_api, height=300, disabled=True, key="patch_content_fallback")
        elif patched_content_from_api: # Only patch exists, or it's identical to original (no diff)
            st.info("Generated patch content shown below (no difference from original or original not available for diff).")
            st.text_area("Generated Patch", value=patched_content_from_api, height=300, disabled=True, key="patch_only_display")
        elif original_content:
            st.info("Original content loaded, but no patch has been generated yet or an error occurred.")
            st.text_area("Original Content", value=original_content, height=300, disabled=True, key="original_only_display")
        else:
            st.info("No patch or original content available to display diff.")

        # Patch Editor (always show if patch exists from API, allowing edits)
        if patched_content_from_api is not None: # Check if patch key exists and is not None
            st.markdown("### ✏️ Edit Patch")
            # The editor takes the API patch as its initial value.
            # If the user edits it, st.session_state.analysis_results['patch'] will be updated.
            edited_patch = st_ace(
                value=st.session_state.analysis_results.get('patch', ''), # Use current session state value (could be edited)
                language="python", # TODO: Make configurable if language changes
                theme="monokai",
                height=300,
                key="patch_editor_ace",
                auto_update=True # Updates session state on change if widget value is assigned to session state elsewhere
            )
            # Update session state if editor content has changed from what's currently in the session state (originating from API or previous edit)
            if edited_patch != st.session_state.analysis_results.get('patch'):
                st.session_state.analysis_results['patch'] = edited_patch
                # st.experimental_rerun() # Usually not needed with st_ace if auto_update handles binding well
                st.caption("Patch updated with your edits.")


        st.markdown("### 💬 Explanation")
        explanation = st.session_state.analysis_results.get('explanation', 'No explanation available.')
        st.text_area("Patch Explanation", value=explanation, height=150, disabled=True, key="explanation_display")

with tab2: # QA Tab
    st.subheader("Run Quality Assurance on Patch")
    if st.button("🛡️ Run QA on Patch", key="run_qa_button"):
        current_patch_content = st.session_state.analysis_results.get('patch') # This is the potentially edited patch
        original_trace = st.session_state.analysis_results.get('trace')
        source_files = st.session_state.analysis_results.get('source_files_content', {})
        patched_file_name = st.session_state.analysis_results.get('patched_file_name')

        if current_patch_content is None: # Check for None explicitly
            st.warning("Please run analysis and generate/edit a patch first.")
        elif not patched_file_name:
            st.warning("Patched file name is missing from analysis results. Please re-run analysis.")
        else:
            with st.spinner("🛡️ Running QA on the patch..."):
                payload = {
                    "trace": original_trace,
                    "patch": current_patch_content,
                    "language": "python", # TODO: Configurable
                    "source_files": source_files,
                    "patched_file_name": patched_file_name
                }
                qa_data = make_api_request("POST", QA_URL, json_payload=payload, operation_name="QA")

                if qa_data:
                    st.session_state.qa_result = qa_data
                    st.success("✅ QA complete.")
                    logger.info("QA successful, results updated in session state.")
                else:
                    st.session_state.qa_result = None # Clear previous result
                    st.error("QA failed. See error message above or check logs.")

    if st.session_state.get('qa_result'):
        st.markdown("### LLM Review")
        st.markdown(st.session_state.qa_result.get("llm_qa_result", "No LLM feedback provided."))

        st.markdown("### Static Analysis")
        static_analysis_result = st.session_state.qa_result.get("static_analysis_result", {})
        if static_analysis_result and isinstance(static_analysis_result, dict) and static_analysis_result:
            st.json(static_analysis_result)
        elif static_analysis_result: # If it's not an empty dict but some other form of "empty"
            st.info(f"Static analysis returned: {static_analysis_result}")
        else:
            st.info("No static analysis results available.")

with tab3: # Docs Tab
    st.subheader("📘 Auto-Generated Documentation")
    doc_summary = st.session_state.analysis_results.get("doc_summary", "No documentation summary available. Run analysis first.")
    st.markdown(doc_summary if doc_summary else "_No documentation generated yet._")


with tab4: # Issue Inbox Tab
    st.subheader("📥 Issue Inbox")
    if st.button("🔄 Refresh Inbox", key="refresh_inbox_button"):
        st.session_state.inbox_data = None # Clear cached data in session state
        st.rerun()

    # Fetch data only if not in session_state or if explicitly cleared
    if st.session_state.inbox_data is None:
        with st.spinner("Loading inbox..."):
            inbox_content = make_api_request("GET", INBOX_URL, operation_name="Issue Inbox")
            if inbox_content is not None: # Check if None, not just falsy
                 st.session_state.inbox_data = inbox_content
            # If inbox_content is None, an error was already shown by make_api_request

    inbox = st.session_state.get("inbox_data") # Use .get for safety

    if inbox and "issues" in inbox and isinstance(inbox["issues"], list):
        if not inbox["issues"]:
            st.info("No issues currently in the inbox.")
        for i, issue in enumerate(inbox["issues"]):
            issue_id = issue.get('id', f'UnknownID_{i}')
            issue_classification = issue.get('classification', 'N/A')
            issue_status = issue.get('status', 'N/A')
            expander_title = f"Issue {issue_id} - {issue_classification} [{issue_status}]"

            with st.expander(expander_title, expanded=False):
                st.json(issue) # Display full issue details
                if st.button(f"▶️ Trigger Workflow for Issue {issue_id}", key=f"trigger_workflow_button_{issue_id}"):
                    with st.spinner(f"Triggering workflow for {issue_id}..."):
                        response = make_api_request(
                            "POST",
                            WORKFLOW_RUN_URL,
                            json_payload={"issue_id": issue_id},
                            operation_name=f"Workflow Trigger for {issue_id}"
                        )
                        if response:
                            st.success(f"Workflow successfully triggered for {issue_id}! Details: {response.get('message', 'Check status tab.')}")
                            st.session_state.inbox_data = None # Refresh inbox to reflect potential status changes
                            st.rerun()
                        # Error already handled by make_api_request
    elif inbox is not None: # Inbox data was fetched but not in expected format
        st.warning("Inbox data received is not in the expected format or contains no 'issues' list.")
        logger.warning(f"Unexpected inbox data format: {inbox}")
    # If inbox is None, an error was already shown during fetch attempt


# --- Autonomous Workflow Orchestration Tab ---
with tab5:
    st.subheader("🤖 Autonomous Workflow Orchestration")
    if autonomous_tab_imported and callable(show_autonomous_workflow_tab):
        logger.info("Loading Autonomous Workflow Orchestration tab content.")
        # Pass necessary parameters like BACKEND_URL if the imported function needs them
        show_autonomous_workflow_tab(BACKEND_URL)
    elif not autonomous_tab_imported:
        # Error message is already shown at the top of the page
        st.info("The content for this tab could not be loaded due to an import error (see details at the top of the page).")
    else: # Imported but not callable (should not happen with proper import)
        st.error("The Autonomous Workflow Orchestration module was imported but is not callable.")
        logger.error("show_autonomous_workflow_tab is not callable.")

# --- Workflow Status Tab ---
with tab6:
    st.subheader("🔁 Live Workflow Timeline")
    if st.button("🔄 Refresh Status", key="refresh_status_button"):
        st.session_state.workflow_status = None # Clear session state cache
        st.rerun()

    if st.session_state.workflow_status is None:
        with st.spinner("Loading workflow status..."):
            status_data = make_api_request("GET", WORKFLOW_STATUS_URL, operation_name="Workflow Status")
            if status_data is not None:
                st.session_state.workflow_status = status_data
            # Error handled by make_api_request

    workflow_status_data = st.session_state.get("workflow_status")
    if workflow_status_data:
        st.json(workflow_status_data) # Assuming the status is well-formatted JSON
    elif workflow_status_data is not None: # Fetched but empty or unexpected
        st.info("Workflow status data is currently empty or in an unexpected format.")
        logger.info(f"Workflow status data was present but possibly empty/unexpected: {workflow_status_data}")
    # If None, error already handled


# === Voice Agent Section ===
st.markdown("---")
st.markdown("## 🎙️ DebugIQ Voice Agent")
st.caption("Note: Real-time voice processing in web apps can be resource-intensive. For production with many users, consider dedicated backend audio processing services.")

# webrtc_streamer component handles its own UI (Start/Stop button)
# Key ensures component re-initialization if BACKEND_URL changes, which might be desired if it affects behavior
try:
    ctx = webrtc_streamer(
        key=f"voice_agent_stream_{BACKEND_URL}",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        ),
        # audio_receiver_size is deprecated. Buffering is handled manually.
        # send_target_rate_bits_per_sec can be used to suggest bitrate if needed
        # desired_playing_state can be used to control play/pause from server if bidirectional
    )
except Exception as e: # Catch potential errors during webrtc_streamer initialization
    st.error(f"Failed to initialize voice agent: {e}")
    logger.exception("Error initializing webrtc_streamer")
    ctx = None # Ensure ctx is None if initialization fails

if ctx and ctx.audio_receiver:
    try:
        audio_frames = ctx.audio_receiver.get_frames(timeout=0.1) # Non-blocking with timeout

        if audio_frames:
            current_sample_rate = st.session_state.audio_sample_rate
            current_sample_width = st.session_state.audio_sample_width
            current_num_channels = st.session_state.audio_num_channels

            # Attempt to infer audio parameters from the first frame if defaults are still set
            # This assumes consistency across frames from the same stream source
            first_frame_format = audio_frames[0].format
            if first_frame_format:
                if current_sample_rate == DEFAULT_VOICE_SAMPLE_RATE and first_frame_format.rate:
                    st.session_state.audio_sample_rate = first_frame_format.rate
                    logger.info(f"Inferred sample rate: {first_frame_format.rate}")
                if current_sample_width == DEFAULT_VOICE_SAMPLE_WIDTH and first_frame_format.bytes: # This might be sample width
                    st.session_state.audio_sample_width = first_frame_format.bytes # Usually 2 for s16
                    logger.info(f"Inferred sample width (bytes): {first_frame_format.bytes}")
                if current_num_channels == DEFAULT_VOICE_CHANNELS and first_frame_format.channels:
                    st.session_state.audio_num_channels = first_frame_format.channels
                    logger.info(f"Inferred number of channels: {first_frame_format.channels}")

            for frame in audio_frames:
                # Ensure frame is in a format we can process (e.g., s16 PCM)
                # This part is crucial and depends heavily on the audio source format.
                # common formats: 's16' (signed 16-bit int), 'flt' (float)
                if frame.format.name == 's16':
                    audio_data = frame.to_ndarray().tobytes()
                elif frame.format.name in ['f32', 'flt32', 'flt']: # Common float formats
                    # Convert float32 to int16. Max value of int16 is 2**15 - 1.
                    float_array = frame.to_ndarray()
                    int16_array = (float_array * (2**15 -1)).astype(np.int16)
                    audio_data = int16_array.tobytes()
                else:
                    logger.warning(f"Unsupported audio frame format: {frame.format.name}. Skipping frame.")
                    continue

                st.session_state.audio_buffer += audio_data
                st.session_state.audio_frame_count += frame.samples # Number of samples in this frame

            st.sidebar.caption(f"Audio Buffered: {st.session_state.audio_frame_count} samples (~{st.session_state.audio_frame_count / st.session_state.audio_sample_rate:.2f}s)")

            # Process buffer periodically
            processing_threshold_samples = AUDIO_PROCESSING_THRESHOLD_SECONDS * st.session_state.audio_sample_rate
            if st.session_state.audio_frame_count >= processing_threshold_samples and st.session_state.audio_buffer:
                st.info(f"🎙️ Processing ~{st.session_state.audio_frame_count / st.session_state.audio_sample_rate:.2f}s of audio...")
                temp_wav_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                        temp_wav_file_path = tmp_wav_file.name

                    with wave.open(temp_wav_file_path, 'wb') as wav_writer:
                        wav_writer.setnchannels(st.session_state.audio_num_channels)
                        wav_writer.setsampwidth(st.session_state.audio_sample_width)
                        wav_writer.setframerate(st.session_state.audio_sample_rate)
                        wav_writer.writeframes(st.session_state.audio_buffer)
                    logger.info(f"Temporary WAV file created at {temp_wav_file_path} with {st.session_state.audio_frame_count} frames.")

                    with open(temp_wav_file_path, "rb") as f_audio:
                        files_payload = {"file": (f"audio_segment_{abs(hash(temp_wav_file_path))}.wav", f_audio, "audio/wav")} # More descriptive filename
                        transcribe_response = requests.post(TRANSCRIBE_URL, files=files_payload, timeout=20) # Timeout for transcribe
                    transcribe_response.raise_for_status()
                    transcript_data = transcribe_response.json()
                    transcript = transcript_data.get("transcript")

                    if transcript:
                        st.success(f"🗣️ You (Transcribed): \"{transcript}\"")
                        logger.info(f"Transcription successful: {transcript}")
                        command_response_data = make_api_request(
                            "POST",
                            COMMAND_URL,
                            json_payload={"text_command": transcript},
                            operation_name="Voice Command"
                        )
                        if command_response_data:
                            st.info(f"🤖 DebugIQ Agent: {command_response_data.get('spoken_text', 'No spoken response generated.')}")
                            # Potentially trigger actions based on command_data.get('action_code') etc.
                        else:
                             st.warning("Voice command sent, but no actionable response from agent.")
                    else:
                        st.info("Transcription returned empty. Try speaking more clearly or ensure microphone is active.")
                        logger.info("Transcription was empty.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Voice processing error (API): {e}")
                    logger.exception("Error during voice transcription/command API call")
                except wave.Error as e:
                    st.error(f"Could not create WAV file: {e}")
                    logger.exception("Wave file creation error")
                except Exception as e:
                    st.error(f"An unexpected error occurred during voice processing: {e}")
                    logger.exception("Unexpected error in voice processing block")
                finally:
                    if temp_wav_file_path and os.path.exists(temp_wav_file_path):
                        try:
                            os.remove(temp_wav_file_path)
                            logger.info(f"Temporary WAV file {temp_wav_file_path} removed.")
                        except OSError as e:
                            logger.error(f"Error removing temporary WAV file {temp_wav_file_path}: {e}")
                    # Clear buffer and count AFTER processing (or attempting to)
                    st.session_state.audio_buffer = b""
                    st.session_state.audio_frame_count = 0
                    # st.rerun() # Might be needed if state changes should immediately reflect elsewhere

    except av.error.TimeoutError: # Specifically catch av.error.TimeoutError
        pass # Expected if no frames are available within the timeout, normal operation
    except Exception as e:
        # Catch other potential errors from audio_receiver or frame processing
        if ctx and ctx.audio_receiver and not ctx.audio_receiver.is_closed: # Check if receiver is still active
             st.warning(f"An issue occurred with the audio stream: {e}. Try restarting the voice agent if issues persist.")
             logger.error(f"Error processing audio frames: {e}", exc_info=True)
        # If receiver is closed, it might be user stopping it, so error might not be needed.

elif ctx and not ctx.audio_receiver:
    # This state means the component is active but not receiving (e.g., user stopped microphone)
    if st.session_state.audio_buffer: # If there's leftover buffer when mic stops
        logger.info("Audio stream stopped with remaining buffer. Clearing buffer.")
        st.session_state.audio_buffer = b""
        st.session_state.audio_frame_count = 0
    # st.sidebar.caption("Voice agent stopped or microphone not active.") # Optional feedback
