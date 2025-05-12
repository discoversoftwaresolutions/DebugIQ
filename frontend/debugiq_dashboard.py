# dashboard.py

import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings # Ensured ClientSettings is imported
import numpy as np
import av
# HtmlDiff is used as difflib.HtmlDiff().make_file()
import streamlit.components.v1 as components
import wave
import json
import logging
import base64 # For handling audio data if sent as base64
import re # For GitHub URL parsing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
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

# --- Helper Function to Clear GitHub State ---
def clear_all_github_session_state():
    """Resets all GitHub related session state AND clears loaded analysis files."""
    st.session_state.github_repo_url_input = ""
    st.session_state.current_github_repo_url = None
    st.session_state.github_branches = []
    st.session_state.github_selected_branch = None
    st.session_state.github_path_stack = [""]
    st.session_state.github_repo_owner = None
    st.session_state.github_repo_name = None
    
    # Ensure analysis_results is initialized before trying to access sub-keys
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {"trace": None, "source_files_content": {}} # Default structure
    
    st.session_state.analysis_results["trace"] = None
    st.session_state.analysis_results["source_files_content"] = {}
    logger.info("Cleared all GitHub session state and related analysis inputs.")

# --- Streamlit Config ---
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# --- Initialize Session State Defaults ---
session_defaults = {
    "audio_sample_rate": DEFAULT_VOICE_SAMPLE_RATE,
    "audio_sample_width": DEFAULT_VOICE_SAMPLE_WIDTH,
    "audio_num_channels": DEFAULT_VOICE_CHANNELS,
    "audio_buffer": b"",
    "audio_frame_count": 0,
    "chat_history": [], # For Gemini integration (if added later)
    "edited_patch": "",  # From patch tab editor
    
    # GitHub related state
    "github_repo_url_input": "", # Current value in the input box
    "current_github_repo_url": None, # The URL that was last successfully processed for branches
    "github_branches": [],
    "github_selected_branch": None,
    "github_path_stack": [""],
    "github_repo_owner": None,
    "github_repo_name": None,
    
    # Analysis results (populated by GitHub, manual upload, or backend analysis)
    "analysis_results": {
        "trace": None,
        "source_files_content": {}, # Dict of {filepath_in_repo: content}
        "patch": None, # Patched code string from backend
        "explanation": None, # Explanation from backend
        "doc_summary": None, # Doc summary from backend
        "patched_file_name": None, # Filename that the patch applies to
        "original_patched_file_content": None # Original content of the file that was patched
    },
    
    "qa_result": None,
    "inbox_data": None,
    "workflow_status_data": None,
    "metrics_data": None,
    "qa_code_to_validate": None # Temp state for QA tab
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- API Endpoints --- (Defined once)
ANALYZE_URL = f"{BACKEND_URL}/debugiq/analyze"
SUGGEST_PATCH_URL = f"{BACKEND_URL}/debugiq/suggest_patch" # As used in original Tab 1
QA_URL = f"{BACKEND_URL}/qa/"
DOC_URL = f"{BACKEND_URL}/doc/"
VOICE_TRANSCRIBE_URL = f"{BACKEND_URL}/voice/transcribe"
GEMINI_CHAT_URL = f"{BACKEND_URL}/gemini-chat"
ISSUES_INBOX_URL = f"{BACKEND_URL}/issues/inbox"
WORKFLOW_RUN_URL = f"{BACKEND_URL}/workflow/run"
WORKFLOW_CHECK_URL = f"{BACKEND_URL}/workflow/status"
METRICS_URL = f"{BACKEND_URL}/metrics/summary"

# --- API Helper Function --- (Defined once)
def make_api_request(method, url, json_payload=None, files=None, operation_name="API Call"):
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
    except requests.exceptions.HTTPError as http_err:
        error_text = http_err.response.text if http_err.response else "No details from server."
        logger.error(f"HTTP error for {operation_name} to {url}: {http_err}. Response: {error_text}")
        st.error(f"{operation_name} failed: {http_err}. Server said: {error_text}")
        return {"error": str(http_err), "details": error_text}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"RequestException for {operation_name} to {url}: {req_err}")
        st.error(f"Communication error for {operation_name}: {req_err}")
        return {"error": str(req_err)}
    except Exception as e:
        logger.exception(f"Unexpected error during {operation_name} to {url}")
        st.error(f"Unexpected error with {operation_name}: {e}")
        return {"error": str(e)}

# === GitHub Repo Integration Sidebar ===
with st.sidebar:
    st.markdown("### 📦 Load From GitHub Repo")
    st.text_input(
        "Public GitHub URL",
        placeholder="https://github.com/user/repo",
        key="github_repo_url_input" # Session state automatically updated
    )

    if st.button("Load Repo / Reset GitHub View", key="load_reset_github_btn"):
        # When button is clicked, process the current input value
        # If input is empty, it effectively serves as a reset.
        if not st.session_state.github_repo_url_input:
            clear_all_github_session_state()
        else:
            # Clear previous state before loading new repo, except the input URL itself
            st.session_state.current_github_repo_url = None
            st.session_state.github_branches = []
            st.session_state.github_selected_branch = None
            st.session_state.github_path_stack = [""]
            st.session_state.github_repo_owner = None
            st.session_state.github_repo_name = None
            # Clear analysis files too, as new repo implies new context
            if "analysis_results" in st.session_state: # Ensure key exists
                 st.session_state.analysis_results["trace"] = None
                 st.session_state.analysis_results["source_files_content"] = {}
        st.rerun() # Rerun to process the URL below or reflect reset

    current_url_input = st.session_state.get("github_repo_url_input", "").strip()

    if current_url_input:
        # Process URL only if it's different from the last successfully processed one, or if forced by button
        # This logic is triggered on rerun after button click or URL input change (if st.rerun used)
        
        match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", current_url_input)
        if match:
            owner, repo = match.groups()
            
            # Update owner/repo if they changed from last processed valid URL
            if st.session_state.github_repo_owner != owner or st.session_state.github_repo_name != repo:
                st.session_state.github_repo_owner = owner
                st.session_state.github_repo_name = repo
                st.session_state.current_github_repo_url = None # Mark for branch reload

            # Fetch branches if current_github_repo_url doesn't match the input field (implies new or reset)
            if st.session_state.current_github_repo_url != current_url_input:
                st.session_state.current_github_repo_url = current_url_input # Mark as processed for branches
                api_branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
                logger.info(f"Fetching branches from {api_branches_url}")
                try:
                    with st.spinner(f"Loading branches for {owner}/{repo}..."):
                        branches_res = requests.get(api_branches_url, timeout=10)
                        branches_res.raise_for_status()
                        st.session_state.github_branches = [b["name"] for b in branches_res.json()]
                    if st.session_state.github_branches:
                        if st.session_state.github_selected_branch not in st.session_state.github_branches:
                            st.session_state.github_selected_branch = st.session_state.github_branches[0]
                        st.session_state.github_path_stack = [""] # Reset path
                        st.success(f"Repo '{owner}/{repo}' branches loaded.")
                    else:
                        st.warning("No branches found for this repository.")
                        st.session_state.github_branches = [] # Ensure empty list
                        st.session_state.github_selected_branch = None
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Could not fetch branches: {e}")
                    clear_all_github_session_state() # Full reset on this kind of error
                    st.rerun() # Rerun to clear the UI elements dependent on this state
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decoding branch data: {e}")
                    clear_all_github_session_state()
                    st.rerun()

            # Display branch selector if branches are loaded
            branches_list_display = st.session_state.get("github_branches", [])
            if branches_list_display:
                try:
                    branch_idx = branches_list_display.index(st.session_state.github_selected_branch) if st.session_state.github_selected_branch in branches_list_display else 0
                except ValueError:
                    branch_idx = 0
                
                selected_branch_name_display = st.selectbox(
                    "Branch",
                    branches_list_display,
                    index=branch_idx,
                    key="github_branch_selector_widget" # Changed key
                )
                if selected_branch_name_display != st.session_state.github_selected_branch:
                    st.session_state.github_selected_branch = selected_branch_name_display
                    st.session_state.github_path_stack = [""] # Reset path
                    st.rerun()

            # Display file browser if owner, repo, and branch are set
            gh_owner_val = st.session_state.get("github_repo_owner")
            gh_repo_val = st.session_state.get("github_repo_name")
            gh_branch_val = st.session_state.get("github_selected_branch")

            if gh_owner_val and gh_repo_val and gh_branch_val:
                path_stack_display = st.session_state.github_path_stack
                current_dir_display = "/".join([p for p in path_stack_display if p])

                @st.cache_data(ttl=120, show_spinner="Fetching directory content...")
                def fetch_github_dir_content_cached(owner_val, repo_val, dir_path_val, branch_val):
                    api_url = f"https://api.github.com/repos/{owner_val}/{repo_val}/contents/{dir_path_val}?ref={branch_val}"
                    logger.info(f"Fetching GitHub directory (cached): {api_url}")
                    try:
                        response = requests.get(api_url, timeout=10)
                        response.raise_for_status()
                        return response.json()
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Cannot fetch content for '{dir_path_val}': {e}")
                        return None
                    except json.JSONDecodeError as e:
                        st.warning(f"Error decoding content JSON for '{dir_path_val}': {e}.")
                        return None
                
                entries = fetch_github_dir_content_cached(gh_owner_val, gh_repo_val, current_dir_display, gh_branch_val)

                if entries is not None:
                    dirs_in_path = sorted([e["name"] for e in entries if e["type"] == "dir"])
                    files_in_path = sorted([e["name"] for e in entries if e["type"] == "file" and any(e["name"].endswith(ext) for ext in RECOGNIZED_FILE_EXTENSIONS)])

                    st.markdown("##### 📁 Navigate Directories")
                    if current_dir_display:
                        if st.button("⬆️ .. (Up)", key=f"gh_up_dir_{current_dir_display}", use_container_width=True):
                            st.session_state.github_path_stack.pop()
                            st.rerun()
                    for d_item_name in dirs_in_path:
                        if st.button(f"📁 {d_item_name}", key=f"gh_dir_{current_dir_display}_{d_item_name}", use_container_width=True):
                            st.session_state.github_path_stack.append(d_item_name)
                            st.rerun()
                    
                    st.markdown("##### 📄 Load Files")
                    for f_item_name in files_in_path:
                        if st.button(f"📄 {f_item_name}", key=f"gh_file_{current_dir_display}_{f_item_name}", use_container_width=True):
                            file_url_path_segment = f"{current_dir_display}/{f_item_name}".strip("/")
                            raw_file_content_url = f"https://raw.githubusercontent.com/{gh_owner_val}/{gh_repo_val}/{gh_branch_val}/{file_url_path_segment}"
                            logger.info(f"Fetching file content: {raw_file_content_url}")
                            try:
                                with st.spinner(f"Loading {f_item_name}..."):
                                    file_res = requests.get(raw_file_content_url, timeout=10)
                                    file_res.raise_for_status()
                                file_content_str = file_res.text
                                st.success(f"Loaded: {f_item_name}")

                                full_path_key = os.path.join(current_dir_display, f_item_name).replace("\\","/") if current_dir_display else f_item_name
                                
                                if f_item_name.endswith(TRACEBACK_EXTENSION):
                                    st.session_state.analysis_results['trace'] = file_content_str
                                    st.info(f"'{full_path_key}' loaded as traceback.")
                                else: # Already filtered by RECOGNIZED_FILE_EXTENSIONS for button creation
                                    st.session_state.analysis_results['source_files_content'][full_path_key] = file_content_str
                                    st.info(f"'{full_path_key}' loaded as source file.")
                                st.rerun() # Rerun to update the "Loaded Analysis Inputs" expander
                            except requests.exceptions.RequestException as e:
                                st.error(f"Failed to load file {f_item_name}: {e}")
                elif current_dir_display: # entries is None and not in root
                     st.warning("Could not list directory contents.")
        elif current_url_input: # URL is present but doesn't match regex
            st.warning("Invalid GitHub repo URL format. Use: https://github.com/owner/repo")
            # No full clear here, user might be correcting the URL.

    st.markdown("---")
    with st.expander("📬 Loaded Analysis Inputs", expanded=True):
        trace = st.session_state.analysis_results.get('trace')
        sources = st.session_state.analysis_results.get('source_files_content', {})
        if trace:
            st.text_area("Current Traceback:", value=trace, height=100, disabled=True, key="sidebar_trace_view")
        else:
            st.caption("No traceback loaded.")
        if sources:
            st.write("Current Source Files:")
            for name in sources.keys():
                st.caption(f"- {name}")
        else:
            st.caption("No source files loaded.")
    st.markdown("---")

# --- Manual File Uploader (Main Page Area) ---
st.markdown("### ⬆️ Or, Upload Files Manually")
manual_uploaded_files_list = st.file_uploader(
    "📄 Upload traceback (.txt) and/or source files",
    type=[ext.lstrip('.') for ext in RECOGNIZED_FILE_EXTENSIONS],
    accept_multiple_files=True,
    key="manual_uploader_widget_main"
)

if manual_uploaded_files_list:
    manual_trace_content_str = None
    manual_source_files_dict = {}
    manual_summary_list = []

    for uploaded_file_obj in manual_uploaded_files_list:
        try:
            file_content_bytes = uploaded_file_obj.getvalue()
            file_content_decoded = file_content_bytes.decode("utf-8")
            if uploaded_file_obj.name.endswith(TRACEBACK_EXTENSION):
                manual_trace_content_str = file_content_decoded
                manual_summary_list.append(f"Traceback: {uploaded_file_obj.name}")
            elif any(uploaded_file_obj.name.endswith(ext) for ext in SUPPORTED_SOURCE_EXTENSIONS):
                manual_source_files_dict[uploaded_file_obj.name] = file_content_decoded
                manual_summary_list.append(f"Source: {uploaded_file_obj.name}")
        except UnicodeDecodeError:
            st.error(f"Could not decode '{uploaded_file_obj.name}'. Please ensure UTF-8 encoding.")
        except Exception as e:
            st.error(f"Error processing '{uploaded_file_obj.name}': {e}")

    if manual_trace_content_str is not None:
        st.session_state.analysis_results['trace'] = manual_trace_content_str
        # If a trace is manually uploaded, it implies a new context.
        # Overwrite source files with only those uploaded *with* this trace.
        st.session_state.analysis_results['source_files_content'] = manual_source_files_dict 
        clear_all_github_session_state() # Also clear GitHub state
        st.success(f"Manually uploaded: {', '.join(manual_summary_list)}. GitHub selection cleared.")
        st.rerun()
    elif manual_source_files_dict: # Only source files, no trace in this batch
        # Add to existing source files. If you want to replace, change update to assignment.
        st.session_state.analysis_results['source_files_content'].update(manual_source_files_dict)
        clear_all_github_session_state()
        st.success(f"Manually added source files: {', '.join(manual_summary_list)}. GitHub selection cleared.")
        st.rerun()


# --- Tabs Setup --- (Defined once)
tabs_list = [
    "📄 Traceback + Patch",
    "✅ QA Validation",
    "📘 Documentation",
    "📣 Issue Notices",
    "🤖 Autonomous Workflow",
    "🔍 Workflow Check",
    "📊 Metrics"
]
tabs = st.tabs(tabs_list)

# --- Tab 1: Traceback + Patch ---
with tabs[0]:
    st.header("📄 Analyze Traceback & Generate/Edit Patch")

    # Display currently loaded data for context
    loaded_trace = st.session_state.analysis_results.get('trace')
    loaded_sources = st.session_state.analysis_results.get('source_files_content', {})
    if loaded_trace:
        st.info("Using loaded traceback:")
        st.text_area("Loaded Traceback", value=loaded_trace, height=100, disabled=True, key="tab1_loaded_trace_view")
    if loaded_sources:
        st.info(f"Using loaded source files: {', '.join(loaded_sources.keys())}")

    # Button from your original code for this tab
    # This assumes 'original_code' for the payload comes from the uploaded file logic
    # Now it should use session state.
    
    # This part of your original code was for a file uploader *within the tab*.
    # I've moved file uploading (manual and GitHub) to be global.
    # So, the button here should operate on st.session_state.analysis_results
    
    # Renaming button from your original for clarity
    if st.button("🔬 Analyze Loaded Data & Suggest Patch", key="tab1_analyze_btn"):
        if not loaded_trace and not loaded_sources:
            st.warning("Please load a traceback or source files using the sidebar or manual uploader first.")
        else:
            payload = {
                "trace": loaded_trace, # Use trace from session state
                "language":"python", # TODO: Make this dynamic if necessary
                "source_files": loaded_sources # Use source_files from session state
                }
            # Your original code used SUGGEST_PATCH_URL here. Let's stick to that if it's specific.
            # Otherwise, ANALYZE_URL might be more general if it returns patch, explanation etc.
            # For now, using SUGGEST_PATCH_URL as per your tab 1 logic.
            patch_api_response = make_api_request("POST", SUGGEST_PATCH_URL, json_payload=payload, operation_name="Patch Suggestion")

            if patch_api_response and not patch_api_response.get("error"):
                # Update session state with all relevant fields from response
                st.session_state.analysis_results['patch'] = patch_api_response.get("patched_code", "")
                st.session_state.analysis_results['explanation'] = patch_api_response.get("explanation") # Assuming backend sends this
                st.session_state.analysis_results['doc_summary'] = patch_api_response.get("doc_summary") # Assuming backend sends this
                st.session_state.analysis_results['patched_file_name'] = patch_api_response.get("patched_file_name") # Filename the patch applies to

                # For diffing, we need the original content of the specific file that was patched
                # This might require another piece of info from backend or smart handling of source_files
                # For now, if 'original_patched_file_content' is sent by backend, use it.
                # Otherwise, try to find it in loaded_sources if patched_file_name is known.
                original_content_for_diff = patch_api_response.get("original_patched_file_content")
                if not original_content_for_diff and patch_api_response.get("patched_file_name") in loaded_sources:
                    original_content_for_diff = loaded_sources[patch_api_response.get("patched_file_name")]
                st.session_state.analysis_results['original_patched_file_content'] = original_content_for_diff
                
                st.success("Patch suggestion received.")
            else:
                st.error(f"Patch generation failed. {patch_api_response.get('details', '') if patch_api_response else 'No response'}")
                # Clear previous patch attempt results
                st.session_state.analysis_results['patch'] = None
                st.session_state.analysis_results['original_patched_file_content'] = None
                st.session_state.analysis_results['patched_file_name'] = None


    # Display Diff View and Editor (logic adapted from my previous refactoring)
    display_original_content = st.session_state.analysis_results.get('original_patched_file_content')
    display_patched_code = st.session_state.analysis_results.get('patch')

    if display_original_content is not None and display_patched_code is not None:
        st.markdown("### 🔍 Diff View")
        if display_original_content != display_patched_code:
            html_diff_display = difflib.HtmlDiff(wrapcolumn=70, tabsize=4).make_table(
                display_original_content.splitlines(keepends=True),
                display_patched_code.splitlines(keepends=True),
                "Original", "Patched", context=True, numlines=3
            )
            components.html(html_diff_display, height=450, scrolling=True)
        else:
            st.info("The suggested patch makes no changes to the original content.")
        
        st.markdown("### ✍️ Edit Patch (Live)")
        edited_code_val = st_ace(
            value=display_patched_code, 
            language="python", # TODO: Infer from patched_file_name
            theme="monokai", 
            height=300, 
            key="tab1_patch_editor_ace"
        )
        if edited_code_val != display_patched_code: # Update if changed by user
            st.session_state.analysis_results['patch'] = edited_code_val
            st.session_state.edited_patch = edited_code_val # Legacy key
            st.caption("Patch updated in session.")
    elif display_patched_code is not None: # Only patched code available (e.g., new file, or original not provided for diff)
        st.markdown("### ✨ Generated/Patched Code")
        st.text_area("Code:", value=display_patched_code, height=300, disabled=True, key="tab1_patched_code_only_view")
        st.markdown("### ✍️ Edit This Code")
        edited_code_val_no_orig = st_ace(
            value=display_patched_code, language="python", theme="monokai", height=300, key="tab1_patch_editor_no_orig_ace"
        )
        if edited_code_val_no_orig != display_patched_code:
             st.session_state.analysis_results['patch'] = edited_code_val_no_orig
             st.session_state.edited_patch = edited_code_val_no_orig
             st.caption("Code updated in session.")
             
    # Display explanation if available
    explanation_text = st.session_state.analysis_results.get('explanation')
    if explanation_text:
        st.markdown("### 💬 Explanation")
        st.text_area("Explanation of Patch:", value=explanation_text, height=150, disabled=True, key="tab1_explanation_view")


# --- Other Tabs (using the structure from your file, with make_api_request) ---

with tabs[1]: # QA Validation
    st.header("✅ QA Validation")
    qa_code_for_validation = st.session_state.analysis_results.get('patch') # Use the current patch from session

    if qa_code_for_validation is not None:
        st.info("Using the current patched code (from Tab 1, including edits) for QA.")
        st.text_area("Code for QA:", value=qa_code_for_validation, height=200, disabled=True, key="qa_code_display")
        
        if st.button("🛡️ Run QA Validation", key="qa_run_validation_btn"):
            payload = {
                "code": qa_code_for_validation,
                "patched_file_name": st.session_state.analysis_results.get('patched_file_name'),
                "trace": st.session_state.analysis_results.get('trace'),
                "source_files": st.session_state.analysis_results.get('source_files_content')
            }
            qa_response = make_api_request("POST", QA_URL, json_payload=payload, operation_name="QA Validation")
            if qa_response and not qa_response.get("error"):
                st.session_state.qa_result = qa_response # Store full QA result
                st.success("QA Validation Complete.")
                st.json(qa_response) # Display raw JSON result
            else:
                st.error(f"QA Validation failed. {qa_response.get('details', '') if qa_response else 'No response.'}")
    else:
        st.warning("No patched code available from Tab 1 to validate. Please generate/load a patch first.")


with tabs[2]: # Documentation
    st.header("📘 Documentation")
    # Display doc_summary from analysis_results if available
    doc_summary_from_analysis = st.session_state.analysis_results.get('doc_summary')
    if doc_summary_from_analysis:
        st.subheader("Documentation Summary from Last Analysis:")
        st.markdown(doc_summary_from_analysis)
        st.markdown("---")

    st.subheader("Generate Documentation for Specific Code:")
    doc_code_input_val = st.text_area("Paste code to generate documentation:", key="doc_code_input_area_tab3", height=200)
    if st.button("📝 Generate Ad-hoc Documentation", key="doc_generate_btn_tab3"):
        if doc_code_input_val:
            payload = {"code": doc_code_input_val}
            doc_response = make_api_request("POST", DOC_URL, json_payload=payload, operation_name="Documentation Generation")
            if doc_response and not doc_response.get("error"):
                st.markdown(doc_response.get("doc", "No documentation generated or 'doc' key missing."))
            else:
                st.error(f"Doc generation failed. {doc_response.get('details', '') if doc_response else 'No response.'}")
        else:
            st.warning("Please paste code into the text area.")

with tabs[3]: # Issue Notices
    st.header("📣 Issue Notices") # Simpler title from your file
    if st.button("🔄 Refresh Issue Notices", key="issues_fetch_btn_tab4"): # Updated key for uniqueness
        issues_response = make_api_request("GET", ISSUES_INBOX_URL, operation_name="Fetch Issues")
        if issues_response and not issues_response.get("error"):
            st.session_state.inbox_data = issues_response # Cache
            st.json(issues_response)
        else:
            st.error(f"Issue data fetch failed. {issues_response.get('details', '') if issues_response else 'No response.'}")
            st.session_state.inbox_data = None
    elif st.session_state.get('inbox_data'):
        st.info("Displaying cached issues. Refresh for latest.")
        st.json(st.session_state.inbox_data)
    else:
        st.info("Click button to fetch issue notices.")


with tabs[4]: # Autonomous Workflow
    st.header("🤖 Run DebugIQ Autonomous Workflow")
    issue_id_val = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101", key="workflow_issue_id_input_tab5")
    if st.button("▶️ Run Workflow", key="workflow_run_btn_tab5"):
        if issue_id_val:
            payload = {"issue_id": issue_id_val}
            workflow_response = make_api_request("POST", WORKFLOW_RUN_URL, json_payload=payload, operation_name="Workflow Run")
            if workflow_response and not workflow_response.get("error"):
                st.success(f"Workflow triggered for {issue_id_val}.")
                st.json(workflow_response)
            else:
                st.error(f"Workflow execution failed. {workflow_response.get('details', '') if workflow_response else 'No response.'}")
        else:
            st.warning("Please enter an Issue ID.")

with tabs[5]: # Workflow Check
    st.header("🔍 Workflow Status Check")
    if st.button("🔄 Refresh Workflow Status", key="workflow_check_btn_tab6"):
        workflow_check_response = make_api_request("GET", WORKFLOW_CHECK_URL, operation_name="Workflow Status Check")
        if workflow_check_response and not workflow_check_response.get("error"):
            st.session_state.workflow_status_data = workflow_check_response # Cache
            st.json(workflow_check_response)
        else:
            st.error(f"Workflow check failed. {workflow_check_response.get('details', '') if workflow_check_response else 'No response.'}")
            st.session_state.workflow_status_data = None
    elif st.session_state.get('workflow_status_data'):
        st.info("Displaying cached workflow status. Refresh for latest.")
        st.json(st.session_state.workflow_status_data)
    else:
        st.info("Click button to fetch workflow status.")

with tabs[6]: # Metrics
    st.header("📊 Metrics")
    if st.button("📈 Fetch Metrics", key="metrics_fetch_btn_tab7"):
        metrics_response = make_api_request("GET", METRICS_URL, operation_name="Fetch Metrics")
        if metrics_response and not metrics_response.get("error"):
            st.session_state.metrics_data = metrics_response # Cache
            st.json(metrics_response) # Consider a more visual display
        else:
            st.error(f"Metrics fetch failed. {metrics_response.get('details', '') if metrics_response else 'No response.'}")
            st.session_state.metrics_data = None
    elif st.session_state.get('metrics_data'):
        st.info("Displaying cached metrics. Refresh for latest.")
        st.json(st.session_state.metrics_data)
    else:
        st.info("Click button to fetch metrics.")


# === Voice Agent Section (Bi-Directional Gemini Integration) ===
st.markdown("---")
st.markdown("## 🎙️ DebugIQ Voice Agent (with Gemini)") # Title reflecting Gemini goal
st.caption("Speak your query, and DebugIQ's Gemini assistant will respond.")

# Display chat history
# This should be at the beginning of this UI section to show existing messages on rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio_base64" in message and message["role"] == "assistant":
            try:
                audio_bytes = base64.b64decode(message["audio_base64"])
                st.audio(audio_bytes, format="audio/mp3") # Or "audio/wav" depending on your backend
            except Exception as e:
                logger.error(f"Error playing audio for assistant message: {e}")
                st.caption("(Could not play audio for this message)")

try:
    # Ensure ClientSettings is imported at the top of your script:
    # from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    ctx = webrtc_streamer(
        key=f"gemini_voice_agent_stream_{BACKEND_URL}", 
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings( # Using ClientSettings
             rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
             media_stream_constraints={"audio": True, "video": False}
        )
    )
except Exception as e:
    st.error(f"Failed to initialize voice agent: {e}")
    logger.exception("Error initializing webrtc_streamer for Gemini Voice Agent")
    ctx = None

if ctx and ctx.audio_receiver:
    status_indicator_voice = st.empty() 
    try:
        if not ctx.state.playing: # Only show "Listening" if a connection is active but no audio is being sent
             status_indicator_voice.caption("Click 'Start' on the voice agent to speak.")
        else:
             status_indicator_voice.info("Listening...")

        audio_frames = ctx.audio_receiver.get_frames(timeout=0.2)

        if audio_frames:
            status_indicator_voice.info("Processing audio...")
            # (Audio parameter inference logic - keep as is)
            first_frame_format = audio_frames[0].format
            if first_frame_format:
                if st.session_state.audio_sample_rate == DEFAULT_VOICE_SAMPLE_RATE and first_frame_format.rate:
                    st.session_state.audio_sample_rate = first_frame_format.rate
                if st.session_state.audio_sample_width == DEFAULT_VOICE_SAMPLE_WIDTH and first_frame_format.bytes:
                    st.session_state.audio_sample_width = first_frame_format.bytes
                if st.session_state.audio_num_channels == DEFAULT_VOICE_CHANNELS and first_frame_format.channels:
                    st.session_state.audio_num_channels = first_frame_format.channels
            
            for frame in audio_frames:
                if frame.format.name == 's16':
                    audio_data = frame.to_ndarray().tobytes()
                elif frame.format.name in ['f32', 'flt32', 'flt']:
                    float_array = frame.to_ndarray()
                    int16_array = (float_array * (2**15 - 1)).astype(np.int16)
                    audio_data = int16_array.tobytes()
                else:
                    logger.warning(f"Unsupported audio frame format: {frame.format.name}. Skipping frame.")
                    continue
                st.session_state.audio_buffer += audio_data
                st.session_state.audio_frame_count += frame.samples
            
            st.sidebar.caption(f"Audio Buffered: {st.session_state.audio_frame_count} samples (~{st.session_state.audio_frame_count / st.session_state.audio_sample_rate:.2f}s)")

            processing_threshold_samples = AUDIO_PROCESSING_THRESHOLD_SECONDS * st.session_state.audio_sample_rate
            if st.session_state.audio_frame_count >= processing_threshold_samples and st.session_state.audio_buffer:
                status_indicator_voice.info("🎙️ Transcribing and preparing request...")
                temp_wav_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                        temp_wav_file_path = tmp_f.name
                    with wave.open(temp_wav_file_path, 'wb') as wf:
                        wf.setnchannels(st.session_state.audio_num_channels)
                        wf.setsampwidth(st.session_state.audio_sample_width)
                        wf.setframerate(st.session_state.audio_sample_rate)
                        wf.writeframes(st.session_state.audio_buffer)
                    
                    logger.info(f"Temporary WAV file for Gemini agent created at {temp_wav_file_path}")

                    with open(temp_wav_file_path, "rb") as f_aud:
                        files_payload = {"file": ("segment.wav", f_aud, "audio/wav")}
                        transcribe_resp = make_api_request(
                            "POST", VOICE_TRANSCRIBE_URL, files=files_payload, 
                            operation_name="Voice Transcription for Agent"
                        )
                    
                    transcript = transcribe_resp.get("transcript") if transcribe_resp and not transcribe_resp.get("error") else None
                    
                    if transcript:
                        logger.info(f"Agent Transcription successful: {transcript}")
                        
                        # Add user message to history
                        st.session_state.chat_history.append({"role": "user", "content": transcript})
                        
                        # Prepare payload for Gemini (or your command processor)
                        # Send the current transcript and relevant history
                        # Example: send last few messages or handle context on backend
                        MAX_HISTORY_LEN = 10 # Example limit
                        history_for_payload = [msg for msg in st.session_state.chat_history if msg['role'] == 'user' or msg.get('is_gemini_response')] # Adapt if needed
                        if len(history_for_payload) > MAX_HISTORY_LEN:
                            history_for_payload = history_for_payload[-MAX_HISTORY_LEN:]

                        agent_payload = {
                            "text_command": transcript, 
                            "conversation_history": history_for_payload[:-1] # History before current utterance
                        }
                        status_indicator_voice.info(f"🗣️ You: \"{transcript}\" - Sending to DebugIQ Agent...")

                        # Using GEMINI_CHAT_URL for the agent interaction
                        agent_response_data = make_api_request(
                            "POST", 
                            GEMINI_CHAT_URL, 
                            json_payload=agent_payload, 
                            operation_name="DebugIQ Voice Agent Interaction"
                        )

                        if agent_response_data and not agent_response_data.get("error"):
                            assistant_text = agent_response_data.get("text_response", "I'm not sure how to respond to that.")
                            assistant_audio_b64 = agent_response_data.get("audio_content_base64")
                            
                            assistant_msg_obj = {"role": "assistant", "content": assistant_text, "is_gemini_response": True} # Flagging as Gemini response
                            if assistant_audio_b64:
                                assistant_msg_obj["audio_base64"] = assistant_audio_b64
                            st.session_state.chat_history.append(assistant_msg_obj)
                        else:
                            err_detail = agent_response_data.get('details', 'Failed to get a response from the agent.') if agent_response_data else "No response from agent."
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {err_detail}", "is_gemini_response": True})
                        
                        status_indicator_voice.empty() 
                        st.rerun() # Rerun to display the new user and assistant messages in the chat history

                    else:
                        status_indicator_voice.warning("Transcription was empty or failed. Please try speaking again.")
                        if transcribe_resp and transcribe_resp.get("error"):
                             logger.error(f"Agent transcription error: {transcribe_resp.get('details')}")
                
                except Exception as e_proc:
                    status_indicator_voice.error(f"An error occurred during voice processing: {e_proc}")
                    logger.exception("Error in voice agent processing block")
                finally:
                    if temp_wav_file_path and os.path.exists(temp_wav_file_path):
                        try: os.remove(temp_wav_file_path)
                        except OSError as e_os: logger.error(f"Error removing temp WAV: {e_os}")
                    st.session_state.audio_buffer = b""
                    st.session_state.audio_frame_count = 0
                    # Clear status only if we are not about to show a result from the processing
                    if not transcript: # If transcript was empty, no rerun happened yet
                        status_indicator_voice.empty()

        elif ctx.state.playing: # No new frames, but streamer is active (listening)
            status_indicator_voice.info("Listening...")
        else: # Not playing, probably stopped by user or not started
            status_indicator_voice.empty()

    except av.error.TimeoutError: # Normal if no audio is coming in
        if ctx.state.playing: # Only show listening if it's supposed to be active
             status_indicator_voice.info("Listening...")
        else:
             status_indicator_voice.empty()
        pass 
    except Exception as e_outer:
        if ctx and ctx.state.playing and ctx.audio_receiver and not ctx.audio_receiver.is_closed:
            st.warning(f"An issue occurred with the audio stream: {e_outer}. Try restarting the voice agent.")
        logger.error(f"Outer error in voice agent section: {e_outer}", exc_info=True)
        status_indicator_voice.empty()

elif ctx and not ctx.audio_receiver: # Streamer context exists but no audio receiver (e.g., mic permission denied after start)
    if st.session_state.audio_buffer: # Clear any leftover buffer
        logger.info("Audio receiver became unavailable. Clearing buffer.")
        st.session_state.audio_buffer = b""
        st.session_state.audio_frame_count = 0
    status_indicator_voice.caption("Voice agent is not receiving audio. Check microphone permissions or restart.")

# else: ctx is None (failed to initialize) or not ctx.audio_receiver - error/message handled at component init
Key changes in this corrected voice agent block:
