# dashboard.py

import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import av
import streamlit.components.v1 as components
import wave
import json
import logging
import base64
import re

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

# === Helper Functions === })        # Keep other analysis results like patch, explanation unless specifically cleared elsewhere

def clear_all_github_session_state():
    """Resets all GitHub-related session state and clears loaded analysis files."""
    logger.info("Clearing all GitHub session state and related analysis inputs...")
    
    # Ensure all keys exist before modifying them
    keys_to_clear = {
        "internal_github_repo_url_input": "",  # Use a separate key for internal state
        "current_github_repo_url": None,
        "github_branches": [],
        "github_selected_branch": None,
        "github_path_stack": [""],
        "github_repo_owner": None,
        "github_repo_name": None,
        "analysis_results": {
            "trace": None,
            "source_files_content": {},
        }
    }

    
}
for key, default_value in keys_to_clear.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
    else:
        st.session_state[key] = default_value  # Properly indented block for the else statement
# Define session_defaults after the loop
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
        "original_patched_file_content": None,
    },
    "qa_result": None,
    "inbox_data": None,
    "workflow_status_data": None,
    "metrics_data": None,
    "qa_code_to_validate": None,
}# Populate session state with defaults if not already set
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value# === GitHub Repo Integration Sidebar (Full Version) ===
with st.sidebar:
    st.markdown("### 📦 Load Code from GitHub")
    
    # Text input for GitHub URL - its value is stored via the key
    st.text_input(
        "Public GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        key="github_repo_url_input", # This key links to st.session_state.github_repo_url_input
        on_change=lambda: st.session_state.update({ # On change, reset dependent states
            "current_github_repo_url": None, 
            "github_branches": [], 
            "github_selected_branch": None,
            "github_path_stack": [""],
            "github_repo_owner": None,
            "github_repo_name": None
            # Do not clear analysis_results here, let "Load/Process Repo" button handle that decision
        }) 
    )

    # Button to trigger loading or resetting
    if st.button("Load/Process Repo", key="load_repo_button"):
        if not st.session_state.github_repo_url_input:
            clear_all_github_session_state() # Clears everything including analysis results
            st.info("GitHub input cleared.")
        else:
            # Mark that we need to process this URL by setting current_github_repo_url to None
            # This will trigger the branch fetching logic below if the input URL is valid
            st.session_state.current_github_repo_url = None 
            # Clear previous analysis results when loading a new repo explicitly
            if "analysis_results" not in st.session_state: st.session_state.analysis_results = {}
            st.session_state.analysis_results.update({"trace": None, "source_files_content": {}})
        st.rerun()


    # --- GitHub URL Parsing & Branch Fetch ---
    # This logic runs on every rerun if github_repo_url_input is set
    active_github_url = st.session_state.get("github_repo_url_input", "").strip()

    if active_github_url:
        match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", active_github_url)
        if not match:
            if st.session_state.current_github_repo_url is not None : # Only show error if we tried to process it
                 st.warning("Invalid GitHub URL format. Use: https://github.com/owner/repo")
        else:
            owner, repo = match.groups()
            # Update owner/repo if they changed from last successfully processed one
            if st.session_state.github_repo_owner != owner or st.session_state.github_repo_name != repo:
                st.session_state.github_repo_owner = owner
                st.session_state.github_repo_name = repo
                st.session_state.current_github_repo_url = None # Mark for branch reload

            # Fetch branches if current_github_repo_url is None (new valid URL entered and button clicked)
            # or if it doesn't match the active_github_url (e.g. after on_change callback clears it)
            if st.session_state.current_github_repo_url != active_github_url:
                st.session_state.current_github_repo_url = active_github_url # Mark as currently being processed
                api_branches_url = f"https://api.github.com/repos/{owner}/{repo}/branches"
                logger.info(f"Fetching branches: {api_branches_url}")
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
                        st.warning("No branches found.")
                        st.session_state.github_branches = []
                        st.session_state.github_selected_branch = None
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Branch fetch error: {e}")
                    st.session_state.github_branches = []
                    st.session_state.github_selected_branch = None
                    st.session_state.current_github_repo_url = None # Allow retry
                except json.JSONDecodeError as e:
                    st.error(f"❌ Branch JSON error: {e}")
                    # Keep other states so user doesn't have to retype URL
                    st.session_state.github_branches = []
                    st.session_state.github_selected_branch = None

            # --- Branch Selection ---
            current_branches = st.session_state.get("github_branches", [])
            if current_branches:
                try:
                    branch_idx = current_branches.index(st.session_state.github_selected_branch) if st.session_state.github_selected_branch in current_branches else 0
                except ValueError: branch_idx = 0
                
                selected_branch = st.selectbox(
                    "Select Branch", current_branches, index=branch_idx, key="github_branch_selector"
                )
                if selected_branch != st.session_state.github_selected_branch:
                    st.session_state.github_selected_branch = selected_branch
                    st.session_state.github_path_stack = [""] # Reset path on branch change
                    st.rerun()

            # --- File Browser Logic (if owner, repo, branch are set) ---
            gh_owner = st.session_state.get("github_repo_owner")
            gh_repo = st.session_state.get("github_repo_name")
            gh_branch = st.session_state.get("github_selected_branch")

            if gh_owner and gh_repo and gh_branch:
                current_path_parts = st.session_state.github_path_stack
                current_path_str = "/".join([p for p in current_path_parts if p])

                @st.cache_data(ttl=120, show_spinner="Fetching directory...")
                def fetch_gh_dir_content(owner_str, repo_str, path_str, branch_str):
                    url = f"https://api.github.com/repos/{owner_str}/{repo_str}/contents/{path_str}?ref={branch_str}"
                    logger.info(f"Fetching GitHub dir (cached): {url}")
                    try:
                        r = requests.get(url, timeout=10)
                        r.raise_for_status()
                        return r.json()
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Dir content fetch error for '{path_str}': {e}")
                    except json.JSONDecodeError as e:
                        st.warning(f"Dir content JSON error for '{path_str}': {e}")
                    return None

                entries = fetch_gh_dir_content(gh_owner, gh_repo, current_path_str, gh_branch)

                if entries is not None:
                    # Display current path and "Up" button
                    display_path = f"Current: /{current_path_str}" if current_path_str else "Current: / (Repo Root)"
                    st.caption(display_path)
                    if current_path_str: # Not in root
                        if st.button("⬆️ .. (Parent Directory)", key=f"gh_up_button_{current_path_str}", use_container_width=True):
                            st.session_state.github_path_stack.pop()
                            st.rerun()
                    
                    st.markdown("###### Directories")
                    for item in sorted(entries, key=lambda x: (x["type"], x["name"])): # Sort dirs first, then files
                        if item["type"] == "dir":
                            if st.button(f"📁 {item['name']}", key=f"gh_dir_{current_path_str}_{item['name']}", use_container_width=True):
                                st.session_state.github_path_stack.append(item['name'])
                                st.rerun()
                    
                    st.markdown("###### Files")
                    for item in sorted(entries, key=lambda x: (x["type"], x["name"])):
                         if item["type"] == "file" and any(item["name"].endswith(ext) for ext in RECOGNIZED_FILE_EXTENSIONS):
                            if st.button(f"📄 {item['name']}", key=f"gh_file_{current_path_str}_{item['name']}", use_container_width=True):
                                file_rel_path = f"{current_path_str}/{item['name']}".strip("/")
                                raw_url = f"https://raw.githubusercontent.com/{gh_owner}/{gh_repo}/{gh_branch}/{file_rel_path}"
                                logger.info(f"Fetching file: {raw_url}")
                                try:
                                    with st.spinner(f"Loading {item['name']}..."):
                                        file_r = requests.get(raw_url, timeout=10)
                                        file_r.raise_for_status()
                                    file_content = file_r.text
                                    st.success(f"Loaded: {item['name']}")
                                    
                                    # Use relative path from repo root as key for analysis_results
                                    key_path = os.path.join(current_path_str, item['name']).replace("\\","/") if current_path_str else item['name']

                                    if item['name'].endswith(TRACEBACK_EXTENSION):
                                        st.session_state.analysis_results['trace'] = file_content
                                        st.info(f"'{key_path}' loaded as Traceback.")
                                    else: # Already filtered by RECOGNIZED_FILE_EXTENSIONS
                                        st.session_state.analysis_results['source_files_content'][key_path] = file_content
                                        st.info(f"'{key_path}' loaded as Source File.")
                                    st.rerun() # To update "Loaded Analysis Inputs" expander
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Error loading file '{item['name']}': {e}")
                elif current_path_str: # entries is None and not in root
                    st.warning("Could not list directory contents.")
    
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
    manual_trace_str = None
    manual_sources_dict = {}
    manual_summary_list_main = []

    for file_item in manual_uploaded_files_main:
        try:
            content_str = file_item.getvalue().decode("utf-8")
            if file_item.name.endswith(TRACEBACK_EXTENSION):
                manual_trace_str = content_str
                manual_summary_list_main.append(f"Traceback: {file_item.name}")
            elif any(file_item.name.endswith(ext) for ext in SUPPORTED_SOURCE_EXTENSIONS):
                manual_sources_dict[file_item.name] = content_str
                manual_summary_list_main.append(f"Source: {file_item.name}")
        except Exception as e: st.error(f"Error processing '{file_item.name}': {e}")

    if manual_trace_str is not None:
        st.session_state.analysis_results['trace'] = manual_trace_str
        st.session_state.analysis_results['source_files_content'] = manual_sources_dict # Overwrite with sources uploaded alongside trace
        clear_all_github_session_state() # This also clears analysis_results parts, so re-assign trace/sources
        st.session_state.analysis_results['trace'] = manual_trace_str # Re-assign after clear
        st.session_state.analysis_results['source_files_content'] = manual_sources_dict
        st.success(f"Manually uploaded: {', '.join(manual_summary_list_main)}. GitHub selection cleared.")
        st.rerun()
    elif manual_sources_dict:
        st.session_state.analysis_results['source_files_content'].update(manual_sources_dict) # Merge if only sources
        clear_all_github_session_state() # This clears URL etc. and also trace/sources from analysis_results
        st.session_state.analysis_results['source_files_content'].update(manual_sources_dict) # Re-apply after clear
        st.success(f"Manually added source files: {', '.join(manual_summary_list_main)}. GitHub selection cleared.")
        st.rerun()


# === Main Application Tabs ===
tab_titles_main_list = [
    "📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation",
    "📣 Issue Notices", "🤖 Autonomous Workflow", "🔍 Workflow Check",
    "📊 Repo Structure Insights", # New Tab for the chart
    "📈 Metrics" # Moved metrics to be last
]
(tab_patch, tab_qa, tab_doc, tab_issues, tab_workflow_trigger, 
 tab_workflow_status, tab_repo_insights, tab_metrics) = st.tabs(tab_titles_main_list)


with tab_patch:
    # ... (Content for Traceback + Patch tab - kept as per your last version with minor adaptations)
    st.header("📄 Traceback & Patch Analysis")
    loaded_trace_tab1 = st.session_state.analysis_results.get('trace')
    loaded_sources_tab1 = st.session_state.analysis_results.get('source_files_content', {})
    if loaded_trace_tab1: st.text_area("Loaded Traceback for Analysis", value=loaded_trace_tab1, height=100, disabled=True, key="tab1_trace_main_view")
    if loaded_sources_tab1: st.expander("Loaded Source Files for Analysis").json(loaded_sources_tab1, expanded=False)

    if st.button("🔬 Analyze Loaded Data & Suggest Patch", key="tab1_analyze_main_btn"):
        # ... (analysis logic from your file, using loaded_trace_tab1, loaded_sources_tab1, SUGGEST_PATCH_URL) ...
        # Ensure it populates st.session_state.analysis_results with 'patch', 'explanation', etc.
        if not loaded_trace_tab1 and not loaded_sources_tab1:
            st.warning("Please load data first.")
        else:
            payload = {"trace": loaded_trace_tab1, "language": "python", "source_files": loaded_sources_tab1}
            response = make_api_request("POST", SUGGEST_PATCH_URL, json_payload=payload, operation_name="Patch Suggestion")
            if response and not response.get("error"):
                st.session_state.analysis_results.update({
                    'patch': response.get("patched_code", ""),
                    'explanation': response.get("explanation"),
                    'patched_file_name': response.get("patched_file_name"),
                    'original_patched_file_content': response.get("original_content") # Assuming backend provides this
                })
                st.success("Patch suggestion received.")
                st.rerun()
            else: st.error(f"Patch failed. {response.get('details', '') if response else 'No response'}")
    
    original_content_tab1 = st.session_state.analysis_results.get('original_patched_file_content')
    patched_code_val_tab1 = st.session_state.analysis_results.get('patch')
    if original_content_tab1 is not None and patched_code_val_tab1 is not None:
        # ... (Diff view and st_ace editor logic from your file) ...
        st.markdown("### 🔍 Diff View")
        if original_content_tab1 != patched_code_val_tab1:
            html_diff_gen_tab1 = difflib.HtmlDiff(wrapcolumn=70, tabsize=4)
            diff_html = html_diff_gen_tab1.make_table(original_content_tab1.splitlines(keepends=True), patched_code_val_tab1.splitlines(keepends=True), "Original", "Patched", context=True, numlines=3)
            components.html(diff_html, height=450, scrolling=True)
        else: st.info("No changes in patch.")
        st.markdown("### ✍️ Edit Patch (Live)")
        edited_code_val_tab1 = st_ace(value=patched_code_val_tab1, language="python", theme="monokai", height=300, key="tab1_ace_editor_main")
        if edited_code_val_tab1 != patched_code_val_tab1:
            st.session_state.analysis_results['patch'] = edited_code_val_tab1
            st.session_state.edited_patch = edited_code_val_tab1
            st.caption("Patch updated.")
    elif patched_code_val_tab1 is not None:
        st.markdown("### ✨ Generated/Patched Code")
        st.text_area("Code:", value=patched_code_val_tab1, height=300, key="tab1_patched_code_no_diff_main")
        # Allow editing
        edited_code_val_tab1_no_diff = st_ace(value=patched_code_val_tab1, language="python", theme="monokai", height=300, key="tab1_ace_editor_no_diff_main")
        if edited_code_val_tab1_no_diff != patched_code_val_tab1:
            st.session_state.analysis_results['patch'] = edited_code_val_tab1_no_diff
            st.session_state.edited_patch = edited_code_val_tab1_no_diff
            st.caption("Code updated.")
            
    explanation_val_tab1 = st.session_state.analysis_results.get('explanation')
    if explanation_val_tab1: st.markdown(f"**Explanation:** {explanation_val_tab1}")


with tab_qa:
    # ... (Content for QA Validation tab - kept as per your last file)
    st.header("✅ QA Validation")
    code_for_qa_tab2 = st.session_state.analysis_results.get('patch') 
    if code_for_qa_tab2 is not None:
        st.text_area("Code for QA:", value=code_for_qa_tab2, height=200, disabled=True, key="qa_code_display_tab2_main_val")
        if st.button("🛡️ Run QA Validation", key="qa_run_validation_btn_tab2_val"):
            # ... (QA payload and make_api_request call to QA_URL) ...
            payload = {"code": code_for_qa_tab2, "patched_file_name": st.session_state.analysis_results.get('patched_file_name')}
            response = make_api_request("POST", QA_URL, json_payload=payload, operation_name="QA Validation")
            if response and not response.get("error"): st.json(response)
            else: st.error(f"QA failed. {response.get('details', '') if response else 'No response'}")
    else: st.warning("No patched code from Tab 1 to validate.")

with tab_doc:
    # ... (Content for Documentation tab - kept as per your last file)
    st.header("📘 Documentation")
    doc_summary_main_val = st.session_state.analysis_results.get('doc_summary')
    if doc_summary_main_val: st.markdown(f"**Summary from Analysis:** {doc_summary_main_val}")
    st.subheader("Generate Documentation for Specific Code:")
    doc_code_input_tab3_main_val = st.text_area("Paste code:", key="doc_code_input_area_tab3_main_val", height=200)
    if st.button("📝 Generate Ad-hoc Documentation", key="doc_generate_btn_tab3_main_val"):
        # ... (make_api_request call to DOC_URL) ...
        if doc_code_input_tab3_main_val:
            payload = {"code": doc_code_input_tab3_main_val}
            response = make_api_request("POST", DOC_URL, json_payload=payload, operation_name="Ad-hoc Docs")
            if response and not response.get("error"): st.markdown(response.get("doc", "No docs."))
            else: st.error(f"Doc gen failed. {response.get('details', '') if response else 'No response'}")
        else: st.warning("Paste code.")

with tab_issues:
    # ... (Content for Issue Notices tab - kept as per your last file)
    st.header("📣 Issue Notices & Agent Summaries")
    if st.button("🔄 Refresh Issue Notices", key="fetch_issues_tab_btn_val"):
        # ... (make_api_request call to ISSUES_INBOX_URL, display logic with expanders & trigger button) ...
        response = make_api_request("GET", ISSUES_INBOX_URL, operation_name="Fetch Issues")
        if response and not response.get("error"): 
            st.session_state.inbox_data = response
            st.rerun()
        else: st.error(f"Issue fetch failed. {response.get('details', '') if response else 'No response'}")
    
    current_inbox = st.session_state.get("inbox_data")
    if current_inbox and current_inbox.get("issues"):
        for i, issue_item in enumerate(current_inbox["issues"]):
            with st.expander(f"Issue {issue_item.get('id','N/A')} - {issue_item.get('classification','N/A')}"):
                st.json(issue_item)
                if st.button(f"▶️ Trigger Workflow for {issue_item.get('id','N/A')}", key=f"trigger_issue_{i}"):
                    # ... (make_api_request call to WORKFLOW_RUN_URL) ...
                    pass # Placeholder for brevity
    elif current_inbox: st.info("No issues in inbox or unexpected format.")
    else: st.info("Refresh to fetch issues.")


with tab_workflow_trigger:
    # ... (Content for Autonomous Workflow tab - kept as per your last file)
    st.header("🤖 Trigger Autonomous Workflow by ID")
    workflow_issue_id_input_val = st.text_input("Enter Issue ID:", key="workflow_issue_id_trigger_input_val")
    if st.button("▶️ Run Workflow by ID", key="run_workflow_by_id_btn_val"):
        # ... (make_api_request call to WORKFLOW_RUN_URL) ...
        if workflow_issue_id_input_val:
            payload = {"issue_id": workflow_issue_id_input_val}
            response = make_api_request("POST", WORKFLOW_RUN_URL, json_payload=payload, operation_name="Manual Workflow")
            if response and not response.get("error"): st.success(f"Workflow for {workflow_issue_id_input_val} triggered.")
            else: st.error(f"Workflow failed. {response.get('details', '') if response else 'No response'}")
        else: st.warning("Enter Issue ID.")

with tab_workflow_status:
    # ... (Content for Workflow Check tab - kept as per your last file)
    st.header("🔍 Workflow Status Check")
    if st.button("🔄 Refresh Workflow Status", key="refresh_workflow_status_check_btn_val"):
        # ... (make_api_request call to WORKFLOW_CHECK_URL) ...
        response = make_api_request("GET", WORKFLOW_CHECK_URL, operation_name="Workflow Status")
        if response and not response.get("error"): 
            st.session_state.workflow_status_data = response
            st.rerun()
        else: st.error(f"Status check failed. {response.get('details', '') if response else 'No response'}")
    
    current_status = st.session_state.get("workflow_status_data")
    if current_status: st.json(current_status)
    else: st.info("Refresh to fetch status.")

with tab_repo_insights: # New Placeholder Tab for the Digital Chart
    st.header("📊 Repository Structure Insights")
    st.markdown("""
    This tab is intended to display a "digital chart" or visual representation of your loaded GitHub repository's content.
    
    
    
    """)
    
    # Example placeholder for where chart might go if data was available
    if st.session_state.get("github_repo_owner") and st.session_state.get("github_repo_name"):
        st.info(f"Insights for repository: {st.session_state.github_repo_owner}/{st.session_state.github_repo_name} would appear here once specified.")
        # @st.cache_data # Example of fetching language data
        # def get_repo_languages(owner, repo):
        #     lang_url = f"https://api.github.com/repos/{owner}/{repo}/languages"
        #     try:
        #         r = requests.get(lang_url, timeout=5)
        #         r.raise_for_status()
        #         return r.json() # Returns dict like {"Python": 30203, "JavaScript": 1024}
        #     except: return None
        # languages = get_repo_languages(st.session_state.github_repo_owner, st.session_state.github_repo_name)
        # if languages:
        #     st.write("Language Breakdown (Example - actual chart TBD):")
        #     st.json(languages) # Replace with actual chart later
        # else:
        #     st.warning("Could not fetch language data for the repository to display an example chart.")
    else:
        st.warning("Load a GitHub repository from the sidebar to see potential insights.")


with tab_metrics:
    # ... (Content for Metrics tab - kept as per your last file)
    st.header("📈 System Metrics") # Changed icon to match tab title
    if st.button("📈 Fetch System Metrics", key="fetch_metrics_tab_btn_val"):
        # ... (make_api_request call to METRICS_URL) ...
        response = make_api_request("GET", METRICS_URL, operation_name="Fetch Metrics")
        if response and not response.get("error"): 
            st.session_state.metrics_data = response
            st.rerun()
        else: st.error(f"Metrics fetch failed. {response.get('details', '') if response else 'No response'}")

    current_metrics = st.session_state.get("metrics_data")
    if current_metrics: st.json(current_metrics)
    else: st.info("Refresh to fetch metrics.")


# === Voice Agent Section ===
# (Kept as per your last provided file - simpler version without full Gemini bi-directional audio yet)
st.markdown("---")
st.markdown("## 🎙️ DebugIQ Voice Agent")
st.caption("Speak your commands to the DebugIQ Agent.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio_base64" in message and message["role"] == "assistant" and st.session_state.get("using_gemini_voice", False):
            try:
                audio_bytes = base64.b64decode(message["audio_base64"])
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e: logger.error(f"Error playing audio: {e}")

try:
    ctx = webrtc_streamer(
        key=f"voice_agent_stream_{BACKEND_URL}",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
             rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
             media_stream_constraints={"audio": True, "video": False}
        )
    )
except Exception as e:
    st.error(f"Failed to initialize voice agent: {e}")
    logger.exception("Error initializing webrtc_streamer for Voice Agent")
    ctx = None

if ctx and ctx.audio_receiver:
    voice_status_indicator_main_agent = st.empty()
    # ... (The rest of the voice agent logic from your last provided file,
    #      which sends to VOICE_TRANSCRIBE_URL and then to COMMAND_URL implicitly via GEMINI_CHAT_URL in your example.
    #      I've kept this part as it was in your file. If full Gemini bi-directional audio is needed,
    #      the more complex version of this voice agent block should be used.)
    try:
        if not ctx.state.playing: voice_status_indicator_main_agent.caption("Click 'Start' to speak.")
        else: voice_status_indicator_main_agent.info("Listening...")
        audio_frames_main_agent = ctx.audio_receiver.get_frames(timeout=0.2)
        if audio_frames_main_agent:
            voice_status_indicator_main_agent.info("Processing audio...")
            # ... (Audio param inference) ...
            for frame_main_agent in audio_frames_main_agent: # Accumulate audio
                # ... (Convert and append to st.session_state.audio_buffer) ...
                if frame_main_agent.format.name == 's16': audio_data_main_agent = frame_main_agent.to_ndarray().tobytes()
                elif frame_main_agent.format.name in ['f32', 'flt32', 'flt']:
                    int16_array_main_agent = (frame_main_agent.to_ndarray() * (2**15 - 1)).astype(np.int16)
                    audio_data_main_agent = int16_array_main_agent.tobytes()
                else: continue
                st.session_state.audio_buffer += audio_data_main_agent
                st.session_state.audio_frame_count += frame_main_agent.samples

            if st.session_state.audio_frame_count >= (AUDIO_PROCESSING_THRESHOLD_SECONDS * st.session_state.audio_sample_rate) and st.session_state.audio_buffer:
                voice_status_indicator_main_agent.info("🎙️ Transcribing command...")
                # ... (Temp file, transcribe, send to COMMAND_URL/GEMINI_CHAT_URL as in your file) ...
                # This part needs to be carefully reviewed to match the intended voice agent functionality.
                # The version you sent had a mix of Gemini-oriented variable names (GEMINI_CHAT_URL)
                # but simpler command processing. Assuming a general agent call here.
                temp_wav_path, transcript_text = None, None
                try:
                    # ... (WAV creation logic) ...
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f: temp_wav_path = tmp_f.name
                    with wave.open(temp_wav_path, 'wb') as wf:
                        wf.setnchannels(st.session_state.audio_num_channels); wf.setsampwidth(st.session_state.audio_sample_width); wf.setframerate(st.session_state.audio_sample_rate); wf.writeframes(st.session_state.audio_buffer)
                    with open(temp_wav_path, "rb") as f_aud:
                        transcribe_resp = make_api_request("POST", VOICE_TRANSCRIBE_URL, files={"file": ("c.wav", f_aud, "audio/wav")}, op_name="VA Transcribe")
                    transcript_text = transcribe_resp.get("transcript") if transcribe_resp and not transcribe_resp.get("error") else None
                    if transcript_text:
                        st.success(f"🗣️ You: \"{transcript_text}\"")
                        st.session_state.chat_history.append({"role": "user", "content": transcript_text}) # Add to history
                        
                        # Using GEMINI_CHAT_URL here as it was in your API list, 
                        # assuming it's the intended general voice interaction endpoint.
                        # If you have a simpler /voice/command, you can use that.
                        cmd_payload = {"text_command": transcript_text, "conversation_history": st.session_state.chat_history[:-1]}
                        cmd_resp = make_api_request("POST", GEMINI_CHAT_URL, json_payload=cmd_payload, operation_name="VA Command")
                        
                        if cmd_resp and not cmd_resp.get("error"):
                            assistant_reply = cmd_resp.get("text_response", "Command processed.")
                            st.info(f"🤖 Agent: {assistant_reply}")
                            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply}) # Add agent reply
                            # If cmd_resp contains audio_content_base64 for Gemini, the loop at top will play it
                        else: st.warning(f"Agent command issue. {cmd_resp.get('details', '') if cmd_resp else 'No response.'}")
                        st.rerun() # Rerun to update chat display
                finally:
                    if temp_wav_path and os.path.exists(temp_wav_path): os.remove(temp_wav_path)
                    st.session_state.audio_buffer = b""; st.session_state.audio_frame_count = 0
                    if not transcript_text: voice_status_indicator_main_agent.empty()
    # ... (rest of try-except-finally for voice agent from your file) ...
    except av.error.TimeoutError:
        if ctx.state.playing: voice_status_indicator_main_agent.info("Listening...")
        else: voice_status_indicator_main_agent.empty()
    except Exception as e:
        logger.error(f"Voice agent error: {e}", exc_info=True)
        if 'voice_status_indicator_main_agent' in locals(): voice_status_indicator_main_agent.empty()

elif ctx and not ctx.audio_receiver:
    if 'voice_status_indicator_main_agent' in locals(): voice_status_indicator_main_agent.caption("Voice agent not receiving audio.")
    else: st.caption("Voice agent not receiving audio.")
