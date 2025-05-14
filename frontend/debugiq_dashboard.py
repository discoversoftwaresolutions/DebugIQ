# File: frontend/debugiq_dashboard.py (Updated to match v2 logic)

# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# Updated imports for Voice Agent section (if you intend to keep that)
# import av # Required for processing audio frames from streamlit-webrtc
# import numpy as np # Required for processing audio frames
# import io # Required for in-memory WAV file creation
# import wave # Required for WAV file creation
# from streamlit_webrtc import webrtc_streamer, WebRtcMode # Also needed if keeping voice
# import logging # Already imported by requests, but good to be explicit
# import base64 # Needed for voice/image encoding
# import re # Needed for GitHub URL parsing
# import threading # Potentially needed for thread-safe buffer if issues arise

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
# Use environment variable for the backend URL, with a fallback
# Set BACKEND_URL environment variable in Railway frontend settings
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app") # <-- Ensure this fallback is correct or use env var

# Define API endpoint paths relative to BACKEND_URL

ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",  # Correct path for analyze.py endpoint
    "qa_validation": "/qa/run",  # Based on /qa prefix and @router.post("/run")
    "doc_generation": "/doc/generate",  # Based on /doc prefix and @router.post("/generate")
    "issues_inbox": "/issues/attention-needed",  # Based on no prefix and @router.get("/issues/attention-needed")
    "workflow_run": "/workflow/run_autonomous_workflow",  # Based on /workflow prefix and @router.post("/run_autonomous_workflow")
    # Workflow status needs issue_id formatting
    "workflow_status": "/issues/{issue_id}/status",  # Based on no prefix and @router.get("/issues/{issue_id}/status")
    "system_metrics": "/metrics/status",  # Based on no prefix and @router.get("/metrics/status")
    # Paths for Voice/Gemini - CONFIRM THESE WITH YOUR BACKEND ROUTERS
    "voice_transcribe": "/voice/transcribe",  # Example path - CHECK YOUR BACKEND
    "gemini_chat": "/gemini/chat",  # Example path - CHECK YOUR BACKEND
    "tts": "/voice/tts"  # Example path - CHECK YOUR BACKEND
}

# === Helper Functions ===
# Modified make_api_request to construct the full URL from BACKEND_URL and path
def make_api_request(method, endpoint_key, payload=None, return_json=True): # Takes endpoint_key, not full url
    """Makes an API request to the backend."""
    # Ensure endpoint_key exists in ENDPOINTS
    if endpoint_key not in ENDPOINTS:
        logger.error(f"Invalid endpoint key: {endpoint_key}")
        return {"error": f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'."}

    # Handle endpoints that require formatting (like workflow_status)
    path_template = ENDPOINTS[endpoint_key]

    # --- Construct the path ---
    # This needs to be smarter if other endpoints require formatting.
    # For now, handle workflow_status specifically.
    if endpoint_key == "workflow_status":
         issue_id = st.session_state.get("active_issue_id")
         if not issue_id:
              logger.error("Workflow status requested but no active_issue_id in session state.")
              # Indicate that polling should stop if no issue_id
              st.session_state.workflow_completed = True
              return {"error": "No active issue ID to check workflow status."}
         try:
            # Format the path using the issue_id
            path = path_template.format(issue_id=issue_id)
         except KeyError as e:
             logger.error(f"Failed to format workflow_status path: Missing key {e}")
             st.session_state.workflow_completed = True
             return {"error": f"Internal error formatting workflow status URL: Missing issue ID key."}
    else:
         # For all other endpoints, the path is static from ENDPOINTS
         path = path_template

    # Construct the full URL by joining BACKEND_URL and the path
    # Use urljoin for robust joining, especially if BACKEND_URL might or might not end with /
    # Requires 'from urllib.parse import urljoin'
    from urllib.parse import urljoin # <--- ADD THIS IMPORT AT THE TOP
    url = urljoin(BACKEND_URL, path) # <--- CONSTRUCT THE FULL URL HERE

    try:
        logger.info(f"Making API request: {method} {url}")
        # Set a reasonable timeout for API calls
        response = requests.request(method, url, json=payload, timeout=60) # Pass the constructed url
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"API request successful: {method} {url}")
        if return_json:
            return response.json()
        else:
            return response.content # Return raw content for binary data like audio
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out: {method} {url}")
        return {"error": "API request timed out. The backend might be slow or unresponsive."}
    except requests.exceptions.ConnectionError:
        logger.error(f"API connection error: {method} {url}")
        return {"error": "Could not connect to the backend API. Please check the backend URL and status."}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        # Try to include backend error detail if available
        detail = str(e)
        backend_detail = "N/A"
        if e.response is not None:
            detail = f"Status {e.response.status_code}"
            try:
                 # Attempt to parse JSON detail from backend
                 backend_json = e.response.json()
                 backend_detail = backend_json.get('detail', backend_json)
                 detail = f"Status {e.response.status_code} - {backend_detail}" # More informative error

            except json.JSONDecodeError:
                 # If not JSON, include raw text
                 backend_detail = e.response.text
                 detail = f"Status {e.response.status_code} - Response Text: {backend_detail}"
            except Exception as json_e:
                 logger.warning(f"Could not parse backend error response as JSON: {json_e}")


        return {"error": f"API request failed: {detail}", "backend_detail": backend_detail}


# === Audio Processing Helper ===
# Include frames_to_wav_bytes function here if keeping Voice Agent

# === WebRTC Audio Frame Callback ===
# Include audio_frame_callback here if keeping Voice Agent

# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

# Initialize session state for recording and chat history (if keeping Voice Agent)
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
# Removed Voice Agent tab as it's now a dedicated section
# Ensure tab count and names match your desired UI
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Check", "📈 Metrics"])
# Assign tabs to variables based on their index
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics = tabs # Corrected variable names to match tab count

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    # Updated file uploader types based on backend model
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"])

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        # Display original code - using st.code is better for syntax highlighting
        st.subheader("Original Code")
        # Attempt to guess language from file type
        original_language = uploaded_file.type.split('/')[-1] if uploaded_file.type in ["py", "java", "js", "cpp", "c"] else "plaintext"
        st.code(file_content, language=original_language, height=300)

        if st.button("🔬 Analyze & Suggest Patch"):
            with st.spinner("Analyzing and suggesting patch..."):
                # Corrected Payload to match backend AnalyzeRequest model
                # Requires 'code', 'language', (optional 'context')
                payload = {
                    "code": file_content,
                    "language": original_language, # Use detected language
                    # "context": {} # <--- Add optional context if needed by backend
                }
                # Use make_api_request with the endpoint key
                response = make_api_request("POST", "suggest_patch", payload) # <--- Use endpoint key

            if "error" not in response:
                # Assumes backend returns 'diff' and 'explanation' keys
                suggested_diff = response.get("diff", "No diff suggested.")
                explanation = response.get("explanation", "No explanation provided.")

                st.subheader("Suggested Patch")
                # Display diff with diff language highlighting
                st.code(suggested_diff, language="diff", height=300)
                st.markdown(f"💡 **Explanation:** {explanation}")


                # Code Editor for Editing Patch
                st.markdown("### ✍️ Edit Suggested Patch")
                edited_patch = st_ace(
                    value=suggested_diff, # Start with the suggested diff
                    language=original_language, # Use original language for editing
                    theme="monokai",
                    height=350,
                    key="ace_editor_patch" # Unique key
                )

                # Diff View (Original vs. Edited Patch)
                st.markdown("### 🔍 Diff View (Original vs. Edited Patch)")
                # Check if both original code and edited patch are available
                if edited_patch is not None and file_content is not None:
                    # Generate diff as HTML
                    diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
                        fromlines=file_content.splitlines(),
                        tolines=edited_patch.splitlines(),
                        fromdesc="Original Code",
                        todesc="Edited Patch",
                        context=True
                    )
                    # Display HTML in Streamlit
                    st.components.v1.html(diff_html, height=400, scrolling=True) # Adjusted height
                else:
                    st.info("Upload original code and generate/edit patch to see diff.")

            else:
                # Display error from make_api_request helper
                st.error(response["error"])
                # Optionally display backend detail if available
                if "backend_detail" in response and response["backend_detail"] not in ["N/A", "", response["error"].split(" - ", 1)[-1]]:
                     st.json({"Backend Detail": response["backend_detail"]})


# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    st.write("Upload a patch file to run QA validation checks.")
    uploaded_patch = st.file_uploader("Upload Patch File", type=["txt", "diff", "patch"]) # Added diff/patch types

    if uploaded_patch:
        patch_content = uploaded_patch.read().decode("utf-8")
        st.subheader("Patch Content")
        st.code(patch_content, language="diff", height=200)

        if st.button("🛡️ Validate Patch"):
            if uploaded_patch:
                with st.spinner("Running QA validation..."):
                    # Check your backend's expected payload for /qa/run
                    payload = {"patch_diff": patch_content} # Assuming backend expects 'patch_diff'
                    response = make_api_request("POST", "qa_validation", payload) # <--- Use endpoint key

            if "error" not in response:
                st.subheader("Validation Results")
                # Display validation results - check backend response format
                st.json(response) # Assuming response is a JSON summary/report
            else:
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please upload a patch file first.")


# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    st.write("Upload a code file to generate documentation automatically.")
    uploaded_code = st.file_uploader("Upload Code File for Documentation", type=["txt", "py", "java", "js", "cpp", "c"])

    if uploaded_code:
        code_content = uploaded_code.read().decode("utf-8")
        st.subheader("Code Content")
        doc_language = uploaded_code.type.split('/')[-1] if uploaded_code.type in ["py", "java", "js", "cpp", "c"] else "plaintext"
        st.code(code_content, language=doc_language, height=200)

    if st.button("📝 Generate Documentation"):
        if uploaded_code:
            with st.spinner("Generating documentation..."):
                # Check your backend's expected payload for /doc/generate
                payload = {"code": code_content, "language": doc_language} # Assuming backend needs code and language
                response = make_api_request("POST", "doc_generation", payload) # <--- Use endpoint key

            if "error" not in response:
                # Assuming the response contains a "documentation" key with markdown or text
                st.subheader("Generated Documentation")
                st.markdown(response.get("documentation", "No documentation generated."))
            else:
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please upload a code file first.")


# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues Inbox")
    st.write("This section lists issues needing attention from the autonomous workflow.")

    if st.button("🔄 Refresh Issues"):
        with st.spinner("Fetching issues..."):
            response = make_api_request("GET", "issues_inbox") # <--- Use endpoint key

        if "error" not in response:
            if response.get("issues"):
                st.subheader("Issues Needing Attention")
                # Display issues - check backend response format
                # Assuming backend returns {"issues": [{"id": ..., "status": ..., "error_message": ...}, ...]}
                for issue in response.get("issues", []):
                    with st.expander(f"Issue ID: {issue.get('id', 'N/A')} - Status: {issue.get('status', 'Unknown Status')}"):
                        st.write(f"**Error Details:** {issue.get('error_message', 'No error details provided.')}")
                        # You might add more issue details here if provided by the backend
            else:
                 st.info("No issues needing attention found.")
        else:
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


# === Workflow Tab (Trigger) ===
with tab_workflow:
    st.header("🤖 Autonomous Workflow Trigger")
    st.write("Trigger an autonomous workflow run for a specific issue.")
    issue_id = st.text_input("Issue ID to Trigger Workflow", placeholder="e.g., BUG-123", key="workflow_trigger_issue_id")

    if st.button("▶️ Trigger Workflow"):
        if issue_id:
            with st.spinner(f"Triggering workflow for issue {issue_id}..."):
                # Check backend payload for /workflow/run_autonomous_workflow
                payload = {"issue_id": issue_id} # Assuming backend expects 'issue_id'
                response = make_api_request("POST", "workflow_run", payload) # <--- Use endpoint key

            if "error" not in response:
                st.success(f"Workflow triggered successfully for Issue {issue_id}. Response: {response.get('message', 'No message.')}")
                # Store the issue_id to enable status polling in the next tab
                st.session_state.active_issue_id = issue_id
                st.session_state.workflow_completed = False # Reset completed state to start polling
            else:
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please enter an Issue ID.")

# === Workflow Check Tab (Status/Polling) ===
# Updated this tab name to better reflect its primary function (status/polling)
with tab_status: # This is now the status/polling tab
    st.header("🔍 Autonomous Workflow Status")
    # Add a text input to manually set the issue ID to poll
    issue_id_for_polling = st.text_input("Issue ID to check status (leave blank if triggered workflow above)", placeholder="e.g., BUG-123", key="workflow_status_issue_id")

    # If manually entered issue_id, use it for polling
    if issue_id_for_polling:
        st.session_state.active_issue_id = issue_id_for_polling
        st.session_state.workflow_completed = False # Assume not completed if manually set


    st.write("Checking status for Issue ID:", st.session_state.active_issue_id or "None (Trigger workflow or enter ID above)")

    # Progress tracker logic (kept as is)
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed", # This status name might need adjustment based on backend
        "📦 PR Created" # This status name might need adjustment based on backend
    ]
    progress_map = {
        "Seeded": 0, # Assuming Seeded is initial status
        "Fetching Details": 0,
        "Diagnosis in Progress": 1,
        "Diagnosis Complete": 1, # Add complete status to map
        "Patch Suggestion in Progress": 2,
        "Patch Suggestion Complete": 2, # Add complete status to map
        "Patch Validation in Progress": 3,
        "Patch Validated": 4,
        "PR Creation in Progress": 5, # Add PR creation status
        "PR Created - Awaiting Review/QA": 5 # Correct terminal status
    }
    terminal_status = "PR Created - Awaiting Review/QA"
    failed_status = "Workflow Failed"


    def show_agent_progress(status):
        step = progress_map.get(status, 0)
        st.progress((step + 1) / len(progress_labels))
        for i, label in enumerate(progress_labels):
            # Adjust icon logic: success up to current step, spinner for current, empty for future
            icon = "✅" if i <= step and status != failed_status else ("🔄" if i == step and status != failed_status else "⏳")
            st.markdown(f"{icon} {label}")

    # Polling logic (autorefresh only if an issue is active and workflow not complete)
    if st.session_state.active_issue_id and not st.session_state.workflow_completed:
        logger.info(f"Polling status for issue: {st.session_state.active_issue_id}")
        # Autorefresh triggers a rerun every X milliseconds
        st_autorefresh(interval=2000, key=f"workflow-refresh-{st.session_state.active_issue_id}") # Unique key per issue


    # --- Fetch and Display Status ---
    # This code runs on every rerun, including those triggered by autorefresh
    if st.session_state.active_issue_id:
        try:
            # Use make_api_request with the endpoint key - special handling in make_api_request for formatting
            status_response = make_api_request("GET", "workflow_status") # <--- Use endpoint key

            if "error" not in status_response:
                # Assumes backend returns {"status": "...", "error_message": "..."}
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")

                st.session_state.last_status = current_status
                st.info(f"🔁 Live Status: **{current_status}**")

                if error_message:
                    st.error(f"Workflow Error: {error_message}")

                show_agent_progress(current_status) # Update progress bar and labels

                # Check if workflow has reached a terminal state (completed or failed)
                if current_status == terminal_status or current_status == failed_status:
                    st.session_state.workflow_completed = True # Stop polling
                    if current_status == terminal_status:
                        st.success("✅ DebugIQ agents completed full cycle.")
                    else: # Workflow Failed
                        st.error("❌ DebugIQ workflow failed.")

            else:
                # Handle API request errors (e.g., 404 if issue ID not found)
                st.error(status_response["error"])
                if "backend_detail" in status_response: st.json({"Backend Detail": status_response["backend_detail"]})
                # Stop polling on API errors
                st.session_state.workflow_completed = True

    else:
        # Display idle status if no issue ID is active
        st.info("Enter an Issue ID or trigger a workflow to see status.")
        # Ensure polling is off if no active ID
        st.session_state.workflow_completed = True


# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    st.write("View system and performance metrics for the DebugIQ backend.")

    if st.button("📊 Fetch Metrics"):
        with st.spinner("Fetching system metrics..."):
            response = make_api_request("GET", "system_metrics") # <--- Use endpoint key

        if "error" not in response:
            st.subheader("Backend System Metrics")
            # Display metrics - check backend response format
            st.json(response) # Assuming response is a JSON dictionary
        else:
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


# === DebugIQ Voice Agent Section (Dedicated Section at the bottom) ===
# Include the complete Voice Agent Section here if keeping it

# Make sure to include necessary imports for the voice section if you keep it:
# import av, numpy as np, io, wave, streamlit_webrtc, threading, base64, re
# Include frames_to_wav_bytes and audio_frame_callback functions
# Include the webrtc_streamer component
# Include the recording logic for start/stop buttons
# Include the text chat logic


# Example of voice related content placeholders - REPLACE with your actual Voice Agent code
st.markdown("---")
st.subheader("Voice Agent (Placeholder)")
st.write("Voice agent code goes here.")
