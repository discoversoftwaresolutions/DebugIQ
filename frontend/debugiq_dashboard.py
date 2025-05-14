# File: frontend/debugiq_dashboard.py

# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh
import json  # Explicitly import json for error handling

# Imports for Voice Agent section
import av
import numpy as np
import io
import wave
# Corrected import: Removed ClientSettings
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import base64
import re
import threading # Used for thread-safe buffer access in callback
from urllib.parse import urljoin

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
# Use environment variable for the backend URL, with a fallback.
# Set BACKEND_URL environment variable in your deployment environment (e.g., Railway settings).
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app") # <-- Ensure this fallback is correct or use env var

# Define API endpoint paths relative to BACKEND_URL
ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",  # Analyze.py endpoint
    "qa_validation": "/qa/run",               # QA endpoint
    "doc_generation": "/doc/generate",        # Documentation endpoint
    "issues_inbox": "/issues/attention-needed", # Issues endpoint
    "workflow_run": "/workflow/run_autonomous_workflow", # Workflow trigger
    "workflow_status": "/issues/{issue_id}/status", # Workflow status (requires formatting)
    "system_metrics": "/metrics/status",      # Metrics endpoint
    # Paths for Voice/Gemini - CONFIRM THESE WITH YOUR BACKEND ROUTERS
    "voice_transcribe": "/voice/transcribe",  # Example path - CHECK YOUR BACKEND
    "gemini_chat": "/gemini/chat",            # Example path - CHECK YOUR BACKEND
    "tts": "/voice/tts"                     # Example path - CHECK YOUR BACKEND
}

# === Session State Initialization ===
# Initialize all necessary session state variables at the top level
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'audio_buffer_lock' not in st.session_state:
    # Initialize the lock for thread-safe buffer access
    st.session_state.audio_buffer_lock = threading.Lock()
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    # Store list of {"role": "user" or "ai", "content": "...", "audio": b"..."}
    st.session_state.chat_history = []
if 'active_issue_id' not in st.session_state:
    st.session_state.active_issue_id = None
if 'workflow_completed' not in st.session_state:
    st.session_state.workflow_completed = False
if 'last_status' not in st.session_state:
    st.session_state.last_status = None
if 'error_message' not in st.session_state:
     st.session_state.error_message = None # To store error message from workflow status

# === Helper Functions ===
def make_api_request(method, endpoint_key, payload=None, return_json=True):
    """Makes an API request to the backend."""
    # Ensure endpoint_key exists in ENDPOINTS
    if endpoint_key not in ENDPOINTS:
        logger.error(f"Invalid endpoint key: {endpoint_key}")
        return {"error": f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'."}

    path_template = ENDPOINTS[endpoint_key]
    path = path_template # Default path

    # --- Construct the path, handling special cases ---
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

    # Construct the full URL by joining BACKEND_URL and the path using urljoin for robustness
    url = urljoin(BACKEND_URL, path.lstrip('/')) # Use lstrip('/') to handle potential double slashes

    try:
        logger.info(f"Making API request: {method} {url}")
        # Set a reasonable timeout for API calls
        response = requests.request(method, url, json=payload, timeout=120) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"API request successful: {method} {url}")
        if return_json:
            # Attempt to parse JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response from {url}")
                return {"error": f"Backend response is not valid JSON.", "raw_response": response.text}
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
                # Prioritize 'detail' key, fallback to the whole JSON object
                backend_detail = backend_json.get('detail', backend_json)
                detail = f"Status {e.response.status_code} - {backend_detail}" # More informative error
            except json.JSONDecodeError:
                # If not JSON, include raw text response
                backend_detail = e.response.text
                detail = f"Status {e.response.status_code} - Response Text: {backend_detail}"
            except Exception as json_e:
                logger.warning(f"Could not parse backend error response: {json_e}")

        return {"error": f"API request failed: {detail}", "backend_detail": backend_detail}

# === Audio Processing Helper ===
def frames_to_wav_bytes(frames):
    """Converts a list of audio frames (av.AudioFrame) to WAV formatted bytes."""
    if not frames:
        logger.warning("No audio frames provided for WAV conversion.")
        return None

    # Assume consistent format across frames, use format from session state if available
    audio_format = st.session_state.get('audio_format')
    if not audio_format:
        logger.error("Audio format information not found in session state.")
        # Attempt to infer from the first frame as a fallback
        try:
            frame_0 = frames[0]
            audio_format = {
                'sample_rate': frame_0.sample_rate,
                'format_name': frame_0.format.name,
                'channels': frame_0.layout.channels,
                'sample_width_bytes': frame_0.format.bytes, # Bytes per sample per channel
                'layout_name': frame_0.layout.name # Store layout name too
            }
            logger.warning("Inferred audio format from first frame due to missing session state.")
        except Exception as e:
             logger.error(f"Could not infer audio format from frames: {e}")
             return None


    sample_rate = audio_format['sample_rate']
    format_name = audio_format['format_name']
    channels = audio_format['channels']
    sample_width_bytes = audio_format['sample_width_bytes']
    logger.info(f"Converting audio frames to WAV. Format: {format_name}, Channels: {channels}, Sample Rate: {sample_rate}, Sample Width: {sample_width_bytes} bytes.")

    # Check for common formats and convert to raw bytes
    # streamlit-webrtc typically provides s16, s32p, or f32p
    # s16 is signed 16-bit int, interleaved
    raw_data = b"" # Initialize raw data buffer
    try:
        if 's16' in format_name and audio_format.get('layout_name', '').lower() in ['mono', 'stereo']: # Added layout_name check
            # For s16 interleaved, data is in the first plane. Concatenate raw bytes.
            raw_data = b"".join([frame.planes[0].buffer.tobytes() for frame in frames])
            logger.info(f"Concatenated raw bytes from s16 frames, total size: {len(raw_data)} bytes.")
        elif 's32p' in format_name or 'f32p' in format_name:
            # Planar formats: data for each channel is in a separate plane. Need to interleave.
            logger.info(f"Processing planar audio format: {format_name}")
            all_channels_data = [np.concatenate([frame.planes[i].to_ndarray() for frame in frames]) for i in range(channels)]
            # Stack channel data (e.g., [samples_ch1], [samples_ch2]) -> [[s1_ch1, s1_ch2], [s2_ch1, s2_ch2], ...]
            interleaved_data = np.stack(all_channels_data, axis=-1)
            raw_data = interleaved_data.tobytes()
            logger.info(f"Interleaved planar data, resulting in {len(raw_data)} bytes.")
        else:
            logger.error(f"Unsupported audio format or layout for WAV conversion: {format_name}, {audio_format.get('layout_name', 'Unknown Layout')}. Support for s16, s32p, f32p (mono/stereo) implemented.")
            return None
    except Exception as e:
        logger.error(f"Error processing audio frame bytes: {e}")
        return None

    if not raw_data:
        logger.warning("No raw audio data generated.")
        return None

    # Create a WAV file in memory
    try:
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width_bytes)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_data)
            wav_bytes = wav_buffer.getvalue()
            logger.info(f"Successfully created WAV data of size {len(wav_bytes)} bytes.")
            return wav_bytes
    except Exception as e:
        logger.error(f"Error creating WAV file: {e}")
        return None

# === WebRTC Audio Frame Callback ===
# Callback function to receive and process audio frames from the browser.
def audio_frame_callback(frame: av.AudioFrame):
    """Callback function to receive and process audio frames from the browser."""
    # Use a thread-safe buffer to handle audio frames received in a different thread
    # Lock to ensure thread-safe access to session state from the callback thread
    with st.session_state.audio_buffer_lock:
        if st.session_state.get('is_recording', False):
            # Append the audio frame to the session state's buffer
            st.session_state.audio_buffer.append(frame)

            # Store format info from the first frame if not already stored and buffer is not empty
            if 'audio_format' not in st.session_state and st.session_state.audio_buffer:
                 frame_0 = st.session_state.audio_buffer[0]
                 st.session_state.audio_format = {
                     'sample_rate': frame_0.sample_rate,
                     'format_name': frame_0.format.name,
                     'channels': frame_0.layout.channels,
                     'sample_width_bytes': frame_0.format.bytes,
                     'layout_name': frame_0.layout.name # Store layout name too
                 }
                 logger.info(f"Stored audio format in session state from first frame: {st.session_state.audio_format}")

            # Log the audio frame details for debugging purposes (optional, can be chatty)
            # logger.debug(f"Audio frame received: {len(st.session_state.audio_buffer)} frames buffered.")


# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

# Placeholder for status messages (needed for updating during recording)
status_placeholder = st.empty()

# === Sidebar for GitHub Integration ===
st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repo",
    key="sidebar_github_url"
)
if github_url:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", github_url)
    if match:
        owner, repo = match.groups()
        st.sidebar.success(f"**Repository:** {repo}\n\n(Owner: {owner})")
    else:
        st.sidebar.error("Invalid GitHub URL format.")

# === Application Tabs ===
# Ensure tab count and names match your desired UI
# Added a Voice Agent tab back as it contains interactive elements
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow Trigger", "🔍 Workflow Status", "📈 Metrics", "🎤 Voice Agent"])
# Assign tabs to variables based on their index
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow_trigger, tab_status, tab_metrics, tab_voice = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    # Updated file uploader types based on backend model
    uploaded_file = st.file_uploader(
        "Upload Traceback or Source File",
        type=["txt", "py", "java", "js", "cpp", "c", "log"], # Added log files
        key="trace_file_uploader"
    )

    if uploaded_file:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            # Display original code - using st.code is better for syntax highlighting
            st.subheader("Original Content")
            # Attempt to guess language from file type or extension
            file_extension = os.path.splitext(uploaded_file.name)[1].lstrip('.').lower()
            mime_type_language = uploaded_file.type.split('/')[-1].lower()

            # Simple mapping for common types/extensions
            language_map = {
                "py": "python", "java": "java", "js": "javascript",
                "cpp": "cpp", "c": "c", "txt": "plaintext", "log": "plaintext"
            }
            # Prioritize extension, fallback to mime type
            original_language = language_map.get(file_extension, language_map.get(mime_type_language, "plaintext"))

            st.code(file_content, language=original_language, height=300)

            if st.button("🔬 Analyze & Suggest Patch", key="analyze_patch_btn"):
                with st.spinner("Analyzing and suggesting patch..."):
                    # Corrected Payload to match backend AnalyzeRequest model
                    # Requires 'code', 'language', (optional 'context')
                    payload = {
                        "code": file_content,
                        "language": original_language, # Use detected language
                        # "context": {} # <--- Add optional context if needed by backend
                    }
                    # Use make_api_request with the endpoint key
                    response = make_api_request("POST", "suggest_patch", payload)

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
                    # Pass the original file content to the editor initially
                    edited_patch_content = st_ace(
                        value=file_content, # Start with the original file content for editing
                        language=original_language, # Use original language for editing
                        theme="monokai",
                        height=350,
                        key="ace_editor_patch" # Unique key
                    )

                    # Diff View (Original vs. Edited Content)
                    st.markdown("### 🔍 Diff View (Original vs. Edited Content)")
                    # Check if both original code and edited content are available
                    if edited_patch_content is not None and file_content is not None:
                        # Generate diff as HTML
                        diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
                            fromlines=file_content.splitlines(),
                            tolines=edited_patch_content.splitlines(),
                            fromdesc=f"Original Content ({uploaded_file.name})",
                            todesc="Edited Content",
                            context=True
                        )
                        # Display HTML in Streamlit
                        st.components.v1.html(diff_html, height=400, scrolling=True) # Adjusted height
                    else:
                        st.info("Upload original content and edit it above to see the diff.")

                else:
                    # Display error from make_api_request helper
                    st.error(response["error"])
                    # Optionally display backend detail if available
                    if "backend_detail" in response and response["backend_detail"] not in ["N/A", "", response["error"].split(" - ", 1)[-1]]:
                         st.json({"Backend Detail": response["backend_detail"]})
        except Exception as e:
             st.error(f"An error occurred processing the uploaded file: {e}")
             logger.exception("Error processing uploaded file in Traceback tab.")


# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    st.write("Upload a patch file to run QA validation checks.")
    uploaded_patch = st.file_uploader(
        "Upload Patch File",
        type=["txt", "diff", "patch"],
        key="qa_patch_uploader"
    )

    if uploaded_patch:
        try:
            patch_content = uploaded_patch.read().decode("utf-8")
            st.subheader("Patch Content")
            st.code(patch_content, language="diff", height=200)

            if st.button("🛡️ Validate Patch", key="qa_validate_btn"):
                 with st.spinner("Running QA validation..."):
                    # Check your backend's expected payload for /qa/run
                    payload = {"patch_diff": patch_content} # Assuming backend expects 'patch_diff'
                    response = make_api_request("POST", "qa_validation", payload)

                 if "error" not in response:
                    st.subheader("Validation Results")
                    # Display validation results - check backend response format
                    st.json(response) # Assuming response is a JSON summary/report
                 else:
                    st.error(response["error"])
                    if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        except Exception as e:
             st.error(f"An error occurred processing the uploaded patch file: {e}")
             logger.exception("Error processing uploaded patch file in QA tab.")
    else:
         st.warning("Please upload a patch file to enable validation.")


# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    st.write("Upload a code file to generate documentation automatically.")
    uploaded_code_doc = st.file_uploader( # Renamed key to avoid conflict
        "Upload Code File for Documentation",
        type=["txt", "py", "java", "js", "cpp", "c"],
        key="doc_code_uploader"
    )

    if uploaded_code_doc:
        try:
            code_content = uploaded_code_doc.read().decode("utf-8")
            st.subheader("Code Content")
            file_extension_doc = os.path.splitext(uploaded_code_doc.name)[1].lstrip('.').lower()
            mime_type_language_doc = uploaded_code_doc.type.split('/')[-1].lower()
            doc_language = language_map.get(file_extension_doc, language_map.get(mime_type_language_doc, "plaintext")) # Use language_map

            st.code(code_content, language=doc_language, height=200)

            if st.button("📝 Generate Documentation", key="doc_generate_btn"):
                 with st.spinner("Generating documentation..."):
                    # Check your backend's expected payload for /doc/generate
                    payload = {"code": code_content, "language": doc_language} # Assuming backend needs code and language
                    response = make_api_request("POST", "doc_generation", payload)

                 if "error" not in response:
                    # Assuming the response contains a "documentation" key with markdown or text
                    st.subheader("Generated Documentation")
                    st.markdown(response.get("documentation", "No documentation generated."))
                 else:
                    st.error(response["error"])
                    if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        except Exception as e:
             st.error(f"An error occurred processing the uploaded code file for documentation: {e}")
             logger.exception("Error processing uploaded code file in Documentation tab.")
    else:
         st.warning("Please upload a code file to enable documentation generation.")

# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues Inbox")
    st.write("This section lists issues needing attention from the autonomous workflow.")

    # Ensure issues are fetched on initial load or tab switch if needed, or just on button click.
    # Let's stick to button click for explicit control.
    if st.button("🔄 Refresh Issues", key="issues_refresh_btn"):
        with st.spinner("Fetching issues..."):
            response = make_api_request("GET", "issues_inbox")

        if "error" not in response:
            issues = response.get("issues", []) # Default to empty list if no 'issues' key
            if issues:
                st.subheader("Issues Needing Attention")
                # Display issues - check backend response format
                # Assuming backend returns {"issues": [{"id": ..., "status": ..., "error_message": ...}, ...]}
                for issue in issues:
                    issue_id_display = issue.get('id', 'N/A')
                    issue_status_display = issue.get('status', 'Unknown Status')
                    issue_error_display = issue.get('error_message', 'No error details provided.')

                    with st.expander(f"Issue ID: {issue_id_display} - Status: {issue_status_display}"):
                        st.write(f"**Error Details:** {issue_error_display}")
                        # You might add more issue details here if provided by the backend, e.g.,
                        # if 'metadata' in issue:
                        #    st.json(issue['metadata'])
            else:
                st.info("No issues needing attention found.")
        else:
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})

# === Workflow Tab (Trigger) ===
with tab_workflow_trigger: # Use the correct variable name
    st.header("🤖 Autonomous Workflow Trigger")
    st.write("Trigger an autonomous workflow run for a specific issue.")
    issue_id_trigger = st.text_input( # Renamed variable to avoid conflict
        "Issue ID to Trigger Workflow",
        placeholder="e.g., BUG-123",
        key="workflow_trigger_issue_id"
    )

    if st.button("▶️ Trigger Workflow", key="workflow_trigger_btn"):
        if issue_id_trigger:
            with st.spinner(f"Triggering workflow for issue {issue_id_trigger}..."):
                # Check backend payload for /workflow/run_autonomous_workflow
                payload = {"issue_id": issue_id_trigger} # Assuming backend expects 'issue_id'
                response = make_api_request("POST", "workflow_run", payload)

            if "error" not in response:
                st.success(f"Workflow triggered successfully for Issue {issue_id_trigger}.")
                st.json(response.get('details', {'message': 'No specific details returned.'})) # Display response details

                # Store the issue_id to enable status polling in the next tab
                st.session_state.active_issue_id = issue_id_trigger
                st.session_state.workflow_completed = False # Reset completed state to start polling
                st.session_state.last_status = None # Reset last status
                st.session_state.error_message = None # Reset error message
                st.rerun() # Rerun to immediately show status tab update
            else:
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please enter an Issue ID to trigger the workflow.")

# === Workflow Check Tab (Status/Polling) ===
with tab_status: # Use the correct variable name
    st.header("🔍 Autonomous Workflow Status")
    st.write("Check the live status of an ongoing workflow.")

    # Add a text input to manually set the issue ID to poll
    issue_id_for_polling_manual = st.text_input(
        "Enter Issue ID to check status manually:",
        placeholder="e.g., BUG-123",
        key="workflow_status_issue_id_manual"
    )

    # If manually entered issue_id, use it for polling, prioritizing manual input
    if issue_id_for_polling_manual:
        if st.session_state.get("active_issue_id") != issue_id_for_polling_manual:
            # Only update if the ID has changed
            st.session_state.active_issue_id = issue_id_for_polling_manual
            st.session_state.workflow_completed = False # Assume not completed if manually set
            st.session_state.last_status = None # Reset last status
            st.session_state.error_message = None # Reset error message
            st.rerun() # Rerun to start polling for the new ID
    # If no manual input, use the ID from the trigger tab if available
    elif not st.session_state.get("active_issue_id") and st.session_state.get("workflow_trigger_issue_id"):
         st.session_state.active_issue_id = st.session_state.workflow_trigger_issue_id # Use the ID from the trigger input if no manual override


    display_issue_id = st.session_state.get("active_issue_id") or "None (Trigger workflow or enter ID above)"
    st.markdown(f"Checking status for Issue ID: **{display_issue_id}**")

    # Progress tracker logic
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed", # This status name might need adjustment based on backend
        "📦 PR Created" # This status name might need adjustment based on backend
    ]
    # Map backend status strings to progress steps (0-indexed)
    # !!! IMPORTANT: These strings must exactly match the statuses returned by your backend !!!
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
        """Displays the workflow progress based on the current status."""
        step = progress_map.get(status, 0)
        # Ensure step doesn't exceed the number of labels for progress bar calculation
        progress_value = min(step + 1, len(progress_labels)) / len(progress_labels) if status != failed_status else 0
        st.progress(progress_value)

        for i, label in enumerate(progress_labels):
            if status == failed_status:
                icon = "❌" if i == step else " " # Mark the step where it failed if mapped
                st.markdown(f"{icon} {label}")
            else:
                icon = "✅" if i < step else ("🔄" if i == step else "⏳") # Spinner for the *current* step
                st.markdown(f"{icon} {label}")

        # Display final status messages below the progress bar
        if status == failed_status:
             st.error("❌ Workflow Failed")
             if st.session_state.get('error_message'):
                  st.error(f"Details: {st.session_state.error_message}")
        elif status == terminal_status:
             st.success("✅ DebugIQ agents completed full cycle.")


    # Polling logic (autorefresh only if an issue is active and workflow not complete)
    # Use a unique key for autorefresh based on the active_issue_id
    if st.session_state.get("active_issue_id") and not st.session_state.get("workflow_completed", True):
        # Autorefresh triggers a rerun every X milliseconds (e.g., 3000 = 3 seconds)
        st_autorefresh(interval=3000, key=f"workflow-refresh-{st.session_state.active_issue_id}")
        logger.info(f"Autorefresh enabled for issue: {st.session_state.active_issue_id}")
    else:
        logger.info("Autorefresh disabled.")


    # --- Fetch and Display Status ---
    # This code runs on every rerun, including those triggered by autorefresh
    # Only fetch if an issue ID is active and we believe the workflow is still running
    if st.session_state.get("active_issue_id") and not st.session_state.get("workflow_completed", True):
        try:
            # Use make_api_request with the endpoint key - special handling in make_api_request for formatting
            status_response = make_api_request("GET", "workflow_status")

            if "error" not in status_response:
                # Assumes backend returns {"status": "...", "error_message": "..."}
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")

                st.session_state.last_status = current_status # Store last status
                st.session_state.error_message = error_message # Store last error message if any

                status_placeholder.info(f"🔁 Live Status: **{current_status}**") # Use placeholder to update status

                show_agent_progress(current_status) # Update progress bar and labels

                # Check if workflow has reached a terminal state (completed or failed)
                if current_status == terminal_status or current_status == failed_status:
                    st.session_state.workflow_completed = True # Stop polling
                    status_placeholder.empty() # Clear the polling status message
                    logger.info(f"Workflow for issue {st.session_state.active_issue_id} reached terminal state: {current_status}. Polling stopped.")
                    st.rerun() # Rerun one last time to show final state without polling enabled

            else:
                # Handle API request errors (e.g., 404 if issue ID not found)
                st.error(status_response["error"])
                if "backend_detail" in status_response: st.json({"Backend Detail": status_response["backend_detail"]})
                # Stop polling on API errors
                st.session_state.workflow_completed = True
                st.session_state.last_status = failed_status # Indicate failure state
                st.session_state.error_message = status_response.get("error", "API Error")
                status_placeholder.empty() # Clear the polling status message
                logger.error(f"API error while polling status for issue {st.session_state.active_issue_id}. Polling stopped.")
                st.rerun() # Rerun to show failure message

        except Exception as e:
             st.error(f"An unexpected error occurred during status polling: {e}")
             logger.exception(f"Unexpected error polling workflow status for issue {st.session_state.active_issue_id}.")
             st.session_state.workflow_completed = True
             st.session_state.last_status = failed_status
             st.session_state.error_message = str(e)
             status_placeholder.empty()
             st.rerun()


    # Display final state if workflow is completed or no issue is active
    elif st.session_state.get("workflow_completed", True) and st.session_state.get("active_issue_id"):
        # Workflow was completed or failed, show the final state using the last known status
        status_placeholder.empty() # Ensure placeholder is empty
        show_agent_progress(st.session_state.get("last_status", "Unknown"))
        # Explicitly display the error message if it failed
        if st.session_state.get("last_status") == failed_status and st.session_state.get("error_message"):
             st.error(f"Last recorded error: {st.session_state.error_message}")

    elif not st.session_state.get("active_issue_id"):
        # No active issue ID set
        status_placeholder.info("Enter an Issue ID or trigger a workflow to see status.")
        # Ensure polling is off if no active ID
        st.session_state.workflow_completed = True


# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    st.write("View system and performance metrics for the DebugIQ backend.")

    if st.button("📊 Fetch Metrics", key="metrics_fetch_btn"):
        with st.spinner("Fetching system metrics..."):
            response = make_api_request("GET", "system_metrics")

        if "error" not in response:
            st.subheader("Backend System Metrics")
            # Display metrics - check backend response format
            st.json(response) # Assuming response is a JSON dictionary
        else:
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


# === DebugIQ Voice Agent Section ===
# Dedicated Section at the bottom, now in its own tab
with tab_voice: # Use the correct variable name
    st.header("🎤 DebugIQ Voice Agent")
    st.write("Interact with the DebugIQ agent using your voice or text.")

    # Use direct arguments for rtc_configuration and media_stream_constraints
    webrtc_ctx = webrtc_streamer(
        key="voice_agent_streamer",
        mode=WebRtcMode.SENDONLY, # Send audio from browser to server
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Direct config
        media_stream_constraints={"audio": True, "video": False}, # Direct constraints
        audio_frame_callback=audio_frame_callback, # Make sure this callback is defined above
        async_processing=True # Process frames asynchronously
    )

    # Display recording status
    status_placeholder_voice = st.empty()
    # Update initial status check to use the webrtc_ctx state safely
    is_webrtc_ready = webrtc_ctx is not None and hasattr(webrtc_ctx.state, 'playing')
    if not st.session_state.is_recording and st.session_state.recording_status == "Idle" and is_webrtc_ready and webrtc_ctx.state.playing:
         status_placeholder_voice.success("Status: Microphone Ready")
    elif not st.session_state.is_recording and st.session_state.recording_status == "Idle":
        status_placeholder_voice.info("Status: Idle")
    else:
         # Show current recording status if not Idle
         if st.session_state.recording_status == "Recording...":
              status_placeholder_voice.warning(f"Status: {st.session_state.recording_status}")
         elif st.session_state.recording_status != "Idle":
              status_placeholder_voice.info(f"Status: {st.session_state.recording_status}")



    # --- Recording Control Buttons ---
    col1, col2 = st.columns(2)

    with col1:
        # Check if webrtc_ctx is available and stream is playing
        is_webrtc_playing = webrtc_ctx is not None and hasattr(webrtc_ctx.state, 'playing') and webrtc_ctx.state.playing
        if st.button("🔴 Start Recording", key="start_recording_btn", disabled=not is_webrtc_playing): # Disable if stream not ready
            st.session_state.is_recording = True
            st.session_state.audio_buffer = [] # Clear buffer for new recording
            st.session_state.recording_status = "Recording..."
            # Status is updated by the logic block below, no need to update here
            logger.info("Recording started.")


    with col2:
        # Disable Stop button if not recording
        if st.button("⏹️ Stop Recording", key="stop_recording_btn", disabled=not st.session_state.is_recording):
            st.session_state.is_recording = False
            st.session_state.recording_status = "Processing Audio..."
            # Status is updated by the logic block below, no need to update here
            logger.info("Recording stopped. Processing audio...")

            # Process the recorded audio when recording stops
            if st.session_state.audio_buffer:
                with st.spinner("Transcribing audio..."):
                    # Convert audio frames to WAV bytes
                    wav_bytes = frames_to_wav_bytes(st.session_state.audio_buffer)

                    if wav_bytes:
                        # Encode WAV bytes to Base64 for sending to backend
                        encoded_audio = base64.b64encode(wav_bytes).decode('utf-8')

                        # --- Send audio to backend for transcription ---
                        st.session_state.recording_status = "Sending for Transcription..."
                        status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here
                        transcribe_payload = {"audio_base64": encoded_audio}
                        transcription_response = make_api_request("POST", "voice_transcribe", transcribe_payload)

                        if "error" not in transcription_response:
                            transcribed_text = transcription_response.get("text", "Could not transcribe audio.")
                            st.session_state.recording_status = "Transcription Complete."
                            status_placeholder_voice.success(f"Transcription: {transcribed_text}") # Update status here

                            # Add user's voice input to chat history
                            st.session_state.chat_history.append({"role": "user", "content": transcribed_text, "audio": wav_bytes})


                            # --- Send transcribed text to Gemini Chat ---
                            st.session_state.recording_status = "Sending to Gemini..."
                            status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here
                            with st.spinner("Getting response from Gemini..."):
                                # Send the transcribed text to Gemini. Backend interprets.
                                gemini_payload = {"text": transcribed_text}
                                # Use make_api_request with the endpoint key
                                gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

                            if "error" not in gemini_response:
                                ai_response_text = gemini_response.get("text", "No response from AI.")
                                # Optional: Get audio response from TTS if available and desired
                                ai_response_audio_base64 = gemini_response.get("audio_base64") # Assuming backend returns base64 audio

                                ai_response_audio = None
                                if ai_response_audio_base64:
                                    try:
                                         ai_response_audio = base64.b64decode(ai_response_audio_base64)
                                         logger.info("Received AI audio response.")
                                    except (base64.binascii.Error, TypeError) as e:
                                         logger.error(f"Failed to decode base64 audio from backend: {e}")
                                         st.warning("Received audio response from backend, but failed to decode it.")


                                # Add AI response (text and optional audio) to chat history
                                st.session_state.chat_history.append({
                                    "role": "ai",
                                    "content": ai_response_text,
                                    "audio": ai_response_audio # Store raw bytes
                                })

                                st.session_state.recording_status = "AI Response Received."
                                status_placeholder_voice.success(f"Status: {st.session_state.recording_status}") # Update status here

                                # Clear the buffer after processing
                                st.session_state.audio_buffer = []
                                st.session_state.audio_format = None # Clear format info too
                                logger.info("Audio buffer cleared.")
                                st.rerun() # Rerun to display chat history

                            else:
                                st.error(f"Gemini Chat API Error: {gemini_response['error']}")
                                if "backend_detail" in gemini_response: st.json({"Backend Detail": gemini_response["backend_detail"]})
                                st.session_state.recording_status = "AI Request Failed."
                                status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                                st.rerun() # Rerun to display error in chat history

                    else:
                        st.warning("Failed to convert recorded audio frames to WAV.")
                        st.session_state.recording_status = "Audio Processing Failed."
                        status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                # Ensure buffer is cleared even if WAV conversion failed
                st.session_state.audio_buffer = []
                st.session_state.audio_format = None # Clear format info too
                logger.info("Audio buffer cleared after processing attempt.")
                st.rerun() # Rerun to update UI

            else:
                st.info("No audio was recorded.")
                st.session_state.recording_status = "Idle"
                status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here
                st.rerun() # Rerun to update UI


    # --- Text Chat Input (Alternative to Voice) ---
    st.markdown("---") # Separator
    st.subheader("Text Chat")
    text_query_input = st.text_input("Type your query here:", key="text_chat_input")
    send_text_button_text = st.button("Send Text Query", key="send_text_btn")

    if send_text_button_text and text_query_input:
        # Use make_api_request for the text query to the backend Gemini Chat endpoint
        st.session_state.recording_status = "Processing Text Query..."
        status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here

        user_text_query = text_query_input
        # Add user's text to chat history immediately
        st.session_state.chat_history.append({"role": "user", "content": user_text_query})

        # Clear the text input box after sending (optional but good UX)
        st.session_state.text_chat_input = ""
        st.rerun() # Rerun to clear input and show updated chat history

    # --- Process Text Query (after rerun triggered by button) ---
    # This logic should run if a text query was just added to history and needs processing
    # We can detect this by checking the last message role and if it's new
    # Use try-except to handle potential IndexError if chat_history is empty unexpectedly
    try:
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user" and "processed" not in st.session_state.chat_history[-1]:
             user_text_to_process = st.session_state.chat_history[-1]["content"]

             st.session_state.recording_status = "Sending Text to Gemini..."
             status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here

             with st.spinner("Getting response from Gemini..."):
                 # Send the text query to Gemini. Backend interprets.
                 gemini_payload = {"text": user_text_to_process}
                 gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

             # Mark the user message as processed
             st.session_state.chat_history[-1]["processed"] = True

             if "error" not in gemini_response:
                 ai_response_text = gemini_response.get("text", "No response from AI.")
                 ai_response_audio_base64 = gemini_response.get("audio_base64")

                 ai_response_audio = None
                 if ai_response_audio_base64:
                     try:
                          ai_response_audio = base64.b64decode(ai_response_audio_base64)
                          logger.info("Received AI audio response for text query.")
                     except (base64.binascii.Error, TypeError) as e:
                          logger.error(f"Failed to decode base64 audio from backend (text query): {e}")
                          st.warning("Received audio response from backend, but failed to decode it.")

                 # Add AI response to chat history
                 st.session_state.chat_history.append({
                     "role": "ai",
                     "content": ai_response_text,
                     "audio": ai_response_audio # Store raw bytes
                 })
                 st.session_state.recording_status = "AI Response Received."
                 status_placeholder_voice.success(f"Status: {st.session_state.recording_status}") # Update status here
                 st.rerun() # Rerun to display the new AI message

             else:
                 st.error(f"Gemini Chat API Error: {gemini_response['error']}")
                 if "backend_detail" in gemini_response: st.json({"Backend Detail": gemini_response["backend_detail"]})
                 st.session_state.recording_status = "AI Request Failed."
                 status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                 # Add an error message to chat history as an AI response
                 st.session_state.chat_history.append({
                      "role": "ai",
                      "content": f"Error: {gemini_response['error']}",
                      "audio": None
                 })
                 st.rerun() # Rerun to display the error message in chat
    except IndexError:
        # This can happen on initial load or specific reruns if chat_history is manipulated
        logger.warning("IndexError processing chat history. Skipping text query processing.")


    # --- Display Chat History ---
    st.markdown("---")
    st.subheader("Chat History")
    # Display chat messages in reverse order (latest first)
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        role = "🧑‍💻 User" if message["role"] == "user" else "🤖 AI"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Play audio if available
            if message.get("audio"):
                 try:
                     st.audio(message["audio"], format='audio/wav') # Assuming WAV format
                 except Exception as e:
                     st.warning(f"Could not play audio: {e}")
                     logger.error(f"Error playing audio in chat history: {e}")
