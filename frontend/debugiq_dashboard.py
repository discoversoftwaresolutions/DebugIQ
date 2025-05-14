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
# Corrected import: Removed ClientSettings which is deprecated
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
# IMPORTANT: Replace with your actual backend URL if different from the fallback.
BACKEND_URL=https://debugiq-backend-production.up.railway.app

# Define API endpoint paths relative to BACKEND_URL
# IMPORTANT: These paths must exactly match your backend FastAPI/API endpoints
# after considering router prefixes in main.py.
ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",  # analyze.py endpoint
    "qa_validation": "/qa/run",               # qa endpoint
    "doc_generation": "/doc/generate",        # doc endpoint
    "issues_inbox": "/issues/attention-needed", # issues endpoint
    "workflow_run": "/workflow/run_autonomous_workflow", # workflow trigger
    "workflow_status": "/issues/{issue_id}/status", # workflow status (requires issue_id formatting)
    "system_metrics": "/metrics/status",      # metrics endpoint
    # Paths for Voice/Gemini - CONFIRM THESE WITH YOUR BACKEND ROUTERS
    "voice_transcribe": "/voice/transcribe",  # Example path - CHECK YOUR BACKEND
    "gemini_chat": "/gemini/chat",            # Example path - CHECK YOUR BACKEND
    "tts": "/voice/tts"                     # Example path - CHECK YOUR BACKEND (if used separately)
}

# === Session State Initialization ===
# Initialize all necessary session state variables at the top level
# This ensures they exist on the first run and between reruns.
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'audio_buffer_lock' not in st.session_state:
    # Initialize the lock for thread-safe buffer access from the WebRTC callback thread
    st.session_state.audio_buffer_lock = threading.Lock()
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    # Store list of chat messages, each being a dict like {"role": "user" or "ai", "content": "...", "audio": b"..."}
    st.session_state.chat_history = []
if 'active_issue_id' not in st.session_state:
    # Stores the issue ID currently being tracked by the workflow status tab
    st.session_state.active_issue_id = None
if 'workflow_completed' not in st.session_state:
    # Flag to indicate if the workflow for the active issue ID has finished (success or failure)
    st.session_state.workflow_completed = False
if 'last_status' not in st.session_state:
    # Stores the last known status of the active workflow, used after polling stops
    st.session_state.last_status = None
if 'error_message' not in st.session_state:
     # Stores the last known error message from the active workflow
     st.session_state.error_message = None
if 'workflow_trigger_issue_id' not in st.session_state:
     # Stores the issue ID entered in the workflow trigger text input
     st.session_state.workflow_trigger_issue_id = ""
if 'text_chat_input' not in st.session_state:
     # Stores the current value of the text chat input box
     st.session_state.text_chat_input = ""


# === Helper Functions ===
def make_api_request(method, endpoint_key, payload=None, return_json=True):
    """
    Makes an API request to the backend.

    Args:
        method (str): HTTP method (e.g., "GET", "POST").
        endpoint_key (str): Key from the ENDPOINTS dictionary.
        payload (dict, optional): JSON payload for POST requests. Defaults to None.
        return_json (bool, optional): Whether to return JSON or raw content. Defaults to True.

    Returns:
        dict or bytes: JSON response (dict) or raw content (bytes) on success,
                       a dict with an "error" key on failure.
    """
    # Ensure endpoint_key exists in ENDPOINTS
    if endpoint_key not in ENDPOINTS:
        logger.error(f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'.")
        return {"error": f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'."}

    path_template = ENDPOINTS[endpoint_key]
    path = path_template # Default path assumes no formatting needed

    # --- Construct the path, handling special cases that require formatting ---
    if endpoint_key == "workflow_status":
        issue_id = st.session_state.get("active_issue_id")
        if not issue_id:
            # Indicate that polling should stop if no issue_id
            st.session_state.workflow_completed = True
            # Return a specific error message indicating the missing ID
            logger.warning("Workflow status requested but no active_issue_id in session state.")
            return {"error": "No active issue ID to check workflow status. Please trigger a workflow or enter an ID."}
        try:
            # Format the path using the issue_id from session state
            path = path_template.format(issue_id=issue_id)
        except KeyError as e:
            # Should not happen if logic sets active_issue_id correctly, but included for robustness
            logger.error(f"Failed to format workflow_status path: Missing key {e}", exc_info=True)
            st.session_state.workflow_completed = True # Stop polling on formatting error
            return {"error": f"Internal error formatting workflow status URL: Missing issue ID key ({e})."}

    # Construct the full URL by joining BACKEND_URL and the path using urljoin for robustness
    # lstrip('/') ensures there isn't a double slash if BACKEND_URL ends with one
    url = urljoin(BACKEND_URL, path.lstrip('/'))

    try:
        logger.info(f"Making API request: {method} {url}")
        # Set a reasonable timeout for API calls (e.g., 120 seconds for potentially long operations)
        response = requests.request(method, url, json=payload, timeout=120)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info(f"API request successful: {method} {url}")

        if return_json:
            # Attempt to parse JSON response
            try:
                # Handle empty response bodies or non-JSON responses gracefully
                if response.text:
                    return response.json()
                else:
                    logger.warning(f"API request to {url} returned empty response body.")
                    return {} # Return empty dict for empty successful JSON response
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response from {url}. Response text: {response.text[:500]}...") # Log snippet
                return {"error": f"Backend response is not valid JSON.", "raw_response": response.text}
        else:
            return response.content # Return raw content for binary data like audio

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out: {method} {url}")
        return {"error": "API request timed out. The backend might be slow or unresponsive."}
    except requests.exceptions.ConnectionError:
        logger.error(f"API connection error: {method} {url}", exc_info=True) # Log connection error with traceback
        return {"error": "Could not connect to the backend API. Please check the backend URL and status."}
    except requests.exceptions.RequestException as e:
        # Catch all other requests.exceptions.RequestException (HTTP errors, etc.)
        logger.error(f"API request failed: {e}", exc_info=True) # Log error with traceback
        # Try to include backend error detail if available from the response
        detail = str(e) # Default detail is the exception message
        backend_detail = "N/A"
        if e.response is not None:
            detail = f"Status {e.response.status_code}"
            try:
                # Attempt to parse JSON detail from backend response body (FastAPI often returns {"detail": ...})
                if e.response.text:
                    backend_json = e.response.json()
                    # Prioritize 'detail' key from FastAPI, fallback to the whole JSON object
                    backend_detail = backend_json.get('detail', backend_json)
                    detail = f"Status {e.response.status_code} - {backend_detail}" # More informative error
                else:
                    backend_detail = "Empty response body"
                    detail = f"Status {e.response.status_code} - Empty response body"
            except json.JSONDecodeError:
                # If response is not JSON, include raw text response body
                backend_detail = e.response.text
                detail = f"Status {e.response.status_code} - Response Text: {backend_detail[:500]}..." # Log snippet
            except Exception as json_e:
                # Catch any other errors during response parsing
                logger.warning(f"Could not parse backend error response: {json_e}", exc_info=True)
                backend_detail = e.response.text or "N/A"
                detail = f"Status {e.response.status_code} - Could not parse response."


        return {"error": f"API request failed: {detail}", "backend_detail": backend_detail}

# === Audio Processing Helper ===
def frames_to_wav_bytes(frames):
    """Converts a list of audio frames (av.AudioFrame) to WAV formatted bytes."""
    if not frames:
        logger.warning("No audio frames provided for WAV conversion.")
        return None

    # Use audio format information stored in session state from the first frame received
    # Use a try-except block in case audio_format is missing or incomplete
    try:
        audio_format = st.session_state['audio_format']
        sample_rate = audio_format['sample_rate']
        format_name = audio_format['format_name']
        channels = audio_format['channels']
        sample_width_bytes = audio_format['sample_width_bytes']
        layout_name = audio_format.get('layout_name', 'Unknown Layout') # Use get for safety
    except KeyError as e:
        logger.error(f"Missing audio format information in session state: {e}. Cannot convert to WAV.")
        return None
    except Exception as e:
         logger.error(f"Error accessing audio format from session state: {e}", exc_info=True)
         return None


    logger.info(f"Converting {len(frames)} audio frames to WAV. Format: {format_name}, Channels: {channels}, Sample Rate: {sample_rate}, Sample Width: {sample_width_bytes} bytes, Layout: {layout_name}.")

    raw_data = b"" # Initialize raw data buffer
    try:
        # Handle common interleaved format (s16)
        if 's16' in format_name and layout_name.lower() in ['mono', 'stereo']:
            # For s16 interleaved, data for all channels is in the first plane (index 0).
            # Ensure planes[0].buffer exists before tobytes()
            raw_data = b"".join([frame.planes[0].buffer.tobytes() for frame in frames if frame.planes and frame.planes[0].buffer])
            logger.info(f"Concatenated raw bytes from s16 frames, total size: {len(raw_data)} bytes.")
        # Handle common planar formats (s32p, f32p)
        elif 's32p' in format_name or 'f32p' in format_name:
            # Planar formats: data for each channel is in a separate plane (plane[i] for channel i).
            # Need to concatenate data for each channel across all frames, then interleave.
            logger.info(f"Processing planar audio format: {format_name}")
            all_channels_data = []
            for i in range(channels):
                 # Concatenate data for each channel across all frames
                 channel_data = np.concatenate([frame.planes[i].to_ndarray() for frame in frames if frame.planes and len(frame.planes) > i])
                 all_channels_data.append(channel_data)

            if not all_channels_data:
                 logger.warning("No channel data extracted from planar frames.")
                 return None

            # Stack channel data (e.g., [samples_ch1], [samples_ch2]) -> [[s1_ch1, s1_ch2], [s2_ch1, s2_ch2], ...]
            interleaved_data = np.stack(all_channels_data, axis=-1)
            raw_data = interleaved_data.tobytes() # Convert the stacked NumPy array to bytes.
            logger.info(f"Interleaved planar data, resulting in {len(raw_data)} bytes.")
        else:
            # Log unsupported formats
            logger.error(f"Unsupported audio format or layout for WAV conversion: {format_name}, {layout_name}. Supported: s16, s32p, f32p (mono/stereo).")
            return None
    except Exception as e:
        logger.error(f"Error processing audio frame bytes for conversion: {e}", exc_info=True)
        return None

    # Ensure raw_data is not empty before attempting WAV creation
    if not raw_data:
        logger.warning("No raw audio data generated from frames.")
        return None

    # Create a WAV file header and write the raw data into a bytes buffer in memory
    try:
        with io.BytesIO() as wav_buffer:
            # Use wave.open to write WAV format data
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width_bytes)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_data) # Write the processed raw audio data
            wav_bytes = wav_buffer.getvalue() # Get the complete WAV file bytes
            logger.info(f"Successfully created WAV data of size {len(wav_bytes)} bytes.")
            return wav_bytes
    except Exception as e:
        logger.error(f"Error creating WAV file in memory: {e}", exc_info=True)
        return None

# === WebRTC Audio Frame Callback ===
# This function is called by the streamlit-webrtc thread for each incoming audio frame.
# It must be thread-safe.
def audio_frame_callback(frame: av.AudioFrame):
    """Callback function to receive and process audio frames from the browser."""
    # Acquire the lock before accessing the session state buffer
    with st.session_state.audio_buffer_lock:
        # Only append frames if recording is active
        if st.session_state.get('is_recording', False):
            st.session_state.audio_buffer.append(frame)

            # Store format info from the first frame if not already stored
            # This format info is needed later for WAV conversion
            if 'audio_format' not in st.session_state and st.session_state.audio_buffer:
                 frame_0 = st.session_state.audio_buffer[0]
                 st.session_state.audio_format = {
                     'sample_rate': frame_0.sample_rate,
                     'format_name': frame_0.format.name,
                     'channels': frame_0.layout.channels,
                     'sample_width_bytes': frame_0.format.bytes,
                     'layout_name': frame_0.layout.name
                 }
                 logger.info(f"Stored audio format in session state from first frame: {st.session_state.audio_format}")

            # Optional: Log frequency can be controlled by logger level
            # logger.debug(f"Audio frame received: {len(st.session_state.audio_buffer)} frames buffered.")


# === Main Application Layout and Logic ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

# A general placeholder at the top. Can be used for global messages if needed,
# but tab-specific status updates use placeholders defined within their tabs.
# status_placeholder_global = st.empty()

# === Sidebar for GitHub Integration ===
st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repo",
    key="sidebar_github_url" # Unique key
)
# Validate GitHub URL format if input is provided
if github_url:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", github_url)
    if match:
        owner, repo = match.groups()
        st.sidebar.success(f"**Repository:** {repo}\n\n(Owner: {owner})")
    else:
        st.sidebar.error("Invalid GitHub URL format. Expected format: https://github.com/owner/repo")

# === Application Tabs ===
# Define the list of tab names
tab_names = ["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow Trigger", "🔍 Workflow Status", "📈 Metrics", "🎤 Voice Agent"]
# Create the tabs
tabs = st.tabs(tab_names)
# Assign each tab object to a descriptive variable
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow_trigger, tab_status, tab_metrics, tab_voice = tabs

# Define a mapping for common file extensions to code languages for highlighting
language_map = {
    "py": "python", "java": "java", "js": "javascript",
    "cpp": "cpp", "c": "c", "txt": "plaintext", "log": "plaintext",
    "diff": "diff", "patch": "diff" # Add diff/patch for the QA tab
}

# Helper to guess language based on file name or type
def guess_language(file_details):
    """Guesses the language for code highlighting based on file details."""
    if file_details is None:
        return "plaintext"

    # Prioritize file extension
    file_extension = os.path.splitext(file_details.name)[1].lstrip('.').lower()
    if file_extension in language_map:
        return language_map[file_extension]

    # Fallback to MIME type if extension is not mapped
    mime_type_language = file_details.type.split('/')[-1].lower()
    if mime_type_language in language_map:
        return language_map[mime_type_language]

    # Default to plaintext if neither matches
    return "plaintext"


# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    st.write("Upload a traceback or source file to analyze and get a suggested patch.")
    # Allow relevant file types for analysis
    uploaded_file_trace = st.file_uploader(
        "Upload Traceback or Source File",
        type=["txt", "py", "java", "js", "cpp", "c", "log"], # Supported code/log types
        key="trace_file_uploader" # Unique key
    )

    file_content_trace = None # Initialize file content variable for this tab
    if uploaded_file_trace:
        try:
            file_content_trace = uploaded_file_trace.read().decode("utf-8")
            st.subheader("Original Content")
            # Guess language for syntax highlighting
            original_language_trace = guess_language(uploaded_file_trace)
            st.code(file_content_trace, language=original_language_trace, height=300)

        except Exception as e:
             st.error(f"An error occurred reading the file: {e}")
             logger.exception("Error reading uploaded file in Traceback tab.")
             file_content_trace = None # Ensure content is None if reading failed


    # Button to trigger analysis and patch suggestion
    if st.button("🔬 Analyze & Suggest Patch", key="analyze_patch_btn", disabled=file_content_trace is None): # Disable if no file content
        if file_content_trace: # Double-check content exists
            with st.spinner("Analyzing and suggesting patch..."):
                # Construct payload based on backend's AnalyzeRequest model
                # Ensure endpoint key matches ENDPOINTS
                payload = {
                    "code": file_content_trace,
                    "language": original_language_trace, # Use detected language
                    # "context": {} # <--- Add optional context like traceback lines if your backend needs it
                }
                # Call the backend API using the helper function
                response = make_api_request("POST", "suggest_patch", payload)

            if "error" not in response:
                # Assume backend returns 'diff' (string) and 'explanation' (string) keys
                suggested_diff = response.get("diff", "No diff suggested.")
                explanation = response.get("explanation", "No explanation provided.")

                st.subheader("Suggested Patch")
                # Display diff with 'diff' language highlighting
                st.code(suggested_diff, language="diff", height=300)
                st.markdown(f"💡 **Explanation:** {explanation}")

                # Code Editor for Editing Content - Initialize with ORIGINAL content for editing
                st.markdown("### ✍️ Edit Content")
                edited_content = st_ace(
                    value=file_content_trace, # Start with the original file content
                    language=original_language_trace, # Use original language for editing
                    theme="monokai",
                    height=350,
                    key="ace_editor_patch" # Unique key
                )

                # Diff View (Original vs. Edited Content)
                st.markdown("### 🔍 Diff View (Original vs. Edited Content)")
                # Check if both original content and edited content are available
                if edited_content is not None and file_content_trace is not None:
                    # Generate diff as HTML using difflib
                    diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
                        fromlines=file_content_trace.splitlines(), # Split lines for difflib
                        tolines=edited_content.splitlines(),     # Split lines for difflib
                        fromdesc=f"Original Content ({uploaded_file_trace.name})",
                        todesc="Edited Content",
                        context=True # Show surrounding lines for context
                    )
                    # Display HTML in Streamlit using components.v1.html
                    st.components.v1.html(diff_html, height=400, scrolling=True) # Adjusted height

                else:
                    st.info("Upload original content and edit it above to see the diff.")

            else:
                # Display error from make_api_request helper
                st.error(response["error"])
                # Optionally display backend detail if available
                if "backend_detail" in response and response["backend_detail"] not in ["N/A", "", response["error"].split(" - ", 1)[-1]]:
                     st.json({"Backend Detail": response["backend_detail"]})
        else:
             st.warning("Please upload a file before analyzing.")


# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    st.write("Upload a patch file to run QA validation checks.")
    uploaded_patch = st.file_uploader(
        "Upload Patch File",
        type=["txt", "diff", "patch"], # Standard patch file types
        key="qa_patch_uploader" # Unique key
    )

    patch_content = None # Initialize patch content variable for this tab
    if uploaded_patch:
        try:
            patch_content = uploaded_patch.read().decode("utf-8")
            st.subheader("Patch Content")
            # Use 'diff' language for highlighting patch files
            st.code(patch_content, language="diff", height=200)
        except Exception as e:
             st.error(f"An error occurred reading the patch file: {e}")
             logger.exception("Error reading uploaded patch file in QA tab.")
             patch_content = None

    # Button to trigger QA validation
    if st.button("🛡️ Validate Patch", key="qa_validate_btn", disabled=patch_content is None): # Disable if no patch content
        if patch_content: # Double-check content exists
            with st.spinner("Running QA validation..."):
                # Construct payload - Assuming backend expects 'patch_diff' key with the patch content
                # Ensure endpoint key matches ENDPOINTS
                payload = {"patch_diff": patch_content}
                # Call the backend API using the helper function
                response = make_api_request("POST", "qa_validation", payload)

            if "error" not in response:
                st.subheader("Validation Results")
                # Display validation results - Assuming response is a JSON summary/report
                st.json(response)
            else:
                # Display error from make_api_request helper
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
             st.warning("Please upload a patch file first to validate.")


# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    st.write("Upload a code file to generate documentation automatically.")
    uploaded_code_doc = st.file_uploader(
        "Upload Code File for Documentation",
        type=["txt", "py", "java", "js", "cpp", "c"], # Code file types
        key="doc_code_uploader" # Unique key
    )

    code_content_doc = None # Initialize code content variable for this tab
    if uploaded_code_doc:
        try:
            code_content_doc = uploaded_code_doc.read().decode("utf-8")
            st.subheader("Code Content")
            # Guess language for syntax highlighting
            doc_language = guess_language(uploaded_code_doc)
            st.code(code_content_doc, language=doc_language, height=200)
        except Exception as e:
             st.error(f"An error occurred reading the code file: {e}")
             logger.exception("Error reading uploaded code file in Documentation tab.")
             code_content_doc = None


    # Button to trigger documentation generation
    if st.button("📝 Generate Documentation", key="doc_generate_btn", disabled=code_content_doc is None): # Disable if no code content
        if code_content_doc: # Double-check content exists
            with st.spinner("Generating documentation..."):
                # Construct payload - Assuming backend needs 'code' and 'language' keys
                # Ensure endpoint key matches ENDPOINTS
                payload = {"code": code_content_doc, "language": doc_language}
                # Call the backend API using the helper function
                response = make_api_request("POST", "doc_generation", payload)

            if "error" not in response:
                st.subheader("Generated Documentation")
                # Assuming the response contains a "documentation" key with markdown or text
                st.markdown(response.get("documentation", "No documentation generated."))
            else:
                # Display error from make_api_request helper
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
             st.warning("Please upload a code file to generate documentation.")

# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues Inbox")
    st.write("This section lists issues needing attention from the autonomous workflow, typically those that encountered errors or require manual review.")

    # Button to refresh the list of issues
    if st.button("🔄 Refresh Issues", key="issues_refresh_btn"):
        with st.spinner("Fetching issues..."):
            # Call the backend API using the helper function (GET request, no payload)
            # Ensure endpoint key matches ENDPOINTS
            response = make_api_request("GET", "issues_inbox")

        if "error" not in response:
            # Assume backend returns {"issues": [...]} where [...] is a list of issue objects
            issues = response.get("issues", []) # Default to empty list if key missing or empty

            if issues:
                st.subheader(f"Issues Needing Attention ({len(issues)})")
                # Iterate and display each issue in an expander
                # Assume each issue object has 'id', 'status', and 'error_message' keys
                for issue in issues:
                    issue_id_display = issue.get('id', 'N/A')
                    issue_status_display = issue.get('status', 'Unknown Status')
                    issue_error_display = issue.get('error_message', 'No error details provided.')
                    issue_details_display = issue.get('details', {}) # Optional additional details

                    with st.expander(f"Issue ID: **{issue_id_display}** - Status: **{issue_status_display}**"):
                        st.write(f"**Error/Status Details:** {issue_error_display}")
                        # Display optional details if available
                        if issue_details_display:
                            st.write("**Additional Details:**")
                            st.json(issue_details_display)
            else:
                st.info("No issues needing attention found.")
        else:
            # Display error from make_api_request helper
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


# === Workflow Tab (Trigger) ===
with tab_workflow_trigger:
    st.header("🤖 Autonomous Workflow Trigger")
    st.write("Trigger an autonomous workflow run for a specific issue ID.")

    # Text input for the issue ID to trigger
    # Using session state directly via key="workflow_trigger_issue_id"
    st.text_input(
        "Issue ID to Trigger Workflow",
        placeholder="e.g., BUG-123",
        key="workflow_trigger_issue_id" # Unique key, links to st.session_state.workflow_trigger_issue_id
    )
    # Access the value directly from session state
    issue_id_trigger_value = st.session_state.workflow_trigger_issue_id

    # Button to trigger the workflow
    # Disable the button if the input is empty
    if st.button("▶️ Trigger Workflow", key="workflow_trigger_btn", disabled=not issue_id_trigger_value):
        if issue_id_trigger_value: # Double-check value exists
            with st.spinner(f"Triggering workflow for issue **{issue_id_trigger_value}**..."):
                # Construct payload - Assuming backend expects 'issue_id' key
                # Ensure endpoint key matches ENDPOINTS
                payload = {"issue_id": issue_id_trigger_value}
                # Call the backend API using the helper function
                response = make_api_request("POST", "workflow_run", payload)

            if "error" not in response:
                st.success(f"Workflow triggered successfully for Issue **{issue_id_trigger_value}**.")
                # Display response details - Assuming backend returns a confirmation message or details
                st.json(response.get('details', {'message': response.get('message', 'No specific details returned.')}))

                # Store the triggered issue_id and reset status flags to enable polling in the Status tab
                st.session_state.active_issue_id = issue_id_trigger_value
                st.session_state.workflow_completed = False # Indicate workflow is now running
                st.session_state.last_status = None # Clear last status
                st.session_state.error_message = None # Clear last error message

                # Rerun the app to immediately switch to or update the Status tab (optional, can manually switch)
                # st.rerun() # Commented out to avoid automatic tab switching

            else:
                # Display error from make_api_request helper
                st.error(response["error"])
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        # The disabled state of the button already handles the "Please enter an Issue ID." case


# === Workflow Check Tab (Status/Polling) ===
with tab_status:
    st.header("🔍 Autonomous Workflow Status")
    st.write("Check the live status of an ongoing workflow.")

    # Define a placeholder specifically for the status messages within this tab
    status_placeholder_workflow = st.empty()

    # Add a text input to manually set the issue ID to poll
    issue_id_for_polling_manual = st.text_input(
        "Enter Issue ID to check status manually:",
        placeholder="e.g., BUG-123",
        key="workflow_status_issue_id_manual" # Unique key
    )

    # Logic to determine which issue ID to track: manual input overrides triggered ID
    current_active_id_in_state = st.session_state.get("active_issue_id")

    if issue_id_for_polling_manual and current_active_id_in_state != issue_id_for_polling_manual:
        # Manual input changed and is different from current active ID
        st.session_state.active_issue_id = issue_id_for_polling_manual
        st.session_state.workflow_completed = False # Assume not completed if manually set
        st.session_state.last_status = None # Reset last status
        st.session_state.error_message = None # Reset error message
        st.rerun() # Rerun to start polling for the new manual ID
    elif not current_active_id_in_state and st.session_state.get("workflow_trigger_issue_id"):
         # No active ID set, but an ID was entered in the trigger tab; use that one
         st.session_state.active_issue_id = st.session_state.workflow_trigger_issue_id
         # No need to reset status flags here, they should be set by the trigger button press


    # Display the issue ID currently being tracked
    display_issue_id = st.session_state.get("active_issue_id") or "None (Trigger workflow or enter ID above)"
    st.markdown(f"Checking status for Issue ID: **{display_issue_id}**")

    # Progress tracker labels and mapping
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed", # Example status name - ADJUST BASED ON BACKEND
        "📦 PR Created" # Example status name - ADJUST BASED ON BACKEND
    ]
    # Map backend status strings to progress steps (0-indexed).
    # !!! IMPORTANT: These strings MUST exactly match the statuses returned by your backend's status endpoint !!!
    progress_map = {
        "Seeded": 0, # Assuming Seeded is initial status from backend
        "Fetching Details": 0,
        "Diagnosis in Progress": 1,
        "Diagnosis Complete": 1,
        "Patch Suggestion in Progress": 2,
        "Patch Suggestion Complete": 2,
        "Patch Validation in Progress": 3,
        "Patch Validated": 4,
        "PR Creation in Progress": 5,
        "PR Created - Awaiting Review/QA": 5 # Final success status
    }
    # Define terminal status strings - MUST match backend statuses
    terminal_status_success = "PR Created - Awaiting Review/QA" # Final successful state
    terminal_status_failed = "Workflow Failed" # Final failed state


    def show_agent_progress(status):
        """Displays the workflow progress bar and step labels based on the current status."""
        # Determine the current step index based on the status map
        step = progress_map.get(status, -1) # Default to -1 if status is not in map

        # Calculate progress value for the progress bar (0 to 1)
        # Handle the case where status is failed or not recognized gracefully
        if status == terminal_status_failed or step == -1:
            progress_value = 0 # Or a small value to indicate start/unknown
        else:
            # Progress goes from 0 (before step 0) to 1 (at or after the last step)
            progress_value = min(step + 1, len(progress_labels)) / len(progress_labels)

        st.progress(progress_value) # Display the progress bar

        # Display the step labels with appropriate icons
        for i, label in enumerate(progress_labels):
            icon = "⏳" # Default icon for future steps
            if status == terminal_status_failed:
                # If failed, mark the step where it failed with ❌, others are just text
                # Assuming the last mapped step is where it failed, or just mark all
                 icon = "❌" if i <= step else " " # Mark current/past steps with X on failure
            elif status == terminal_status_success:
                 # If successful, all steps are complete
                 icon = "✅"
            elif i < step:
                # Steps before the current step are complete
                icon = "✅"
            elif i == step:
                # The current step is in progress
                icon = "🔄"

            st.markdown(f"{icon} {label}") # Display the labeled step

        # Display final status messages below the progress bar only when workflow is completed
        if st.session_state.get("workflow_completed", True):
             if status == terminal_status_failed:
                  st.error("❌ Workflow Failed")
                  if st.session_state.get('error_message'):
                       st.error(f"Details: {st.session_state.error_message}")
             elif status == terminal_status_success:
                  st.success("✅ DebugIQ agents completed full cycle.")
             elif status is not None and status != "Unknown": # Handle other potential terminal states not explicitly listed
                  st.info(f"Workflow finished with status: **{status}**")
             elif status == "Unknown":
                  st.info("Workflow status is unknown or not yet started.")


    # --- Polling Logic ---
    # Autorefresh triggers a rerun periodically IF an issue is active AND the workflow is NOT completed
    is_polling_active = st.session_state.get("active_issue_id") is not None and not st.session_state.get("workflow_completed", True)

    if is_polling_active:
        # Use a unique key for autorefresh based on the active_issue_id
        # This ensures a new autorefresh instance starts if the issue ID changes
        st_autorefresh(interval=3000, key=f"workflow-refresh-{st.session_state.active_issue_id}") # Rerun every 3 seconds
        logger.info(f"Autorefresh enabled for issue: {st.session_state.active_issue_id}")
    else:
        logger.info("Autorefresh disabled.")

    # --- Fetch and Display Status ---
    # This code block runs on every rerun (triggered by user interaction or autorefresh)
    # We only attempt to fetch status via API if polling is currently active.
    # If polling is not active (because workflow completed or no ID set), we display the final/idle state.
    if is_polling_active:
        try:
            # Call the backend API to get the latest status for the active issue ID
            # The make_api_request helper handles the URL formatting for workflow_status
            # Ensure endpoint key matches ENDPOINTS
            status_response = make_api_request("GET", "workflow_status")

            if "error" not in status_response:
                # Assume backend returns {"status": "...", "error_message": "..."}
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")

                # Update session state with the latest status and error message
                st.session_state.last_status = current_status
                st.session_state.error_message = error_message # Store error message if backend provides one

                # Update the dedicated status placeholder in this tab
                # Show live status while polling is active
                status_placeholder_workflow.info(f"🔁 Live Status: **{current_status}**")

                # Update progress display based on the current status
                show_agent_progress(current_status)

                # Check if workflow has reached a terminal state (success or failure)
                if current_status == terminal_status_success or current_status == terminal_status_failed:
                    st.session_state.workflow_completed = True # Set flag to stop polling
                    status_placeholder_workflow.empty() # Clear the live status message
                    logger.info(f"Workflow for issue {st.session_state.active_issue_id} reached terminal state: {current_status}. Polling stopped.")
                    # No st.rerun() needed here usually, as the state change will trigger a final render

            else:
                # Handle API request errors during polling (e.g., backend down, 404 for issue ID)
                st.error(status_response["error"])
                if "backend_detail" in status_response: st.json({"Backend Detail": status_response["backend_detail"]})
                # Stop polling on API errors
                st.session_state.workflow_completed = True
                st.session_state.last_status = terminal_status_failed # Indicate failure state
                st.session_state.error_message = status_response.get("error", "API Error during polling.")
                status_placeholder_workflow.empty() # Clear the live status message
                logger.error(f"API error while polling status for issue {st.session_state.active_issue_id}. Polling stopped.")
                # No st.rerun() needed here usually

        except Exception as e:
             # Catch unexpected errors during the polling process
             st.error(f"An unexpected error occurred during status polling: {e}")
             logger.exception(f"Unexpected error polling workflow status for issue {st.session_state.active_issue_id}.")
             st.session_state.workflow_completed = True # Stop polling
             st.session_state.last_status = terminal_status_failed # Indicate failure state
             st.session_state.error_message = str(e) # Store the error message
             status_placeholder_workflow.empty() # Clear the live status message
             # No st.rerun() needed here usually

    # --- Display Final/Idle State ---
    # This block runs when polling is not active (initial load, after workflow completed, or on error)
    # Display the appropriate status message using the dedicated placeholder
    else:
        if st.session_state.get("active_issue_id") is None:
             # No active issue ID set
             status_placeholder_workflow.info("Enter an Issue ID or trigger a workflow to see status.")
             # Ensure polling is off if no active ID
             st.session_state.workflow_completed = True
        elif st.session_state.get("workflow_completed", True):
            # Workflow was completed or failed, display the final state using the last known status
            status_placeholder_workflow.empty() # Ensure placeholder is empty before showing final state
            show_agent_progress(st.session_state.get("last_status")) # Show progress bar based on final status
            # The show_agent_progress function handles displaying the final success/failure text

        # If active_issue_id is set, but not polling (e.g., just entered manual ID, waiting for first poll)
        # The `is_polling_active` check handles the "🔁 Live Status" display.
        # This else block ensures something is shown when not actively polling.
        # The 'Idle' state from the voice tab doesn't apply here.
        # The initial info message should cover the case where active_issue_id is None.
        # If active_issue_id is set but workflow_completed is True, the elif above handles it.
        # No explicit action needed here if the logic above covers all states.
        pass


# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    st.write("View system and performance metrics for the DebugIQ backend.")

    # Button to fetch metrics
    if st.button("📊 Fetch Metrics", key="metrics_fetch_btn"):
        with st.spinner("Fetching system metrics..."):
            # Call the backend API using the helper function (GET request, no payload)
            # Ensure endpoint key matches ENDPOINTS
            response = make_api_request("GET", "system_metrics")

        if "error" not in response:
            st.subheader("Backend System Metrics")
            # Display metrics - Assuming response is a JSON dictionary of metrics
            st.json(response)
        else:
            # Display error from make_api_request helper
            st.error(response["error"])
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


# === DebugIQ Voice Agent Section ===
# Dedicated Section in its own tab
with tab_voice:
    st.header("🎤 DebugIQ Voice Agent")
    st.write("Interact with the DebugIQ agent using your voice or text. Ask about issues, workflows, or general coding help.")

    # Define a placeholder specifically for status messages in the voice tab
    status_placeholder_voice = st.empty()

    # Configure and initialize the WebRTC streamer for microphone input
    # Use direct arguments for rtc_configuration and media_stream_constraints
    webrtc_ctx = webrtc_streamer(
        key="voice_agent_streamer", # Unique key for the component
        mode=WebRtcMode.SENDONLY, # Send audio from browser to server
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # STUN server config
        media_stream_constraints={"audio": True, "video": False}, # Audio only, no video
        audio_frame_callback=audio_frame_callback, # The callback function defined above
        async_processing=True # Process frames asynchronously
    )

    # --- Display Voice Agent Status ---
    # Check if the webrtc context and state are available and the stream is playing
    is_webrtc_ready = webrtc_ctx is not None and hasattr(webrtc_ctx.state, 'playing')
    is_webrtc_playing = is_webrtc_ready and webrtc_ctx.state.playing

    # Update status placeholder based on recording state and webrtc status
    if st.session_state.is_recording:
         status_placeholder_voice.warning(f"Status: {st.session_state.recording_status}")
    elif st.session_state.recording_status != "Idle":
         # Show processing status
         status_placeholder_voice.info(f"Status: {st.session_state.recording_status}")
    elif is_webrtc_playing:
         status_placeholder_voice.success("Status: Microphone Ready")
    else:
        # Show idle or waiting status
        status_placeholder_voice.info("Status: Idle (Waiting for Microphone...)")


    # --- Recording Control Buttons ---
    col1, col2 = st.columns(2)

    with col1:
        # Start Recording button: disabled if already recording or microphone not ready
        if st.button("🔴 Start Recording", key="start_recording_btn", disabled=st.session_state.is_recording or not is_webrtc_playing):
            st.session_state.is_recording = True
            st.session_state.audio_buffer = [] # Clear the buffer for the new recording
            st.session_state.recording_status = "Recording..."
            # Status placeholder updated by the logic block above on rerun
            logger.info("Recording started.")


    with col2:
        # Stop Recording button: disabled if not currently recording
        if st.button("⏹️ Stop Recording", key="stop_recording_btn", disabled=not st.session_state.is_recording):
            st.session_state.is_recording = False # Stop the recording flag
            st.session_state.recording_status = "Processing Audio..."
            # Status placeholder updated by the logic block above on rerun
            logger.info("Recording stopped. Processing audio...")

            # Trigger a rerun immediately to process the audio buffer after stopping
            st.rerun()


    # --- Audio Processing Logic (runs after stop button triggers rerun) ---
    # Check if recording was just stopped and there's audio data to process
    # This block runs on a rerun *after* is_recording is set to False by the stop button
    if not st.session_state.is_recording and st.session_state.audio_buffer and st.session_state.recording_status == "Processing Audio...":
        logger.info(f"Processing {len(st.session_state.audio_buffer)} buffered audio frames.")
        # Use a spinner to indicate background processing
        with st.spinner("Converting audio and sending for transcription..."):
            # Convert audio frames to WAV bytes
            wav_bytes = frames_to_wav_bytes(st.session_state.audio_buffer)

            if wav_bytes:
                logger.info(f"Audio converted to WAV ({len(wav_bytes)} bytes). Sending for transcription.")
                # Encode WAV bytes to Base64 for sending to backend
                encoded_audio = base64.b64encode(wav_bytes).decode('utf-8')

                # --- Send audio to backend for transcription ---
                st.session_state.recording_status = "Sending for Transcription..."
                status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here
                transcribe_payload = {"audio_base64": encoded_audio}
                # Ensure endpoint key matches ENDPOINTS
                transcription_response = make_api_request("POST", "voice_transcribe", transcribe_payload)

                # Check if make_api_request returned an error dict or the expected JSON
                if "error" not in transcription_response:
                    transcribed_text = transcription_response.get("text", "Could not transcribe audio.")
                    st.session_state.recording_status = "Transcription Complete."
                    status_placeholder_voice.success(f"Transcription: {transcribed_text}") # Update status here

                    # Add user's voice input (transcribed text and original audio) to chat history
                    # Add a 'processed' flag for text inputs, though not strictly needed for voice
                    st.session_state.chat_history.append({"role": "user", "content": transcribed_text, "audio": wav_bytes, "processed": True}) # Mark voice as processed

                    # --- Send transcribed text to Gemini Chat ---
                    st.session_state.recording_status = "Sending to Gemini..."
                    status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here
                    with st.spinner("Getting response from Gemini..."):
                        # Send the transcribed text to Gemini. Backend interprets.
                        # Assumes backend /gemini/chat endpoint accepts {"text": "..."}
                        # Ensure endpoint key matches ENDPOINTS
                        gemini_payload = {"text": transcribed_text}
                        gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

                    if "error" not in gemini_response:
                        ai_response_text = gemini_response.get("text", "No response from AI.")
                        # Optional: Get audio response from TTS if backend provides it in base64
                        ai_response_audio_base64 = gemini_response.get("audio_base64")

                        ai_response_audio = None
                        if ai_response_audio_base64:
                            try:
                                 # Decode the base64 audio back to bytes
                                 ai_response_audio = base64.b64decode(ai_response_audio_base64)
                                 logger.info("Received AI audio response for voice query.")
                            except (base64.binascii.Error, TypeError) as e:
                                 logger.error(f"Failed to decode base64 audio from backend (voice query): {e}", exc_info=True)
                                 st.warning("Received audio response from backend, but failed to decode it.")

                        # Add AI response (text and optional audio) to chat history
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "content": ai_response_text,
                            "audio": ai_response_audio # Store raw bytes (None if decoding failed or no audio provided)
                        })

                        st.session_state.recording_status = "AI Response Received."
                        status_placeholder_voice.success(f"Status: {st.session_state.recording_status}") # Update status here

                        # Clear the audio buffer and format info after successful processing
                        st.session_state.audio_buffer = []
                        st.session_state.audio_format = None
                        logger.info("Audio buffer cleared after voice processing.")

                        # Trigger a rerun to display the new chat messages
                        st.rerun()

                    else:
                        # Handle Gemini Chat API error
                        st.error(f"Gemini Chat API Error: {gemini_response['error']}")
                        if "backend_detail" in gemini_response: st.json({"Backend Detail": gemini_response["backend_detail"]})
                        st.session_state.recording_status = "AI Request Failed."
                        status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                        # Add error message to chat history as an AI response for context
                        st.session_state.chat_history.append({
                            "role": "ai",
                            "content": f"Error: Could not get AI response. {gemini_response.get('error', 'Unknown error')}",
                            "audio": None
                        })
                        st.session_state.audio_buffer = [] # Clear buffer even on error
                        st.session_state.audio_format = None
                        st.rerun() # Rerun to display error in chat history


                else:
                    # Handle Transcription API error
                    st.error(f"Transcription API Error: {transcription_response['error']}")
                    if "backend_detail" in transcription_response: st.json({"Backend Detail": transcription_response["backend_detail"]})
                    st.session_state.recording_status = "Transcription Failed."
                    status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                    # Add error message to chat history as a user message indicating transcription failed
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"Error: Could not transcribe audio. {transcription_response.get('error', 'Unknown error')}",
                        "audio": None # No successful audio to store
                    })
                    st.session_state.audio_buffer = [] # Clear buffer even on error
                    st.session_state.audio_format = None
                    st.rerun() # Rerun to display error in chat history

            else:
                # Handle WAV conversion failure
                st.warning("Failed to convert recorded audio frames to WAV format.")
                st.session_state.recording_status = "Audio Processing Failed."
                status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                st.session_state.audio_buffer = [] # Clear buffer
                st.session_state.audio_format = None
                st.rerun() # Rerun to update UI state

        # After attempting processing (whether successful or not), reset status if not already a final state
        # This will happen on the rerun after processing
        # Check if recording is still off and buffer is clear, then reset status
        if not st.session_state.is_recording and not st.session_state.audio_buffer:
             if st.session_state.recording_status not in ["Transcription Complete.", "AI Response Received.", "Transcription Failed.", "AI Request Failed.", "Audio Processing Failed."]:
                 # Only reset if the status is still in a transient processing state
                 st.session_state.recording_status = "Idle"
                 status_placeholder_voice.info("Status: Idle") # Update to Idle


    # --- Text Chat Input (Alternative to Voice) ---
    st.markdown("---") # Separator
    st.subheader("Text Chat")
    # Text input for sending queries via text. Linked to session state.
    st.text_input("Type your query here:", key="text_chat_input")
    # Access the value directly from session state
    text_query_input_value = st.session_state.text_chat_input

    # Send Text Query button: disabled if input is empty
    send_text_button_text = st.button("Send Text Query", key="send_text_btn", disabled=not text_query_input_value)

    # If button is clicked and input has text
    if send_text_button_text and text_query_input_value:
        # Update status placeholder
        st.session_state.recording_status = "Processing Text Query..."
        status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here

        # Add user's text message to chat history immediately
        # Add a flag 'processed' to prevent reprocessing on subsequent reruns
        st.session_state.chat_history.append({"role": "user", "content": text_query_input_value, "processed": False})

        # Clear the text input box after sending for good UX
        st.session_state.text_chat_input = ""
        # Trigger a rerun to clear the input box and process the query in the next block
        st.rerun()

    # --- Process Text Query Logic (runs on rerun after button click) ---
    # This block checks the last message in chat history. If it's a user message
    # that hasn't been processed yet, it sends it to the backend.
    try:
        # Check if chat history is not empty, the last message is from the user, and hasn't been processed
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user" and not st.session_state.chat_history[-1].get("processed", True):
             user_text_to_process = st.session_state.chat_history[-1]["content"]
             logger.info(f"Processing text query: {user_text_to_process}")

             st.session_state.recording_status = "Sending Text to Gemini..."
             status_placeholder_voice.info(f"Status: {st.session_state.recording_status}") # Update status here

             # Use a spinner while waiting for the Gemini response
             with st.spinner("Getting response from Gemini..."):
                 # Send the text query to Gemini Chat endpoint
                 # Assumes backend /gemini/chat endpoint accepts {"text": "..."}
                 # Ensure endpoint key matches ENDPOINTS
                 gemini_payload = {"text": user_text_to_process}
                 gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

             # Mark the user message as processed regardless of API success/failure
             st.session_state.chat_history[-1]["processed"] = True

             if "error" not in gemini_response:
                 ai_response_text = gemini_response.get("text", "No response from AI.")
                 # Check for optional audio response from TTS
                 ai_response_audio_base64 = gemini_response.get("audio_base64")

                 ai_response_audio = None
                 if ai_response_audio_base64:
                     try:
                          # Decode base64 audio
                          ai_response_audio = base64.b64decode(ai_response_audio_base64)
                          logger.info("Received AI audio response for text query.")
                     except (base64.binascii.Error, TypeError) as e:
                          logger.error(f"Failed to decode base64 audio from backend (text query): {e}", exc_info=True)
                          st.warning("Received audio response from backend, but failed to decode it.")

                 # Add AI response (text and optional audio) to chat history
                 st.session_state.chat_history.append({
                     "role": "ai",
                     "content": ai_response_text,
                     "audio": ai_response_audio # Store raw bytes (None if decoding failed or no audio provided)
                 })
                 st.session_state.recording_status = "AI Response Received."
                 status_placeholder_voice.success(f"Status: {st.session_state.recording_status}") # Update status here
                 st.rerun() # Trigger rerun to display the new AI message in chat history

             else:
                 # Handle Gemini Chat API error
                 st.error(f"Gemini Chat API Error: {gemini_response['error']}")
                 if "backend_detail" in gemini_response: st.json({"Backend Detail": gemini_response["backend_detail"]})
                 st.session_state.recording_status = "AI Request Failed."
                 status_placeholder_voice.error(f"Status: {st.session_state.recording_status}") # Update status here
                 # Add an error message to chat history as an AI response for context
                 st.session_state.chat_history.append({
                      "role": "ai",
                      "content": f"Error: Could not get AI response. {gemini_response.get('error', 'Unknown error')}",
                      "audio": None
                 })
                 st.rerun() # Trigger rerun to display the error message in chat history

    except IndexError:
        # Catch potential IndexError if chat_history is empty or modified unexpectedly
        logger.warning("IndexError accessing chat history during text query processing. Skipping.")
    except Exception as e:
        # Catch any other unexpected errors during text query processing
        logger.exception(f"An unexpected error occurred during text query processing: {e}")
        st.error(f"An unexpected error occurred during text query processing: {e}")
        st.session_state.recording_status = "Text Processing Error."
        status_placeholder_voice.error(f"Status: {st.session_state.recording_status}")
        # Optionally add an error message to chat history here as well


    # --- Display Chat History ---
    st.markdown("---") # Separator before chat history
    st.subheader("Chat History")
    # Display chat messages in reverse order (latest first)
    # Iterate over a copy of the history to avoid issues if history is modified during iteration
    # Added key for chat messages to help Streamlit render updates efficiently
    for i, message in enumerate(reversed(st.session_state.chat_history.copy())):
        role = "🧑‍💻 User" if message["role"] == "user" else "🤖 AI"
        # Use a unique key for each chat message element for stable rendering
        message_key = f"chat_message_{len(st.session_state.chat_history) - 1 - i}"
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Display the text content
            # Play audio if available and not None
            if message.get("audio") is not None:
                 try:
                     # Assumes the audio bytes are in WAV format
                     st.audio(message["audio"], format='audio/wav', key=f"{message_key}_audio") # Unique key for audio player
                 except Exception as e:
                     st.warning(f"Could not play audio: {e}")
                     logger.error(f"Error playing audio in chat history: {e}", exc_info=True)
