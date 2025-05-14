# File: frontend/debugiq_dashboard.py

# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# Imports for Voice Agent section
import av # Required for processing audio frames from streamlit-webrtc
import numpy as np # Required for processing audio frames
import io # Required for in-memory WAV file creation
import wave # Required for WAV file creation
from streamlit_webrtc import webrtc_streamer, WebRtcMode # Also needed if keeping voice
import logging # Already imported by requests, but good to be explicit
import base64 # Needed for voice/image encoding
import re # Needed for GitHub URL parsing
import threading # Potentially needed for thread-safe buffer if issues arise
from urllib.parse import urljoin # Needed for constructing URLs robustly

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
    "qa_validation": "/qa/run", # Based on /qa prefix and @router.post("/run")
    "doc_generation": "/doc/generate", # Based on /doc prefix and @router.post("/generate")
    "issues_inbox": "/issues/attention-needed", # Based on no prefix and @router.get("/issues/attention-needed")
    "workflow_run": "/workflow/run_autonomous_workflow", # Based on /workflow prefix and @router.post("/run_autonomous_workflow")
    # Workflow status needs issue_id formatting
    "workflow_status": "/issues/{issue_id}/status", # Based on no prefix and @router.get("/issues/{issue_id}/status")
    "system_metrics": "/metrics/status", # Based on no prefix and @router.get("/metrics/status")
    # Paths for Voice/Gemini - CONFIRM THESE WITH YOUR BACKEND ROUTERS
    "voice_transcribe": "/voice/transcribe", # Example path - CHECK YOUR BACKEND
    "gemini_chat": "/gemini/chat", # Example path - CHECK YOUR BACKEND
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
def frames_to_wav_bytes(frames):
    """Converts a list of audio frames (av.AudioFrame) to WAV formatted bytes."""
    if not frames:
        return None

    logger.info(f"Attempting to convert {len(frames)} audio frames to WAV.")

    # Assume consistent format across frames
    try:
        frame_0 = frames[0]
        sample_rate = frame_0.sample_rate
        format_name = frame_0.format.name
        channels = frame_0.layout.channels
        sample_width_bytes = frame_0.format.bytes # Bytes per sample per channel
        logger.info(f"Detected audio format: {format_name}, channels: {channels}, sample_rate: {sample_rate}, sample_width: {sample_width_bytes} bytes.")
    except Exception as e:
        logger.error(f"Error accessing frame properties: {e}")
        return None

    # Check for common formats and convert to raw bytes
    # streamlit-webrtc typically provides s16, s32p, or f32p
    # s16 is signed 16-bit int, interleaved
    if 's16' in format_name and frame_0.layout.name in ['mono', 'stereo']:
        try:
            # For s16 interleaved, data is in the first plane. Concatenate raw bytes.
            all_bytes = b"".join([frame.planes[0].buffer.tobytes() for frame in frames])
            logger.info(f"Concatenated raw bytes from frames, total size: {len(all_bytes)} bytes.")
            raw_data = all_bytes
        except Exception as e:
             logger.error(f"Error concatenating s16 audio frame bytes: {e}")
             return None
    elif 's32p' in format_name or 'f32p' in format_name:
         # Planar formats: data for each channel is in a separate plane. Need to interleave.
         try:
             # Convert planes to numpy arrays and interleave
             all_channels_data = [np.concatenate([frame.planes[i].to_ndarray() for frame in frames]) for i in range(channels)]
             # Stack channel data (e.g., [samples_ch1], [samples_ch2]) -> [[s1_ch1, s1_ch2], [s2_ch1, s2_ch2], ...]
             interleaved_data = np.stack(all_channels_data, axis=-1)
             raw_data = interleaved_data.tobytes()
             logger.info(f"Interleaved planar data, resulting in {len(raw_data)} bytes.")
         except Exception as e:
             logger.error(f"Error processing planar audio frames: {e}")
             return None
    else:
        logger.error(f"Unsupported audio format or layout for WAV conversion: {format_name}, {frame_0.layout.name}. Support for s16, s32p, f32p (mono/stereo) implemented.")
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
# Moved this function definition to the top level to ensure it's defined before use
def audio_frame_callback(frame: av.AudioFrame):
    """Callback function to receive and process audio frames from the browser."""
    # Use a thread-safe buffer to handle audio frames received in a different thread
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []

    # Lock to ensure thread-safe access to session state from the callback thread
    if "audio_buffer_lock" not in st.session_state:
        st.session_state.audio_buffer_lock = threading.Lock()

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
                    'sample_width_bytes': frame_0.format.bytes
                }

            # Log the audio frame details for debugging purposes (optional, can be chatty)
            # print(f"Audio frame received: {len(st.session_state.audio_buffer)} frames buffered.")


# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

# Initialize session state for recording and chat history
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'audio_buffer_lock' not in st.session_state:
    st.session_state.audio_buffer_lock = threading.Lock() # Initialize the lock
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Store list of {"role": "user" or "ai", "content": "...", "audio": b"..."}
# No longer need last_audio_response as audio is stored in chat_history

# === Sidebar for GitHub Integration ===
st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo", key="sidebar_github_url")
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
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Status", "📈 Metrics"])
# Assign tabs to variables based on their index
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    # Updated file uploader types based on backend model
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"], key="trace_file_uploader")

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        # Display original code - using st.code is better for syntax highlighting
        st.subheader("Original Code")
        # Attempt to guess language from file type
        original_language = uploaded_file.type.split('/')[-1] if uploaded_file.type in ["py", "java", "js", "cpp", "c"] else "plaintext"
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
    uploaded_patch = st.file_uploader("Upload Patch File", type=["txt", "diff", "patch"], key="qa_patch_uploader")

    if uploaded_patch:
        patch_content = uploaded_patch.read().decode("utf-8")
        st.subheader("Patch Content")
        st.code(patch_content, language="diff", height=200)

    if st.button("🛡️ Validate Patch", key="qa_validate_btn"):
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
    uploaded_code = st.file_uploader("Upload Code File for Documentation", type=["txt", "py", "java", "js", "cpp", "c"], key="doc_code_uploader")

    if uploaded_code:
        code_content = uploaded_code.read().decode("utf-8")
        st.subheader("Code Content")
        doc_language = uploaded_code.type.split('/')[-1] if uploaded_code.type in ["py", "java", "js", "cpp", "c"] else "plaintext"
        st.code(code_content, language=doc_language, height=200)

    if st.button("📝 Generate Documentation", key="doc_generate_btn"):
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

    if st.button("🔄 Refresh Issues", key="issues_refresh_btn"):
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

    if st.button("▶️ Trigger Workflow", key="workflow_trigger_btn"):
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
        # Adjust icon logic: success up to current step, spinner for current, empty for future
        if status == failed_status:
            icon = "❌"
            st.markdown(f"{icon} Workflow Failed")
        else:
            st.progress((step + 1) / len(progress_labels))
            for i, label in enumerate(progress_labels):
                 icon = "✅" if i <= step else ("🔄" if i == step + 1 else "⏳") # Spinner for the *next* step
                 st.markdown(f"{icon} {label}")


    # Polling logic (autorefresh only if an issue is active and workflow not complete)
    if st.session_state.active_issue_id and not st.session_state.workflow_completed:
        logger.info(f"Polling status for issue: {st.session_state.active_issue_id}")
        # Autorefresh triggers a rerun every X milliseconds
        st_autorefresh(interval=2000, key=f"workflow-refresh-{st.session_state.active_issue_id}") # Unique key per issue


    # --- Fetch and Display Status ---
    # This code runs on every rerun, including those triggered by autorefresh
    if st.session_state.active_issue_id and not st.session_state.workflow_completed: # Only fetch if active and not completed yet
        try:
            # Use make_api_request with the endpoint key - special handling in make_api_request for formatting
            status_response = make_api_request("GET", "workflow_status") # <--- Use endpoint key

            if "error" not in status_response:
                # Assumes backend returns {"status": "...", "error_message": "..."}
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")

                st.session_state.last_status = current_status # Store last status
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

    else: # Display idle status if no issue ID is active or if workflow is completed
        if st.session_state.last_status: # Show final status if completed
             if st.session_state.last_status == terminal_status:
                 st.success("✅ Workflow completed.")
             elif st.session_state.last_status == failed_status:
                 st.error("❌ Workflow failed.")
                 # Display last known error message if available
                 if "error_message" in st.session_state:
                     st.error(f"Last recorded error: {st.session_state.error_message}")
             else: # Handle other non-polling terminal states
                 st.info(f"Workflow finished with status: **{st.session_state.last_status}**")
             # Optionally show final progress state
             show_agent_progress(st.session_state.last_status)

        else: # No active issue and no last status
            st.info("Enter an Issue ID or trigger a workflow to see status.")

        # Ensure polling is off if no active ID or workflow is completed
        st.session_state.workflow_completed = True # Explicitly set completed if no active ID


# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    st.write("View system and performance metrics for the DebugIQ backend.")

    if st.button("📊 Fetch Metrics", key="metrics_fetch_btn"):
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

# Example Placeholder for WebRTC Streamer component - needed for the callback to fire
# if st.session_state.get('is_recording', False) or True: # Ensure it renders when recording or always if needed
#     try:
#         ctx = webrtc_streamer(
#             key="voice_agent_streamer_bottom",  # Unique key
#             mode=WebRtcMode.SENDONLY,  # Send audio from browser to server
#             frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#             media_stream_constraints={"audio": True, "video": False},
#             audio_frame_callback=audio_frame_callback, # Make sure this callback is defined above
#         )
#         # Handle potential issues with ctx.state.playing if needed
#         # if ctx.state.playing and not st.session_state.get('webrtc_started', False):
#         #     st.session_state.webrtc_started = True
#         #     logger.info("WebRTC streamer started playing.")

#     except Exception as e:
#         st.error(f"Failed to initialize voice agent microphone: {e}")
#         logger.exception("Error initializing webrtc_streamer for Voice Agent")


# Add a simple text input as an alternative way to chat if mic is not preferred
st.markdown("---") # Separator before text input
text_query = st.text_input("Type your query here:", key="text_chat_input")
send_text_button = st.button("Send Text Query", key="send_text_btn")

if send_text_button and text_query:
    # Use make_api_request for the text query to the backend Gemini Chat endpoint
    st.session_state.recording_status = "Processing Text Query..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")

    user_text = text_query
    # Add user's text to chat history immediately
    st.session_state.chat_history.append({"role": "user", "content": user_text})


    # --- Send text query to Gemini Chat ---
    ai_response_text = ""
    ai_response_audio = None # To store TTS audio bytes

    st.session_state.recording_status = "Sending to Gemini..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")
    with st.spinner("Getting response from Gemini..."):
        # Send the text query to Gemini. Backend interprets.
        gemini_payload = {"text": user_text}
        # Use make_api_request with the endpoint key
        gemini_response = make_api_request("POST", "gemini_chat", gemini_payload) # <--- Use endpoint key


    if "error" not in gemini_response:
        ai_response_text = gemini_response.get("response", "No response from Gemini.")
        st.session_state.recording_status = "Gemini Response Received."

        # --- Generate TTS for AI response ---
        if ai_response_text:
             st.session_state.recording_status = "Generating Speech..."
             status_placeholder.info(f"Status: {st.session_state.recording_status}")
             with st.spinner("Generating AI speech..."):
                 tts_payload = {"text": ai_response_text}
                 # Request raw audio bytes (return_json=False)
                 tts_response_data = make_api_request("POST", "tts", tts_payload, return_json=False) # <--- Use endpoint key


             if not isinstance(tts_response_data, dict) or "error" not in tts_response_data:
                 # Assuming tts_response_data is the raw WAV bytes
                 ai_response_audio = tts_response_data
                 st.session_state.recording_status = "Speech Generated."
                 logger.info(f"Received TTS audio bytes, size: {len(ai_response_audio) if ai_response_audio else 0}")
             else:
                 ai_response_text += f"\n(TTS Error: {tts_response_data.get('error', 'Unknown TTS error')})"
                 st.session_state.recording_status = "TTS Error."

        else:
             st.session_state.recording_status = "No AI text response for speech."

    else:
        ai_response_text = f"Error from Gemini: {gemini_response['error']}"
        st.session_state.recording_status = "Gemini Error."


    # Add AI's response (text and potentially audio) to chat history
    if ai_response_text or ai_response_audio:
        # For text input, we don't have captured audio format info. Default sample rate is usually ok for st.audio.
        st.session_state.chat_history.append({"role": "ai", "content": ai_response_text, "audio": ai_response_audio})


    status_placeholder.info(f"Status: {st.session_state.recording_status}")

    # Clear the text input after sending
    st.session_state.text_chat_input = "" # Use the session state key
    # Trigger a rerun to update the chat history display (only if text input was used)
    st.rerun() # Rerun will happen anyway if button is pressed, but explicit for clarity


# === Debugging/Development Info (Optional) ===
# st.sidebar.markdown("---")
# st.sidebar.subheader("Debugging Info")
# st.sidebar.write(f"Backend URL: {BACKEND_URL}")
# st.sidebar.write(f"Active Issue ID: {st.session_state.get('active_issue_id', 'None')}")
# st.sidebar.write(f"Workflow Completed: {st.session_state.get('workflow_completed', 'False')}")
# st.sidebar.json(st.session_state)
