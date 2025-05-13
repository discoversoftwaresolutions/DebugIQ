# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
# Updated import: ClientSettings is no longer needed/active as a separate class for client config
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import base64
import re
import av # Required for processing audio frames from streamlit-webrtc
import numpy as np # Required for processing audio frames
import io # Required for in-memory WAV file creation
import wave # Required for WAV file creation
# import threading # Potentially needed for thread-safe buffer if issues arise

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
# Use environment variable for the backend URL, with a fallback
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app")
ENDPOINTS = {
    "suggest_patch": f"{BACKEND_URL}/debugiq/suggest_patch",
    "qa_validation": f"{BACKEND_URL}/qa/run_qa",
    "doc_generation": f"{BACKEND_URL}/doc/generate_doc",
    "issues_inbox": f"{BACKEND_URL}/issues_inbox",
    "workflow_run": f"{BACKEND_URL}/workflow/run",
    "workflow_status": f"{BACKEND_URL}/workflow/status",
    "system_metrics": f"{BACKEND_URL}/system_metrics",
    "voice_transcribe": f"{BACKEND_URL}/transcribe_audio", # Endpoint for audio transcription
    "gemini_chat": f"{BACKEND_URL}/gemini_chat", # Endpoint for Gemini chat
    "tts": f"{BACKEND_URL}/generate_tts", # Assuming a new TTS endpoint
}

# === Helper Functions ===
def make_api_request(method, url, payload=None, return_json=True):
    """Makes an API request to the backend."""
    try:
        logger.info(f"Making API request: {method} {url}")
        # Set a reasonable timeout for API calls
        response = requests.request(method, url, json=payload, timeout=60) # Increased timeout for potentially longer tasks
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
        return {"error": f"API request failed: {e}"}

# === Audio Processing Helper ===
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
    import threading

    # Use a thread-safe queue to handle audio frames
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []

    # Lock to ensure thread-safe access to session state
    lock = threading.Lock()

    with lock:
        if st.session_state.get('is_recording', False):
            # Append the audio frame to the session state's buffer
            st.session_state.audio_buffer.append(frame)

            # Store format info from the first frame if not already stored
            if 'audio_format' not in st.session_state:
                st.session_state.audio_format = {
                    'sample_rate': frame.sample_rate,
                    'format_name': frame.format.name,
                    'channels': frame.layout.channels,
                    'sample_width_bytes': frame.format.bytes
                }

            # Log the audio frame details for debugging purposes
            print(f"Audio frame received: {len(st.session_state.audio_buffer)} frames buffered.")

# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

# Initialize session state for recording and chat history
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Store list of {"role": "user" or "ai", "content": "...", "audio": b"..."}
# No longer need last_audio_response as audio is stored in chat_history

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
tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Check", "📈 Metrics"])
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics = tabs

# === Traceback + Patch Tab (Modified Window Size) ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"])

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        # Reduced height
        st.text_area("Original Code", value=file_content, height=150) # Reduced height, disabled=True is often redundant if not editable

        if st.button("🔬 Analyze & Suggest Patch"):
            with st.spinner("Analyzing and suggesting patch..."):
                payload = {"trace": file_content}
                response = make_api_request("POST", ENDPOINTS["suggest_patch"], payload)

            if "error" not in response:
                suggested_patch = response.get("suggested_patch", "")
                st.text_area("Suggested Patch (Read-Only)", value=suggested_patch, height=150, disabled=True) # Reduced height

                # Code Editor for Editing Patch
                st.markdown("### ✍️ Edit Suggested Patch")
                # Reduced height
                edited_patch = st_ace(
                    value=suggested_patch,
                    language="python",
                    theme="monokai",
                    height=250, # Reduced height
                    key="ace_editor_patch"
                )

                # Diff View
                st.markdown("### 🔍 Diff View (Original vs. Edited Patch)")
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
                    st.components.v1.html(diff_html, height=350, scrolling=True) # Reduced height slightly
            else:
                st.error(response["error"])

# === QA Validation Tab ===
with tab_qa:
    st.header("✅ QA Validation")
    uploaded_patch = st.file_uploader("Upload Patch File", type=["txt", "py", "java", "js"])

    if uploaded_patch:
        patch_content = uploaded_patch.read().decode("utf-8")
        st.text_area("Patch Content", value=patch_content, height=200, disabled=True)

    if st.button("🛡️ Validate Patch"):
        if uploaded_patch:
            with st.spinner("Running QA validation..."):
                payload = {"patch_code": patch_content}
                response = make_api_request("POST", ENDPOINTS["qa_validation"], payload)

            if "error" not in response:
                st.json(response) # Display the full response JSON
            else:
                st.error(response["error"])
        else:
            st.warning("Please upload a patch file first.")


# === Documentation Tab ===
with tab_doc:
    st.header("📘 Documentation Generation")
    uploaded_code = st.file_uploader("Upload Code File for Documentation", type=["txt", "py", "java", "js"])

    if uploaded_code:
        code_content = uploaded_code.read().decode("utf-8")
        st.text_area("Code Content", value=code_content, height=200, disabled=True)

    if st.button("📝 Generate Documentation"):
        if uploaded_code:
            with st.spinner("Generating documentation..."):
                payload = {"code": code_content}
                response = make_api_request("POST", ENDPOINTS["doc_generation"], payload)

            if "error" not in response:
                # Assuming the response contains a "documentation" key with markdown or text
                st.subheader("Generated Documentation")
                st.markdown(response.get("documentation", "No documentation generated."))
            else:
                st.error(response["error"])
        else:
            st.warning("Please upload a code file first.")

# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues Inbox")
    st.write("This section would typically list issues from a connected issue tracker or an internal inbox.")

    if st.button("🔄 Refresh Issues"):
        with st.spinner("Fetching issues..."):
            response = make_api_request("GET", ENDPOINTS["issues_inbox"])

        if "error" not in response:
            if response.get("issues"):
                st.subheader("Fetched Issues")
                for issue in response.get("issues", []):
                    with st.expander(f"Issue ID: {issue.get('id', 'N/A')} - {issue.get('title', 'No Title')}"):
                         st.json(issue)
            else:
                 st.info("No issues found in the inbox.")
        else:
            st.error(response["error"])

# === Workflow Tab ===
with tab_workflow:
    st.header("🤖 Autonomous Workflow Trigger")
    st.write("Trigger an autonomous workflow run for a specific issue.")
    issue_id = st.text_input("Issue ID to Trigger Workflow", placeholder="e.g., BUG-123")

    if st.button("▶️ Trigger Workflow"):
        if issue_id:
            with st.spinner(f"Triggering workflow for issue {issue_id}..."):
                payload = {"issue_id": issue_id}
                response = make_api_request("POST", ENDPOINTS["workflow_run"], payload)

            if "error" not in response:
                st.success(f"Workflow triggered successfully for Issue {issue_id}. Response: {response.get('message', 'No message.')}")
            else:
                st.error(response["error"])
        else:
            st.warning("Please enter an Issue ID.")

# === Workflow Check Tab ===
with tab_status:
    st.header("🔍 Autonomous Workflow Status")
    st.write("Check the status of running autonomous workflows.")

    if st.button("🔄 Refresh Workflow Status"):
        with st.spinner("Fetching workflow status..."):
            response = make_api_request("GET", ENDPOINTS["workflow_status"])

        if "error" not in response:
            st.subheader("Workflow Status")
            st.json(response)
        else:
            st.error(response["error"])

# === Metrics Tab ===
with tab_metrics:
    st.header("📈 System Metrics")
    st.write("View system and performance metrics for the DebugIQ backend.")

    if st.button("📊 Fetch Metrics"):
        with st.spinner("Fetching system metrics..."):
            response = make_api_request("GET", ENDPOINTS["system_metrics"])

        if "error" not in response:
            st.subheader("Backend System Metrics")
            st.json(response)
        else:
            st.error(response["error"])


# === DebugIQ Voice Agent Section (Dedicated Section at the bottom) ===
st.markdown("---") # Add a separator below the tabs
st.markdown("---") # Add another separator for clear visual distinction
st.header("🎙️ DebugIQ Voice Agent")
st.write("Interact conversationally with DebugIQ using your voice or text. Ask questions or give commands related to debugging tasks.")
st.write("You can ask things like: 'Analyze the traceback', 'Generate documentation for this code', or ask general programming questions.") # Guide the user

# Display chat history
chat_container = st.container(height=400) # Use a container for chat history with a fixed height and scroll
with chat_container:
    for message in st.session_state.chat_history:
        role = "👤 User" if message["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {message['content']}")
        # Add play button or automatically play AI audio response
        if message["role"] == "ai" and message.get("audio"):
             # Use a unique key for each audio player in the history - using content hash for stability
             # Ensure audio is bytes for hashing
             if isinstance(message['audio'], bytes):
                 audio_hash = base64.b64encode(message['audio']).decode('utf-8')[:10] # Simple hash for key
             else:
                 audio_hash = "error" # Indicate issue if not bytes

             try:
                 # Use the sample rate stored with the message if available, default to 44100
                 st.audio(message["audio"], format='audio/wav', sample_rate=message.get("sample_rate", 44100), key=f"audio_player_{audio_hash}")
             except Exception as e:
                 st.warning(f"Could not play audio: {e}") # Handle potential issues with st.audio


# Display recording status
status_placeholder = st.empty()
status_placeholder.info(f"Status: {st.session_state.recording_status}")

# Buttons to control recording
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("▶️ Start Recording", key="voice_start_btn", disabled=st.session_state.is_recording)
with col2:
    stop_button = st.button("⏹️ Stop Recording", key="voice_stop_btn", disabled=not st.session_state.is_recording)


# --- webrtc_streamer component ---
# This component needs to be rendered for the audio stream to be available.
# It runs in the background and provides audio frames via the callback.
# Adding try/except block for robustness during initialization
try:
    ctx = webrtc_streamer(
        key="voice_agent_streamer_bottom",  # Unique key for this component instance
        mode=WebRtcMode.SENDONLY,  # Send audio from browser to server
        # Configuration previously in client_settings is now top-level parameters:
        frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Client-side WebRTC config
        media_stream_constraints={"audio": True, "video": False},  # Media constraints (audio only)
        audio_frame_callback=audio_frame_callback,  # Updated: Callback for processing audio frames
    )
except Exception as e:
    st.error(f"Failed to initialize voice agent microphone: {e}")
    logger.exception("Error initializing webrtc_streamer for Voice Agent")
    ctx = None  # Set ctx to None if initialization fails

# Handle button clicks in the main Streamlit thread
if start_button:
    st.session_state.is_recording = True
    st.session_state.audio_buffer = [] # Clear buffer on start
    st.session_state.pop('audio_format', None) # Clear format info on start
    st.session_state.recording_status = "Recording..."
    # No clearing chat history here, allowing for conversation
    status_placeholder.info(f"Status: {st.session_state.recording_status}")


if stop_button:
    st.session_state.is_recording = False
    st.session_state.recording_status = "Processing..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")

    # Process the recorded audio after stopping
    if st.session_state.audio_buffer:
        logger.info(f"Processing {len(st.session_state.audio_buffer)} frames after stopping.")
        # Get audio format info recorded during capture
        audio_format_info = st.session_state.get('audio_format', {})
        # Convert frames to WAV bytes
        wav_data = frames_to_wav_bytes(st.session_state.audio_buffer)

        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            logger.info(f"Encoded audio data to Base64, size: {len(audio_base64)} bytes.")

            # --- Transcribe audio ---
            st.session_state.recording_status = "Transcribing..."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")
            with st.spinner("Transcribing audio..."):
                transcription_payload = {"audio_base64": audio_base64}
                transcription_response = make_api_request("POST", ENDPOINTS["voice_transcribe"], transcription_payload)

            user_text = "Could not transcribe audio."
            if "error" not in transcription_response:
                user_text = transcription_response.get("transcription", user_text)
                st.session_state.recording_status = "Transcription Complete."
            else:
                user_text = f"Transcription Error: {transcription_response['error']}"
                st.session_state.recording_status = "Transcription Error."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")

            # Add user's transcription to chat history
            # Only add non-empty transcription (excluding placeholder error)
            if user_text and not user_text.startswith("Transcription Error") and user_text.strip() != "" and user_text != "Could not transcribe audio.":
                 st.session_state.chat_history.append({"role": "user", "content": user_text})

            # --- Send transcription to Gemini Chat ---
            ai_response_text = ""
            ai_response_audio = None # To store TTS audio bytes

            # Only send to Gemini if transcription was successful and has meaningful content
            if user_text and not user_text.startswith("Transcription Error") and user_text.strip() != "" and user_text != "Could not transcribe audio.":
                st.session_state.recording_status = "Sending to Gemini..."
                status_placeholder.info(f"Status: {st.session_state.recording_status}")
                with st.spinner("Getting response from Gemini..."):
                    # Send the transcribed text to Gemini. The backend needs to interpret this.
                    gemini_payload = {"text": user_text}
                    gemini_response = make_api_request("POST", ENDPOINTS["gemini_chat"], gemini_payload)


                if "error" not in gemini_response:
                    ai_response_text = gemini_response.get("response", "No response from Gemini.")
                    st.session_state.recording_status = "Gemini Response Received."

                    # --- Generate TTS for AI response ---
                    if ai_response_text: # Only generate TTS if there's text
                         st.session_state.recording_status = "Generating Speech..."
                         status_placeholder.info(f"Status: {st.session_state.recording_status}")
                         with st.spinner("Generating AI speech..."):
                             tts_payload = {"text": ai_response_text}
                             # Request raw audio bytes (or Base64) from backend TTS endpoint
                             tts_response_data = make_api_request("POST", ENDPOINTS["tts"], tts_payload, return_json=False) # Expecting raw bytes

                         if not isinstance(tts_response_data, dict) or "error" not in tts_response_data:
                             # Assuming tts_response_data is the raw WAV bytes
                             ai_response_audio = tts_response_data
                             st.session_state.recording_status = "Speech Generated."
                             logger.info(f"Received TTS audio bytes, size: {len(ai_response_audio) if ai_response_audio else 0}")
                         else:
                             # Handle error if make_api_request returned an error dict
                             ai_response_text += f"\n(TTS Error: {tts_response_data.get('error', 'Unknown TTS error')})" # Append TTS error to text response
                             st.session_state.recording_status = "TTS Error."

                    else:
                         st.session_state.recording_status = "No AI text response for speech."


                else:
                    ai_response_text = f"Error from Gemini: {gemini_response['error']}"
                    st.session_state.recording_status = "Gemini Error."

            elif user_text.startswith("Transcription Error"):
                 ai_response_text = f"Could not process audio: {user_text}"
                 st.session_state.recording_status = "Processing failed."
            else:
                 # Case where transcription was empty or "Could not transcribe audio."
                 ai_response_text = "Please try speaking again."
                 st.session_state.recording_status = "Processing failed."


            # Add AI's response (text and potentially audio) to chat history
            if ai_response_text or ai_response_audio:
                # Include sample rate if known from audio format info for st.audio playback
                 ai_message = {"role": "ai", "content": ai_response_text, "audio": ai_response_audio}
                 if audio_format_info.get("sample_rate"):
                      ai_message["sample_rate"] = audio_format_info["sample_rate"]
                 st.session_state.chat_history.append(ai_message)


            status_placeholder.info(f"Status: {st.session_state.recording_status}")

        else:
            st.session_state.recording_status = "Failed to process audio."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")

    else:
        st.session_state.recording_status = "No audio recorded."
        status_placeholder.info(f"Status: {st.session_state.recording_status}")

    st.session_state.audio_buffer = [] # Clear buffer after processing
    st.session_state.pop('audio_format', None) # Clear audio format info after processing
    # Trigger a rerun to update the chat history display
    st.rerun()

# Add a simple text input as an alternative way to chat if mic is not preferred
st.markdown("---") # Separator before text input
text_query = st.text_input("Type your query here:", key="text_chat_input")
send_text_button = st.button("Send Text Query", key="send_text_btn")

if send_text_button and text_query:
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
        gemini_response = make_api_request("POST", ENDPOINTS["gemini_chat"], gemini_payload)


    if "error" not in gemini_response:
        ai_response_text = gemini_response.get("response", "No response from Gemini.")
        st.session_state.recording_status = "Gemini Response Received."

        # --- Generate TTS for AI response ---
        if ai_response_text:
             st.session_state.recording_status = "Generating Speech..."
             status_placeholder.info(f"Status: {st.session_state.recording_status}")
             with st.spinner("Generating AI speech..."):
                 tts_payload = {"text": ai_response_text}
                 # Request raw audio bytes
                 tts_response_data = make_api_request("POST", ENDPOINTS["tts"], tts_payload, return_json=False)

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
    st.rerun() # Trigger a rerun to update the chat history display
