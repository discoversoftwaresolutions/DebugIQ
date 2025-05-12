# dashboard.py

import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
# Make sure ClientSettings is imported
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import av
from difflib import HtmlDiff # difflib.HtmlDiff is already imported
import streamlit.components.v1 as components
import wave
import json
import logging
import base64 # For handling audio data if sent as base64

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_BACKEND_URL = "https://debugiq-backend.railway.app" # Your existing default
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)

# Voice Agent Specific Constants (add these if not already defined globally)
DEFAULT_VOICE_SAMPLE_RATE = 16000
DEFAULT_VOICE_SAMPLE_WIDTH = 2  # 16-bit audio (2 bytes)
DEFAULT_VOICE_CHANNELS = 1  # Mono
AUDIO_PROCESSING_THRESHOLD_SECONDS = 2 # Increased slightly for potentially more complete phrases

# --- Streamlit Config ---
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# --- Session State Initialization (ensure these for the voice agent) ---
if 'audio_sample_rate' not in st.session_state:
    st.session_state.audio_sample_rate = DEFAULT_VOICE_SAMPLE_RATE
if 'audio_sample_width' not in st.session_state:
    st.session_state.audio_sample_width = DEFAULT_VOICE_SAMPLE_WIDTH
if 'audio_num_channels' not in st.session_state:
    st.session_state.audio_num_channels = DEFAULT_VOICE_CHANNELS
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = b""
if 'audio_frame_count' not in st.session_state:
    st.session_state.audio_frame_count = 0
if 'chat_history' not in st.session_state: # For Gemini conversation
    st.session_state.chat_history = []
if 'edited_patch' not in st.session_state: # From your patch tab
    st.session_state.edited_patch = ""


# --- Tabs Setup ---
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

# --- API Endpoints ---
ANALYZE_URL = f"{BACKEND_URL}/debugiq/analyze" # You had /debugiq/suggest_patch, adjust if needed
QA_URL = f"{BACKEND_URL}/qa/"
DOC_URL = f"{BACKEND_URL}/doc/"
VOICE_TRANSCRIBE_URL = f"{BACKEND_URL}/voice/transcribe"
# VOICE_COMMAND_URL = f"{BACKEND_URL}/voice/command" # We might replace this with Gemini
GEMINI_CHAT_URL = f"{BACKEND_URL}/gemini-chat"  # << NEW ENDPOINT FOR GEMINI
ISSUES_INBOX_URL = f"{BACKEND_URL}/issues/inbox"
WORKFLOW_RUN_URL = f"{BACKEND_URL}/workflow/run"
WORKFLOW_CHECK_URL = f"{BACKEND_URL}/workflow/status" # You had /workflow/integrity-check, adjust if needed
METRICS_URL = f"{BACKEND_URL}/metrics/summary"


# --- Helper function (modified slightly for flexibility) ---
def make_api_request(method, url, json_payload=None, files=None, operation_name="API Call"):
    try:
        logger.info(f"Making {method} request to {url} for {operation_name}...")
        if files:
            response = requests.request(method, url, files=files, data=json_payload, timeout=30) # Adjusted for potential data with files
        else:
            response = requests.request(method, url, json=json_payload, timeout=30)
        
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Try to parse JSON, but handle cases where response might be empty or not JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.warning(f"{operation_name} response was not JSON. Status: {response.status_code}, Content: {response.text[:100]}")
            return {"status_code": response.status_code, "content": response.text} # Return raw content if not JSON

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error during {operation_name} to {url}: {http_err}. Response: {http_err.response.text if http_err.response else 'No response text'}")
        st.error(f"{operation_name} failed: {http_err}. Server said: {http_err.response.text if http_err.response else 'No details'}")
        return {"error": str(http_err), "details": http_err.response.text if http_err.response else "No details"}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"RequestException during {operation_name} to {url}: {req_err}")
        st.error(f"Error communicating with backend for {operation_name}: {req_err}")
        return {"error": str(req_err)}
    except Exception as e: # Catch-all for other unexpected errors
        logger.exception(f"Unexpected error during {operation_name} to {url}")
        st.error(f"An unexpected error occurred with {operation_name}: {e}")
        return {"error": str(e)}

# --- Traceback + Patch ---
with tabs[0]:
    st.header("📄 Traceback + Patch")
    uploaded_file = st.file_uploader("Upload traceback or .py file", type=["py", "txt"], key="patch_uploader")
    if uploaded_file:
        original_code = uploaded_file.read().decode("utf-8")
        st.text_area("Original Code", value=original_code, height=200, disabled=True, key="original_code_display") # Changed to text_area for consistency

        if st.button("🧠 Suggest Patch", key="suggest_patch_btn"):
            payload = {"code": original_code} # Assuming this is what your backend expects
            # The ANALYZE_URL was f"{BACKEND_URL}/debugiq/analyze", your button used /debugiq/suggest_patch
            # Using a specific URL for patch suggestion:
            suggest_patch_url = f"{BACKEND_URL}/debugiq/suggest_patch"
            patch_data = make_api_request("POST", suggest_patch_url, json_payload=payload, operation_name="Patch Suggestion")

            if patch_data and not patch_data.get("error"):
                patched_code = patch_data.get("patched_code", "")
                # patch_diff_text = patch_data.get("diff", "") # If backend provides text diff

                st.markdown("### 🔍 Diff View")
                # Generate diff if original and patched code are available
                if original_code and patched_code:
                    html_diff_generator = difflib.HtmlDiff(wrapcolumn=70)
                    html_diff_output = html_diff_generator.make_file(
                        original_code.splitlines(keepends=True),
                        patched_code.splitlines(keepends=True),
                        fromdesc="Original",
                        todesc="Patched"
                    )
                    st.components.v1.html(html_diff_output, height=450, scrolling=True)
                else:
                    st.info("Could not generate diff (missing original or patched code).")

                st.markdown("### ✍️ Edit Patch (Live)")
                edited_code = st_ace(value=patched_code, language="python", theme="monokai", height=300, key="patch_editor_ace")
                st.session_state.edited_patch = edited_code
            else:
                st.error(f"Patch generation failed. Response: {patch_data.get('details') if patch_data else 'No response'}")


# --- QA Validation --- (Using make_api_request for consistency)
with tabs[1]:
    st.header("✅ QA Validation")
    qa_code_input = st.text_area("Paste updated code for validation:", key="qa_code_input_area", height=200)
    # You might want to use st.session_state.edited_patch here if available
    if st.session_state.get("edited_patch"):
         st.info("Consider using the edited patch from the 'Traceback + Patch' tab for QA.")
         if st.button("Use Edited Patch for QA", key="use_edited_for_qa"):
             qa_code_input = st.session_state.edited_patch # This won't directly update the widget, need st.rerun or callback
             st.session_state.qa_code_to_validate = st.session_state.edited_patch # Store it
             st.rerun() # To make the text_area update with the new value

    # If qa_code_to_validate is set, use it, otherwise use the text_area content directly
    code_for_qa = st.session_state.get("qa_code_to_validate", qa_code_input)


    if st.button("Run QA Validation", key="run_qa_btn"):
        if code_for_qa:
            payload = {"code": code_for_qa}
            qa_result = make_api_request("POST", QA_URL, json_payload=payload, operation_name="QA Validation")
            if qa_result and not qa_result.get("error"):
                st.json(qa_result)
            else:
                st.error(f"QA Validation failed. Response: {qa_result.get('details') if qa_result else 'No response'}")
        else:
            st.warning("Please paste some code or use the edited patch for QA.")
    if "qa_code_to_validate" in st.session_state: # Clear after use if desired
        del st.session_state.qa_code_to_validate


# --- Documentation --- (Using make_api_request)
with tabs[2]:
    st.header("📘 Documentation")
    doc_code_input = st.text_area("Paste code to generate documentation:", key="doc_code_input_area", height=200)
    if st.button("📝 Generate Code Documentation", key="generate_doc_btn"): # Changed button label slightly
        if doc_code_input:
            payload = {"code": doc_code_input}
            doc_result = make_api_request("POST", DOC_URL, json_payload=payload, operation_name="Documentation Generation")
            if doc_result and not doc_result.get("error"):
                st.markdown(doc_result.get("doc", "No documentation generated or key 'doc' missing."))
            else:
                st.error(f"Documentation generation failed. Response: {doc_result.get('details') if doc_result else 'No response'}")
        else:
            st.warning("Please paste some code to generate documentation.")

# --- Issue Notices --- (Using make_api_request)
with tabs[3]:
    st.header("📣 Detected Issues (Autonomous Agent Summary)")
    if st.button("🔍 Fetch Notices", key="fetch_issues_btn"):
        issues_data = make_api_request("GET", ISSUES_INBOX_URL, operation_name="Fetch Issues")
        if issues_data and not issues_data.get("error"):
            st.json(issues_data) # Assuming issues_data is the JSON itself
        else:
            st.error(f"Issue data fetch failed. Response: {issues_data.get('details') if issues_data else 'No response'}")

# --- Autonomous Workflow --- (Using make_api_request)
with tabs[4]:
    st.header("🤖 Run DebugIQ Autonomous Workflow")
    issue_id = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101", key="workflow_issue_id_input")
    if st.button("▶️ Run Workflow", key="run_workflow_btn"):
        if issue_id:
            payload = {"issue_id": issue_id}
            workflow_result = make_api_request("POST", WORKFLOW_RUN_URL, json_payload=payload, operation_name="Workflow Run")
            if workflow_result and not workflow_result.get("error"):
                st.success(f"Workflow triggered for {issue_id}.")
                st.json(workflow_result)
            else:
                st.error(f"Workflow execution failed for {issue_id}. Response: {workflow_result.get('details') if workflow_result else 'No response'}")
        else:
            st.warning("Please enter an Issue ID.")

# --- Workflow Check --- (Using make_api_request)
with tabs[5]:
    st.header("🔍 Workflow Status Check") # Changed header slightly for clarity
    if st.button("🔄 Refresh Workflow Status", key="refresh_workflow_status_btn"): # Added a button to fetch
        status_data = make_api_request("GET", WORKFLOW_CHECK_URL, operation_name="Workflow Status Check")
        if status_data and not status_data.get("error"):
            st.json(status_data)
        else:
            st.error(f"Workflow status check failed. Response: {status_data.get('details') if status_data else 'No response'}")
    else:
        st.info("Click the button to fetch the latest workflow status.")


# --- Metrics --- (Using make_api_request)
with tabs[6]:
    st.header("📊 Metrics")
    if st.button("📈 Fetch Metrics", key="fetch_metrics_btn"): # Added a button to fetch
        metrics_data = make_api_request("GET", METRICS_URL, operation_name="Fetch Metrics")
        if metrics_data and not metrics_data.get("error"):
            st.json(metrics_data)
        else:
            st.error(f"Metrics fetch failed. Response: {metrics_data.get('details') if metrics_data else 'No response'}")
    else:
        st.info("Click the button to fetch metrics.")


# === Voice Agent Section (Bi-Directional Gemini Integration) ===
st.markdown("---")
st.markdown("## 🎙️ DebugIQ Voice Agent (with Gemini)")
st.caption("Speak your query, and DebugIQ's Gemini assistant will respond.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio_base64" in message and message["role"] == "assistant":
            try:
                audio_bytes = base64.b64decode(message["audio_base64"])
                st.audio(audio_bytes, format="audio/mp3") # Or "audio/wav" depending on your backend TTS
            except Exception as e:
                logger.error(f"Error playing audio for assistant message: {e}")
                st.caption("(Could not play audio for this message)")


try:
    ctx = webrtc_streamer(
        key=f"gemini_voice_agent_stream_{BACKEND_URL}", # Changed key slightly
        mode=WebRtcMode.SENDONLY
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        
        # desired_playing_state=True # Use this if you want it to start automatically in some cases
    
except Exception as e:
    st.error(f"Failed to initialize voice agent: {e}")
    logger.exception("Error initializing webrtc_streamer for Gemini Voice Agent")
    ctx = None

if ctx and ctx.audio_receiver:
    status_indicator = st.empty() # To show status like "Listening...", "Processing..."
    try:
        status_indicator.info("Listening...")
        audio_frames = ctx.audio_receiver.get_frames(timeout=0.2) # Increased timeout slightly

        if audio_frames:
            status_indicator.info("Processing audio...")
            # (Audio parameter inference logic - same as your provided code)
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
                status_indicator.info("🎙️ Transcribing and sending to Gemini...")
                temp_wav_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                        temp_wav_file_path = tmp_wav_file.name

                    with wave.open(temp_wav_file_path, 'wb') as wav_writer:
                        wav_writer.setnchannels(st.session_state.audio_num_channels)
                        wav_writer.setsampwidth(st.session_state.audio_sample_width)
                        wav_writer.setframerate(st.session_state.audio_sample_rate)
                        wav_writer.writeframes(st.session_state.audio_buffer)
                    
                    logger.info(f"Temporary WAV file for Gemini created at {temp_wav_file_path}")

                    with open(temp_wav_file_path, "rb") as f_audio:
                        files_payload = {"file": (f"audio_segment.wav", f_audio, "audio/wav")}
                        # Using make_api_request for transcription
                        transcribe_data = make_api_request(
                            "POST", 
                            VOICE_TRANSCRIBE_URL, 
                            files=files_payload, 
                            operation_name="Voice Transcription for Gemini"
                        )

                    transcript = transcribe_data.get("transcript") if transcribe_data and not transcribe_data.get("error") else None
                    
                    if transcript:
                        logger.info(f"Transcription successful: {transcript}")
                        st.session_state.chat_history.append({"role": "user", "content": transcript})
                        # Rerun to show user message immediately
                        st.rerun() 

                        # Now send to Gemini via your backend
                        # The backend should handle conversation history if needed
                        gemini_payload = {
                            "text_command": transcript,
                            "conversation_history": st.session_state.chat_history[:-1] # Send previous history for context
                        }
                        status_indicator.info(f"🗣️ You (Transcribed): \"{transcript}\" - Waiting for Gemini...")

                        gemini_response_data = make_api_request(
                            "POST",
                            GEMINI_CHAT_URL,
                            json_payload=gemini_payload,
                            operation_name="Gemini Chat"
                        )

                        if gemini_response_data and not gemini_response_data.get("error"):
                            assistant_text_response = gemini_response_data.get("text_response", "Sorry, I didn't get that.")
                            assistant_audio_base64 = gemini_response_data.get("audio_content_base64") # Expecting base64 audio from backend

                            assistant_message = {"role": "assistant", "content": assistant_text_response}
                            if assistant_audio_base64:
                                assistant_message["audio_base64"] = assistant_audio_base64
                            
                            st.session_state.chat_history.append(assistant_message)
                            status_indicator.empty() # Clear status
                            st.rerun() # Rerun to display Gemini's response and play audio
                        else:
                            error_detail = gemini_response_data.get('details', 'No specific error details.') if gemini_response_data else "No response from Gemini endpoint."
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error_detail}"})
                            status_indicator.error(f"Error from Gemini: {error_detail}")
                            # No st.rerun here, error shown, history updated if it happens next cycle.

                    else:
                        status_indicator.warning("Transcription returned empty or failed. Please try speaking again.")
                        if transcribe_data and transcribe_data.get("error"):
                             logger.error(f"Transcription error: {transcribe_data.get('details')}")
                        else:
                             logger.info("Transcription was empty.")

                except Exception as e: # Catch errors in the try-block for processing
                    status_indicator.error(f"An error occurred during voice processing: {e}")
                    logger.exception("Unexpected error in Gemini voice processing block")
                finally:
                    if temp_wav_file_path and os.path.exists(temp_wav_file_path):
                        try:
                            os.remove(temp_wav_file_path)
                        except OSError as e:
                            logger.error(f"Error removing temporary WAV file {temp_wav_file_path}: {e}")
                    st.session_state.audio_buffer = b""
                    st.session_state.audio_frame_count = 0
                    if not (ctx and ctx.audio_receiver and audio_frames): # If processing didn't complete due to no frames, clear indicator
                        status_indicator.empty()


        elif ctx and ctx.audio_receiver: # No new frames, but receiver is active
            status_indicator.empty() # Clear "Listening..." if no audio comes through for a bit
            pass


    except av.error.TimeoutError:
        status_indicator.empty() # Clear "Listening..." if timeout occurs
        pass # Expected if no frames are available within the timeout
    except Exception as e:
        if ctx and ctx.audio_receiver and not ctx.audio_receiver.is_closed:
            st.warning(f"An issue occurred with the audio stream: {e}. Try restarting the voice agent.")
            logger.error(f"Error processing audio frames for Gemini: {e}", exc_info=True)
        status_indicator.empty()

elif ctx and not ctx.audio_receiver : # Streamer active, but not receiving (mic stopped by user)
    if 'chat_history' in st.session_state and st.session_state.chat_history:
         if st.session_state.chat_history[-1]["role"] == "user" and "Processing..." in st.session_state.chat_history[-1]["content"]:
             # Clean up if user stops mic mid-processing indicator
             st.session_state.chat_history.pop()

    if st.session_state.audio_buffer:
        logger.info("Audio stream stopped by user with remaining buffer. Clearing buffer.")
        st.session_state.audio_buffer = b""
        st.session_state.audio_frame_count = 0
    # Optionally provide feedback that the agent is stopped
    # st.caption("Voice agent stopped. Click 'Start' to speak.")
