
import streamlit as st
import requests
import os
import difflib
import tempfile
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import av
from difflib import HtmlDiff
import streamlit.components.v1 as components
import wave
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_BACKEND_URL = "https://debugiq-backend.railway.app"
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)

# --- Streamlit Config ---
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ Autonomous Debugging Dashboard")

# --- Tabs Setup ---
tabs = st.tabs([
    "📄 Traceback + Patch",
    "✅ QA Validation",
    "📘 Documentation",
    "📣 Issue Notices",
    "🤖 Autonomous Workflow",
    "🔍 Workflow Check",
    "📊 Metrics"
])

# --- API Endpoints ---
ANALYZE_URL = f"{BACKEND_URL}/debugiq/analyze"
QA_URL = f"{BACKEND_URL}/qa/"
DOC_URL = f"{BACKEND_URL}/doc/"
VOICE_TRANSCRIBE_URL = f"{BACKEND_URL}/voice/transcribe"
VOICE_COMMAND_URL = f"{BACKEND_URL}/voice/command"
ISSUES_INBOX_URL = f"{BACKEND_URL}/issues/inbox"
WORKFLOW_RUN_URL = f"{BACKEND_URL}/workflow/run"
WORKFLOW_CHECK_URL = f"{BACKEND_URL}/workflow/status"
METRICS_URL = f"{BACKEND_URL}/metrics/summary"

# --- Helper function ---
def post_json(url, payload):
    try:
        response = requests.post(url, json=payload, timeout=15)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- Traceback + Patch ---
with tabs[0]:
    st.header("📄 Traceback + Patch")
    uploaded_file = st.file_uploader("Upload traceback or .py file", type=["py", "txt"])
    if uploaded_file:
        original_code = uploaded_file.read().decode("utf-8")
        st.code(original_code, language="python")

        if st.button("🧠 Suggest Patch"):
            res = requests.post(f"{BACKEND_URL}/debugiq/suggest_patch", json={"code": original_code})
            if res.status_code == 200:
                patch = res.json()
                patched_code = patch.get("patched_code", "")
                patch_diff = patch.get("diff", "")

                st.markdown("### 🔍 Diff View")
                html_diff = difflib.HtmlDiff().make_file(
                    original_code.splitlines(),
                    patched_code.splitlines(),
                    fromdesc="Original",
                    todesc="Patched"
                )
                st.components.v1.html(html_diff, height=450, scrolling=True)

                st.markdown("### ✍️ Edit Patch (Live)")
                edited_code = st_ace(value=patched_code, language="python", theme="monokai", height=300)
                st.session_state.edited_patch = edited_code
            else:
                st.error("Patch generation failed.")

# --- QA Validation ---
with tabs[1]:
    st.header("✅ QA Validation")
    qa_input = st.text_area("Paste updated code for validation:")
    if st.button("Run QA Validation"):
        res = requests.post(QA_URL, json={"code": qa_input})
        st.json(res.json() if res.status_code == 200 else {"error": "Validation failed"})

# --- Documentation ---
with tabs[2]:
    st.header("📘 Documentation")
    doc_input = st.text_area("Paste code to generate documentation:")
    if st.button("📝 Generate Patch Doc"):
        doc_res = requests.post(DOC_URL, json={"code": doc_input})
        if doc_res.status_code == 200:
            st.markdown(doc_res.json().get("doc", "No documentation generated."))
        else:
            st.error("Doc generation failed.")

# --- Issue Notices ---
with tabs[3]:
    st.header("📣 Detected Issues (Autonomous Agent Summary)")
    if st.button("🔍 Fetch Notices"):
        issues = requests.get(ISSUES_INBOX_URL)
        st.json(issues.json() if issues.status_code == 200 else {"error": "Issue data fetch failed"})

# --- Autonomous Workflow ---
with tabs[4]:
    st.header("🤖 Run DebugIQ Autonomous Workflow")
    issue_id = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101")
    if st.button("▶️ Run Workflow"):
        res = requests.post(WORKFLOW_RUN_URL, json={"issue_id": issue_id})
        if res.status_code == 200:
            st.success("Workflow triggered.")
            st.json(res.json())
        else:
            st.error("Workflow execution failed.")

# --- Workflow Check ---
with tabs[5]:
    st.header("🔍 Workflow Integrity Check")
    res = requests.get(WORKFLOW_CHECK_URL)
    st.json(res.json() if res.status_code == 200 else {"error": "Workflow check failed"})

# --- Metrics ---
with tabs[6]:
    st.header("📊 Metrics")
    res = requests.get(METRICS_URL)
    st.json(res.json() if res.status_code == 200 else {"error": "Metrics fetch failed"})
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
