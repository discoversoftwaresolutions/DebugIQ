# File: frontend/debugiq_dashboard.py

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

import av
import numpy as np
import io
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import logging
import base64
import re
import threading
from urllib.parse import urljoin
import json

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app")

ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",
    "qa_validation": "/qa/run",
    "doc_generation": "/doc/generate",
    "issues_inbox": "/issues/attention-needed",
    "workflow_run": "/workflow/run_autonomous_workflow",
    "workflow_status": "/issues/{issue_id}/status",
    "system_metrics": "/metrics/status",
    "voice_transcribe": "/voice/transcribe",
    "gemini_chat": "/gemini/chat",
    "tts": "/voice/tts"
}

# === Helper Functions ===
def make_api_request(method, endpoint_key, payload=None, return_json=True):  # Takes endpoint_key, not full url
    """Makes an API request to the backend."""
    if endpoint_key not in ENDPOINTS:
        logger.error(f"Invalid endpoint key: {endpoint_key}")
        return {"error": f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'."}
    path_template = ENDPOINTS[endpoint_key]

    if endpoint_key == "workflow_status":
        issue_id = st.session_state.get("active_issue_id")
        if not issue_id:
            logger.error("Workflow status requested but no active_issue_id in session state.")
            st.session_state.workflow_completed = True
            return {"error": "No active issue ID to check workflow status."}
        try:
            path = path_template.format(issue_id=issue_id)
        except KeyError as e:
            logger.error(f"Failed to format workflow_status path: Missing key {e}")
            st.session_state.workflow_completed = True
            return {"error": "Internal error formatting workflow status URL: Missing issue ID key."}
    else:
        path = path_template

    url = urljoin(BACKEND_URL, path)
    try:
        logger.info(f"Making API request: {method} {url}")
        response = requests.request(method, url, json=payload, timeout=60)
        response.raise_for_status()
        logger.info(f"API request successful: {method} {url}")
        if return_json:
            return response.json()
        else:
            return response.content
    except requests.exceptions.Timeout:
        logger.error(f"API request timed out: {method} {url}")
        return {"error": "API request timed out. The backend might be slow or unresponsive."}
    except requests.exceptions.ConnectionError:
        logger.error(f"API connection error: {method} {url}")
        return {"error": "Could not connect to the backend API. Please check the backend URL and status."}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        detail = str(e)
        backend_detail = "N/A"
        if getattr(e, "response", None) is not None:
            detail = f"Status {e.response.status_code}"
            try:
                backend_json = e.response.json()
                backend_detail = backend_json.get('detail', backend_json)
                if isinstance(backend_detail, list) and all(isinstance(item, dict) for item in backend_detail):
                    backend_detail_str = json.dumps(backend_detail, indent=2)
                elif isinstance(backend_detail, dict):
                    backend_detail_str = json.dumps(backend_detail, indent=2)
                else:
                    backend_detail_str = str(backend_detail)
                detail = f"Status {e.response.status_code} - Backend Detail: {backend_detail_str}"
            except json.JSONDecodeError:
                backend_detail = e.response.text
                detail = f"Status {e.response.status_code} - Response Text: {backend_detail}"
            except Exception as json_e:
                logger.warning(f"Could not parse backend error response as JSON: {json_e}")
        return {"error": f"API request failed: {detail}", "backend_detail": backend_detail}

# === Audio Processing Helper ===
def frames_to_wav_bytes(frames):
    """Converts a list of audio frames (av.AudioFrame) to WAV formatted bytes."""
    if not frames:
        return None

    logger.info(f"Attempting to convert {len(frames)} audio frames to WAV.")

    try:
        frame_0 = frames[0]
        sample_rate = frame_0.sample_rate
        format_name = frame_0.format.name
        channels = frame_0.layout.channels
        sample_width_bytes = frame_0.format.bytes
        logger.info(f"Detected audio format: {format_name}, channels: {channels}, sample_rate: {sample_rate}, sample_width: {sample_width_bytes} bytes.")
    except Exception as e:
        logger.error(f"Error accessing frame properties: {e}")
        return None

    if 's16' in format_name and frame_0.layout.name in ['mono', 'stereo']:
        try:
            all_bytes = b"".join([frame.planes[0].buffer.tobytes() for frame in frames])
            logger.info(f"Concatenated raw bytes from frames, total size: {len(all_bytes)} bytes.")
            raw_data = all_bytes
        except Exception as e:
            logger.error(f"Error concatenating s16 audio frame bytes: {e}")
            return None
    elif 's32p' in format_name or 'f32p' in format_name:
        try:
            all_channels_data = [np.concatenate([frame.planes[i].to_ndarray() for frame in frames]) for i in range(channels)]
            interleaved_data = np.stack(all_channels_data, axis=-1)
            raw_data = interleaved_data.tobytes()
            logger.info(f"Interleaved planar data, resulting in {len(raw_data)} bytes.")
        except Exception as e:
            logger.error(f"Error processing planar audio frames: {e}")
            return None
    else:
        logger.error(f"Unsupported audio format or layout for WAV conversion: {format_name}, {frame_0.layout.name}. Support for s16, s32p, f32p (mono/stereo) implemented.")
        return None

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

def audio_frame_callback(frame: av.AudioFrame):
    """Callback function to receive and process audio frames from the browser."""
    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []
    if "audio_buffer_lock" not in st.session_state:
        st.session_state.audio_buffer_lock = threading.Lock()

    with st.session_state.audio_buffer_lock:
        if st.session_state.get('is_recording', False):
            st.session_state.audio_buffer.append(frame)
            if 'audio_format' not in st.session_state and st.session_state.audio_buffer:
                frame_0 = st.session_state.audio_buffer[0]
                st.session_state.audio_format = {
                    'sample_rate': frame_0.sample_rate,
                    'format_name': frame_0.format.name,
                    'channels': frame_0.layout.channels,
                    'sample_width_bytes': frame_0.format.bytes
                }

# === Main Application ===
st.set_page_config(page_title="DebugIQ Dashboard", layout="wide")
st.title("🧠 DebugIQ ")

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []
if 'audio_buffer_lock' not in st.session_state:
    st.session_state.audio_buffer_lock = threading.Lock()
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Idle"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Status", "📈 Metrics"])
tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics = tabs

# === Traceback + Patch Tab ===
with tab_trace:
    st.header("📄 Traceback & Patch Analysis")
    uploaded_file = st.file_uploader("Upload Traceback or Source Files", type=["txt", "py", "java", "js", "cpp", "c"], key="trace_file_uploader")

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.subheader("Original Code")
        original_language = uploaded_file.type.split('/')[-1] if uploaded_file.type in ["py", "java", "js", "cpp", "c"] else "plaintext"
        st.code(file_content, language=original_language, height=300)

        if st.button("🔬 Analyze & Suggest Patch", key="analyze_patch_btn"):
            with st.spinner("Analyzing and suggesting patch..."):
                payload = {
                    "code": file_content,
                    "language": original_language,
                }
                response = make_api_request("POST", "suggest_patch", payload)

            if "error" not in response:
                suggested_diff = response.get("diff", "No diff suggested.")
                explanation = response.get("explanation", "No explanation provided.")

                st.subheader("Suggested Patch")
                st.code(suggested_diff, language="diff", height=300)
                st.markdown(f"💡 **Explanation:** {explanation}")

                st.markdown("### ✍️ Edit Suggested Patch")
                edited_patch = st_ace(
                    value=suggested_diff,
                    language=original_language,
                    theme="monokai",
                    height=350,
                    key="ace_editor_patch"
                )

                st.markdown("### 🔍 Diff View (Original vs. Edited Patch)")
                if edited_patch is not None and file_content is not None:
                    diff_html = difflib.HtmlDiff(wrapcolumn=80).make_table(
                        fromlines=file_content.splitlines(),
                        tolines=edited_patch.splitlines(),
                        fromdesc="Original Code",
                        todesc="Edited Patch",
                        context=True
                    )
                    st.components.v1.html(diff_html, height=400, scrolling=True)
                else:
                    st.info("Upload original code and generate/edit patch to see diff.")

            else:
                st.error(response["error"])
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
                payload = {"patch_diff": patch_content}
                response = make_api_request("POST", "qa_validation", payload)

            if "error" not in response:
                st.subheader("Validation Results")
                st.json(response)
            else:
                st.error(response["error"])
                if "backend_detail" in response:
                    st.json({"Backend Detail": response["backend_detail"]})
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
                payload = {"code": code_content, "language": doc_language}
                response = make_api_request("POST", "doc_generation", payload)

            if "error" not in response:
                st.subheader("Generated Documentation")
                st.markdown(response.get("documentation", "No documentation generated."))
            else:
                st.error(response["error"])
                if "backend_detail" in response:
                    st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please upload a code file first.")

# === Issues Tab ===
with tab_issues:
    st.header("📣 Issues Inbox")
    st.write("This section lists issues needing attention from the autonomous workflow.")

    if st.button("🔄 Refresh Issues", key="issues_refresh_btn"):
        with st.spinner("Fetching issues..."):
            response = make_api_request("GET", "issues_inbox")

        if "error" not in response:
            if response.get("issues"):
                st.subheader("Issues Needing Attention")
                for issue in response.get("issues", []):
                    with st.expander(f"Issue ID: {issue.get('id', 'N/A')} - Status: {issue.get('status', 'Unknown Status')}"):
                        st.write(f"**Error Details:** {issue.get('error_message', 'No error details provided.')}")
            else:
                st.info("No issues needing attention found.")
        else:
            st.error(response["error"])
            if "backend_detail" in response:
                st.json({"Backend Detail": response["backend_detail"]})

# === Workflow Tab ===
with tab_workflow:
    st.header("🤖 Autonomous Workflow Trigger")
    st.write("Trigger an autonomous workflow run for a specific issue.")
    issue_id = st.text_input("Issue ID to Trigger Workflow", placeholder="e.g., BUG-123", key="workflow_trigger_issue_id")

    if st.button("▶️ Trigger Workflow", key="workflow_trigger_btn"):
        if issue_id:
            with st.spinner(f"Triggering workflow for issue {issue_id}..."):
                payload = {"issue_id": issue_id}
                response = make_api_request("POST", "workflow_run", payload)

            if "error" not in response:
                st.success(f"Workflow triggered successfully for Issue {issue_id}. Response: {response.get('message', 'No message.')}")
                st.session_state.active_issue_id = issue_id
                st.session_state.workflow_completed = False
            else:
                st.error(response["error"])
                if "backend_detail" in response:
                    st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please enter an Issue ID.")

# === Workflow Status Tab ===
with tab_status:
    st.header("🔍 Autonomous Workflow Status")
    issue_id_for_polling = st.text_input("Issue ID to check status (leave blank if triggered workflow above)", placeholder="e.g., BUG-123", key="workflow_status_issue_id")

    if issue_id_for_polling:
        st.session_state.active_issue_id = issue_id_for_polling
        st.session_state.workflow_completed = False

    st.write("Checking status for Issue ID:", st.session_state.get('active_issue_id') or "None (Trigger workflow or enter ID above)")
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed",
        "📦 PR Created"
    ]
    progress_map = {
        "Seeded": 0,
        "Fetching Details": 0,
        "Diagnosis in Progress": 1,
        "Diagnosis Complete": 1,
        "Patch Suggestion in Progress": 2,
        "Patch Suggestion Complete": 2,
        "Patch Validation in Progress": 3,
        "Patch Validated": 4,
        "PR Creation in Progress": 5,
        "PR Created - Awaiting Review/QA": 5
    }
    terminal_status = "PR Created - Awaiting Review/QA"
    failed_status = "Workflow Failed"

    def show_agent_progress(status):
        step = progress_map.get(status, 0)
        if status == failed_status:
            icon = "❌"
            st.markdown(f"{icon} Workflow Failed")
        else:
            st.progress((step + 1) / len(progress_labels))
            for i, label in enumerate(progress_labels):
                icon = "✅" if i <= step else ("🔄" if i == step + 1 else "⏳")
                st.markdown(f"{icon} {label}")

    # Polling logic (autorefresh only if an issue is active and workflow not complete)
    # Corrected the if condition in the previous turn
    if st.session_state.get('active_issue_id') and not st.session_state.workflow_completed:
        logger.info(f"Polling status for issue: {st.session_state.get('active_issue_id')}") # Ensure this also uses .get() if it didn't before
        # Corrected indentation for st_autorefresh
        st_autorefresh(interval=2000, key=f"workflow-refresh-{st.session_state.active_issue_id}") # <--- Corrected line 406 indentation

    # --- Fetch and Display Status ---
    # ... rest of the block starting with if st.session_state.active_issue_id ...
    if st.session_state.get('active_issue_id') and not st.session_state.workflow_completed:
       try:
            status_response = make_api_request("GET", "workflow_status")
            if "error" not in status_response:
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")
                st.session_state.last_status = current_status
                st.info(f"🔁 Live Status: **{current_status}**")
                if error_message:
                    st.error(f"Workflow Error: {error_message}")
                show_agent_progress(current_status)
                if current_status == terminal_status or current_status == failed_status:
                    st.session_state.workflow_completed = True
                    if current_status == terminal_status:
                        st.success("✅ DebugIQ agents completed full cycle.")
                    else:
                        st.error("❌ DebugIQ workflow failed.")
            else:
                st.error(status_response["error"])
                if "backend_detail" in status_response:
                    st.json({"Backend Detail": status_response["backend_detail"]})
                st.session_state.workflow_completed = True
        except Exception as e:
            st.error(f"Error polling workflow status: {e}")
            st.session_state.workflow_completed = True
    else:
        if getattr(st.session_state, "last_status", None):
            if st.session_state.last_status == terminal_status:
                st.success("✅ Workflow completed.")
            elif st.session_state.last_status == failed_status:
                st.error("❌ Workflow failed.")
                if "error_message" in st.session_state:
                    st.error(f"Last recorded error: {st.session_state.error_message}")
            else:
                st.info(f"Workflow finished with status: **{st.session_state.last_status}**")
            show_agent_progress(st.session_state.last_status)
        else:
            st.info("Enter an Issue ID or trigger a workflow to see status.")
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
            st.json(response)
        else:
            st.error(response["error"])
            if "backend_detail" in response:
                st.json({"Backend Detail": response["backend_detail"]})

# === DebugIQ Voice Agent Section (Dedicated Section at the bottom) ===
st.markdown("---")
st.markdown("---")
st.header("🎙️ DebugIQ Voice Agent")
st.write("Interact conversationally with DebugIQ using your voice or text. Ask questions or give commands related to debugging tasks.")
st.write("You can ask things like: 'Analyze the traceback', 'Generate documentation for this code', or ask general programming questions.")

chat_container = st.container(height=400)
with chat_container:
    for message in st.session_state.chat_history:
        role = "👤 User" if message["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {message['content']}")
        if message["role"] == "ai" and message.get("audio"):
            if isinstance(message['audio'], bytes):
                audio_hash = base64.b64encode(message['audio']).decode('utf-8')[:10]
            else:
                audio_hash = "error"
            try:
                st.audio(message["audio"], format='audio/wav', sample_rate=message.get("sample_rate", 44100), key=f"audio_player_{audio_hash}")
            except Exception as e:
                st.warning(f"Could not play audio: {e}")

status_placeholder = st.empty()
status_placeholder.info(f"Status: {st.session_state.recording_status}")

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("▶️ Start Recording", key="voice_start_btn", disabled=st.session_state.is_recording)
with col2:
    stop_button = st.button("⏹️ Stop Recording", key="voice_stop_btn", disabled=not st.session_state.is_recording)

try:
    ctx = webrtc_streamer(
        key="voice_agent_streamer_bottom",
        mode=WebRtcMode.SENDONLY,
        frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback,
    )
except Exception as e:
    st.error(f"Failed to initialize voice agent microphone: {e}")
    logger.exception("Error initializing webrtc_streamer for Voice Agent")
    ctx = None

if start_button:
    if ctx and ctx.state.playing:
        st.session_state.is_recording = True
        with st.session_state.audio_buffer_lock:
            st.session_state.audio_buffer = []
        st.session_state.pop('audio_format', None)
        st.session_state.recording_status = "Recording..."
        status_placeholder.info(f"Status: {st.session_state.recording_status}")
        st.rerun()
    else:
        st.warning("Microphone stream is not active. Please allow microphone access and ensure the WebRTC component is initialized.")
        st.session_state.recording_status = "Idle"
        status_placeholder.info(f"Status: {st.session_state.recording_status}")

if stop_button:
    st.session_state.is_recording = False
    st.session_state.recording_status = "Processing..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")

    audio_frames_to_process = []
    with st.session_state.audio_buffer_lock:
        audio_frames_to_process = st.session_state.audio_buffer[:]
        st.session_state.audio_buffer = []
    audio_format_info = st.session_state.pop('audio_format', {})

    if audio_frames_to_process:
        logger.info(f"Processing {len(audio_frames_to_process)} frames after stopping.")
        wav_data = frames_to_wav_bytes(audio_frames_to_process)
        if wav_data:
            audio_base64 = base64.b64encode(wav_data).decode('utf-8')
            logger.info(f"Encoded audio data to Base64, size: {len(audio_base64)} bytes.")
            st.session_state.recording_status = "Transcribing..."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")
            with st.spinner("Transcribing audio..."):
                transcription_payload = {"audio_base64": audio_base64}
                transcription_response = make_api_request("POST", "voice_transcribe", transcription_payload)

            user_text = "Could not transcribe audio."
            transcription_error = False
            if "error" not in transcription_response:
                user_text = transcription_response.get("transcription", user_text)
                st.session_state.recording_status = "Transcription Complete."
            else:
                user_text = f"Transcription Error: {transcription_response['error']}"
                transcription_error = True
                st.session_state.recording_status = "Transcription Error."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")

            if not transcription_error and user_text and user_text.strip() != "" and user_text != "Could not transcribe audio.":
                st.session_state.chat_history.append({"role": "user", "content": user_text})

            ai_response_text = ""
            ai_response_audio = None

            if not transcription_error and user_text and user_text.strip() != "" and user_text != "Could not transcribe audio.":
                st.session_state.recording_status = "Sending to Gemini..."
                status_placeholder.info(f"Status: {st.session_state.recording_status}")
                with st.spinner("Getting response from Gemini..."):
                    gemini_payload = {"text": user_text}
                    gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

                if "error" not in gemini_response:
                    ai_response_text = gemini_response.get("response", "No response from Gemini.")
                    st.session_state.recording_status = "Gemini Response Received."

                    if ai_response_text:
                        st.session_state.recording_status = "Generating Speech..."
                        status_placeholder.info(f"Status: {st.session_state.recording_status}")
                        with st.spinner("Generating AI speech..."):
                            tts_payload = {"text": ai_response_text}
                            tts_response_data = make_api_request("POST", "tts", tts_payload, return_json=False)

                        if not isinstance(tts_response_data, dict) or "error" not in tts_response_data:
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
            elif user_text.startswith("Transcription Error"):
                ai_response_text = f"Could not process audio: {user_text}"
                st.session_state.recording_status = "Processing failed."
            else:
                ai_response_text = "Please try speaking again."
                st.session_state.recording_status = "Processing failed."

            if ai_response_text or ai_response_audio is not None:
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

    with st.session_state.audio_buffer_lock:
        st.session_state.audio_buffer = []
    st.session_state.pop('audio_format', None)
    st.rerun()

st.markdown("---")
text_query = st.text_input("Type your query here:", key="text_chat_input")
send_text_button = st.button("Send Text Query", key="send_text_btn")

if send_text_button and text_query:
    st.session_state.recording_status = "Processing Text Query..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")

    user_text = text_query
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    ai_response_text = ""
    ai_response_audio = None

    st.session_state.recording_status = "Sending to Gemini..."
    status_placeholder.info(f"Status: {st.session_state.recording_status}")
    with st.spinner("Getting response from Gemini..."):
        gemini_payload = {"text": user_text}
        gemini_response = make_api_request("POST", "gemini_chat", gemini_payload)

    if "error" not in gemini_response:
        ai_response_text = gemini_response.get("response", "No response from Gemini.")
        st.session_state.recording_status = "Gemini Response Received."

        if ai_response_text:
            st.session_state.recording_status = "Generating Speech..."
            status_placeholder.info(f"Status: {st.session_state.recording_status}")
            with st.spinner("Generating AI speech..."):
                tts_payload = {"text": ai_response_text}
                tts_response_data = make_api_request("POST", "tts", tts_payload, return_json=False)

            if not isinstance(tts_response_data, dict) or "error" not in tts_response_data:
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

    if ai_response_text or ai_response_audio is not None:
        st.session_state.chat_history.append({"role": "ai", "content": ai_response_text, "audio": ai_response_audio})

    status_placeholder.info(f"Status: {st.session_state.recording_status}")
    st.session_state.text_chat_input = ""
    st.rerun()
