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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def make_api_request(method, endpoint_key, payload=None, return_json=True):
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
            return {"error": f"Internal error formatting workflow status URL: Missing issue ID key."}
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
        if e.response is not None:
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

def frames_to_wav_bytes(frames):
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
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Status: Idle"
if 'last_status' not in st.session_state: # Initialize last_status
    st.session_state.last_status = None
if 'active_issue_id' not in st.session_state: # Initialize active_issue_id
    st.session_state.active_issue_id = None
if 'workflow_completed' not in st.session_state: # Initialize workflow_completed
    st.session_state.workflow_completed = True

st.sidebar.header("📦 GitHub Integration")
github_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo", key="sidebar_github_url")
if github_url:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", github_url)
    if match:
        owner, repo = match.groups()
        st.sidebar.success(f"**Repository:** {repo} (Owner: {owner})")
    else:
        st.sidebar.error("Invalid GitHub URL.")

tabs = st.tabs(["📄 Traceback + Patch", "✅ QA Validation", "📘 Documentation", "📣 Issues", "🤖 Workflow", "🔍 Workflow Status", "📈 Metrics", "🎙️ Voice Agent"])

tab_trace, tab_qa, tab_doc, tab_issues, tab_workflow, tab_status, tab_metrics, tab_voice = tabs

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
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please upload a patch file first.")

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
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please upload a code file first.")

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
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})

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
                if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})
        else:
            st.warning("Please enter an Issue ID.")

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

    if st.session_state.get('active_issue_id') and not st.session_state.workflow_completed:
        logger.info(f"Polling status for issue: {st.session_state.get('active_issue_id')}")
        st_autorefresh(interval=2000, key=f"workflow-refresh-{st.session_state.get('active_issue_id')}")

        try:
            status_response = make_api_request("GET", "workflow_status")

            if "error" not in status_response:
                current_status = status_response.get("status", "Unknown")
                error_message = status_response.get("error_message")

                st.session_state.last_status = current_status
                st.info(f"🔁 Live Status: **{current_status}**")

                if error_message:
                    st.error(f"Workflow Error: {error_message}")
                    st.session_state.error_message = error_message # Store error message

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
                if "error" in status_response:
                     st.session_state.error_message = status_response["error"] # Store API error

        except Exception as e:
            logger.error(f"API request failed during status check: {e}")
            st.error(f"Error fetching workflow status: {e}")
            st.session_state.workflow_completed = True
            st.session_state.error_message = str(e) # Store exception error

    else:
      if st.session_state.get('last_status'):
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
      # This line was causing the error and should be inside the condition or removed if logic is handled
      # st.session_state.workflow_completed = True # Moved this logic inside the polling block

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
            if "backend_detail" in response: st.json({"Backend Detail": response["backend_detail"]})


with tab_voice: # Added Voice Agent as a dedicated tab
    st.header("🎙️ DebugIQ Voice Agent")
    st.write("Interact conversationally with DebugIQ using your voice or text. Ask questions or give commands related to debugging tasks.")
    st.write("You can ask things like: 'Analyze the traceback', 'Generate documentation for this code', or ask general programming questions.")

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            role = "👤 User" if message["role"] == "user" else "🤖 AI"
            st.markdown(f"**{role}:** {message['content']}")
            if message["role"] == "ai" and message.get("audio") is not None:
                if isinstance(message['audio'], bytes):
                    audio_hash = base64.b64encode(message['audio']).decode('utf-8')[:10]
                    try:
                         st.audio(message['audio'], format='audio/wav', sample_rate=message.get('sample_rate', 44100), key=f"audio_{audio_hash}")
                    except Exception as e:
                         st.warning(f"Could not play audio: {e}")
                else:
                    st.warning("AI audio response is not in bytes format.")

    # Input area for voice and text
    col1, col2 = st.columns([1, 5])

    with col1:
        # WebRTC streamer for audio input
        webrtc_ctx = webrtc_streamer(
            key="voice_input",
            mode=WebRtcMode.SENDONLY,
            audio_html_attrs={"auto": True},
            audio_frame_callback=audio_frame_callback,
        )

        if webrtc_ctx.state.playing:
            st.session_state.recording_status = "Recording..."
            st.session_state.is_recording = True
        else:
            # Only process audio buffer if recording just stopped
            if st.session_state.is_recording and st.session_state.audio_buffer:
                st.session_state.recording_status = "Processing..."
                st.session_state.is_recording = False # Ensure it's set to False immediately

                with st.spinner("Processing audio..."):
                    audio_bytes = None
                    with st.session_state.audio_buffer_lock:
                        # Use stored format info if available, otherwise fallback
                        audio_format = st.session_state.get('audio_format', {'sample_rate': 44100, 'format_name': 's16', 'channels': 1, 'sample_width_bytes': 2})
                        # Reconstruct dummy frames with known format if processing requires format info
                        # A better approach might pass format directly to frames_to_wav_bytes if needed
                        # For now, assuming frames_to_wav_bytes can work with the frame objects directly
                        audio_bytes = frames_to_wav_bytes(st.session_state.audio_buffer)

                    if audio_bytes:
                        st.session_state.audio_buffer = [] # Clear the buffer after processing
                        st.session_state.recording_status = "Transcribing..."

                        # Send audio to backend for transcription
                        transcribe_response = make_api_request("POST", "voice_transcribe", payload={"audio_base64": base64.b64encode(audio_bytes).decode('utf-8')})

                        if "error" not in transcribe_response:
                            transcribed_text = transcribe_response.get("text", "")
                            if transcribed_text:
                                st.session_state.chat_history.append({"role": "user", "content": transcribed_text, "audio": audio_bytes})
                                st.session_state.recording_status = "Thinking..."
                                # Send transcribed text to Gemini chat
                                chat_response = make_api_request("POST", "gemini_chat", payload={"query": transcribed_text, "history": st.session_state.chat_history})

                                if "error" not in chat_response:
                                    ai_response_text = chat_response.get("response", "Sorry, I couldn't generate a response.")
                                    st.session_state.recording_status = "Synthesizing speech..."

                                    # Get TTS audio for the AI response
                                    tts_response = make_api_request("POST", "tts", payload={"text": ai_response_text}, return_json=False)

                                    if "error" not in tts_response:
                                         # Store text and audio for AI response
                                         st.session_state.chat_history.append({
                                             "role": "ai",
                                             "content": ai_response_text,
                                             "audio": tts_response, # This should be the audio bytes
                                             "sample_rate": 44100 # Assuming a standard sample rate for TTS
                                             })
                                    else:
                                        st.session_state.chat_history.append({"role": "ai", "content": f"{ai_response_text} (Audio error: {tts_response['error']})", "audio": None})
                                        st.error(f"TTS Error: {tts_response['error']}")
                                else:
                                    st.session_state.chat_history.append({"role": "ai", "content": f"Sorry, I couldn't process that. (Chat error: {chat_response['error']})", "audio": None})
                                    st.error(f"Chat Error: {chat_response['error']}")
                            else:
                                st.warning("No speech detected.")
                                st.session_state.chat_history.append({"role": "user", "content": "(No speech detected)"})

                        else:
                            st.error(f"Transcription Error: {transcribe_response['error']}")
                            st.session_state.chat_history.append({"role": "user", "content": "(Transcription failed)"})
                    else:
                        st.error("Failed to convert audio frames to WAV.")
                        st.session_state.chat_history.append({"role": "user", "content": "(Audio processing failed)"})


            st.session_state.recording_status = "Idle"

    with col2:
        user_input = st.text_input("Type your message here or use the microphone:", key="voice_text_input")
        if st.button("Send", key="voice_send_button"):
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input, "audio": None})
                st.session_state.recording_status = "Thinking..."

                chat_response = make_api_request("POST", "gemini_chat", payload={"query": user_input, "history": st.session_state.chat_history})

                if "error" not in chat_response:
                    ai_response_text = chat_response.get("response", "Sorry, I couldn't generate a response.")
                    st.session_state.recording_status = "Synthesizing speech..."

                    tts_response = make_api_request("POST", "tts", payload={"text": ai_response_text}, return_json=False)

                    if "error" not in tts_response:
                         st.session_state.chat_history.append({
                             "role": "ai",
                             "content": ai_response_text,
                             "audio": tts_response,
                             "sample_rate": 44100
                             })
                    else:
                        st.session_state.chat_history.append({"role": "ai", "content": f"{ai_response_text} (Audio error: {tts_response['error']})", "audio": None})
                        st.error(f"TTS Error: {tts_response['error']}")
                else:
                    st.session_state.chat_history.append({"role": "ai", "content": f"Sorry, I couldn't process that. (Chat error: {chat_response['error']})", "audio": None})
                    st.error(f"Chat Error: {chat_response['error']}")

                st.session_state.recording_status = "Idle"
                st.rerun()

    st.markdown(f"**Status:** {st.session_state.recording_status}")
