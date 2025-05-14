# File: frontend/debugiq_dashboard.py

# DebugIQ Dashboard with Code Editor, Diff View, Autonomous Features, and Dedicated Voice Agent

import streamlit as st
import requests
import os
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# Imports for Voice Agent section
import av  # Required for processing audio frames from streamlit-webrtc
import numpy as np  # Required for processing audio frames
import io  # Required for in-memory WAV file creation
import wave  # Required for WAV file creation
from streamlit_webrtc import webrtc_streamer, WebRtcMode  # Also needed if keeping voice
import logging  # Already imported by requests, but good to be explicit
import base64  # Needed for voice/image encoding
import re  # Needed for GitHub URL parsing
import threading  # Potentially needed for thread-safe buffer if issues arise
from urllib.parse import urljoin  # Needed for constructing URLs robustly
import json  # Needed for parsing JSON error details

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Backend Constants ===
# Use environment variable for the backend URL, with a fallback
# Set BACKEND_URL environment variable in Railway frontend settings
BACKEND_URL = os.getenv("BACKEND_URL", "https://debugiq-backend.railway.app")  # <-- Ensure this fallback is correct or use env var

# Define API endpoint paths relative to BACKEND_URL
ENDPOINTS = {
    "suggest_patch": "/debugiq/suggest_patch",  # Correct path for analyze.py endpoint
    "qa_validation": "/qa/run",  # Based on /qa prefix and @router.post("/run")
    "doc_generation": "/doc/generate",  # Based on /doc prefix and @router.post("/generate")
    "issues_inbox": "/issues/attention-needed",  # Based on no prefix and @router.get("/issues/attention-needed")
    "workflow_run": "/workflow/run_autonomous_workflow",  # Based on /workflow prefix and @router.post("/run_autonomous_workflow")
    "workflow_status": "/issues/{issue_id}/status",  # Based on no prefix and @router.get("/issues/{issue_id}/status")
    "system_metrics": "/metrics/status",  # Based on no prefix and @router.get("/metrics/status")
    "voice_transcribe": "/voice/transcribe",  # Example path - CHECK YOUR BACKEND
    "gemini_chat": "/gemini/chat",  # Example path - CHECK YOUR BACKEND
    "tts": "/voice/tts"  # Example path - CHECK YOUR BACKEND
}

# === Helper Functions ===
def make_api_request(method, endpoint_key, payload=None, return_json=True):
    """Makes an API request to the backend."""
    if endpoint_key not in ENDPOINTS:
        logger.error(f"Invalid endpoint key: {endpoint_key}")
        return {"error": f"Frontend configuration error: Invalid endpoint key '{endpoint_key}'."}

    path_template = ENDPOINTS[endpoint_key]

    # Handle endpoints that require formatting (like workflow_status)
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
        return response.json() if return_json else response.content
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
                detail = f"Status {e.response.status_code} - Backend Detail: {json.dumps(backend_detail, indent=2)}"
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

# === WebRTC Audio Frame Callback ===
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
