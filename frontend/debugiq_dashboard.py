
import streamlit as st
import requests
from streamlit_ace import st_ace
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from debugiq_gemini_voice import process_text_command, process_voice_file
import tempfile
import av
import wave
import numpy as np

# Config
BACKEND_URL = "https://debugiq-backend.railway.app"
st.set_page_config(page_title="DebugIQ", layout="wide")

# Sidebar
st.sidebar.title("DebugIQ")
st.sidebar.markdown("🚀 Powered by GPT-4o + Gemini")
st.sidebar.markdown("[Frontend Repo](https://github.com/discoversoftwaresolutions/DebugIQ-frontend)")
st.sidebar.markdown("[Backend Repo](https://github.com/discoversoftwaresolutions/DebugIQ-backend)")

# Tabs
tabs = st.tabs(["📄 Trace + Patch", "✅ QA", "📘 Docs", "📣 Notices", "🤖 Workflow", "🎙️ Voice"])

# Tab 1: Patch
with tabs[0]:
    st.header("📄 Upload Trace or Code")
    uploaded = st.file_uploader("Upload traceback or .py file", type=["py", "txt"])
    if uploaded:
        code = uploaded.read().decode("utf-8")
        st.code(code, language="python")
        if st.button("🛠 Suggest Patch"):
            res = requests.post(f"{BACKEND_URL}/suggest_patch", json={"code": code})
            if res.ok:
                patch = res.json()
                st.code(patch.get("diff", ""), language="diff")
                st.markdown(f"💡 {patch.get('explanation', '')}")
                st.session_state.edited_code = patch.get("patched_code", "")
            else:
                st.error("Patch suggestion failed.")

# Tab 2: QA
with tabs[1]:
    st.header("✅ Run QA")
    qa_input = st.text_area("Paste updated code", value=st.session_state.get("edited_code", ""))
    if st.button("Validate QA"):
        res = requests.post(f"{BACKEND_URL}/run_qa", json={"code": qa_input})
        if res.ok:
            st.json(res.json())
        else:
            st.error("QA validation failed.")

# Tab 3: Docs
with tabs[2]:
    st.header("📘 Auto-Documentation")
    doc_input = st.text_area("Paste code to document")
    if st.button("📄 Generate Docs"):
        res = requests.post(f"{BACKEND_URL}/generate_doc", json={"code": doc_input})
        if res.ok:
            st.markdown(res.json().get("doc", ""))
        else:
            st.error("Doc generation failed.")

# Tab 4: Notices
with tabs[3]:
    st.header("📣 Issue Inbox")
    if st.button("🔄 Refresh Issues"):
        res = requests.get(f"{BACKEND_URL}/issues/inbox")
        if res.ok:
            st.json(res.json())
        else:
            st.error("No issues found.")

# Tab 5: Autonomous Workflow
with tabs[4]:
    st.header("🤖 Autonomous Workflow")
    issue_id = st.text_input("Enter Issue ID (e.g. ISSUE-101)")
    if st.button("▶️ Run Agent Workflow"):
        res = requests.post(f"{BACKEND_URL}/run_autonomous_workflow", json={"issue_id": issue_id})
        if res.ok:
            st.success("Workflow complete")
            st.json(res.json())
        else:
            st.error("Workflow failed")

# Tab 6: Voice (Upload or Mic)
with tabs[5]:
    st.header("🎙️ Gemini Voice Agent")
    audio = st.file_uploader("Upload a .wav voice file", type=["wav"])
    if audio and st.button("🧠 Send to Gemini (Upload)"):
        result = process_voice_file(audio.read())
        st.markdown(result.get("response", "Error"))

    st.markdown("#### 🎤 Or Use Microphone")
    class AudioProcessor:
        def __init__(self): self.frames = []
        def recv(self, frame): self.frames.append(frame.to_ndarray()); return frame

    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        in_audio_enabled=True,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_processor_factory=AudioProcessor
    )

    if ctx and ctx.audio_receiver and st.button("🎧 Send Mic Audio to Gemini"):
        pcm = b''.join([f.tobytes() for f in ctx.audio_processor.frames])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm)
            with open(tmp.name, "rb") as f:
                result = process_voice_file(f.read())
                st.markdown(result.get("response", "Gemini error"))
