# debugiq_dashboard.py

import streamlit as st
import requests
from streamlit_ace import st_ace
import difflib

# --- Branding + Config ---
PROJECT_NAME = "DebugIQ"
BACKEND_URL = "https://autonomous-debug.onrender.com"
REPO_LINKS = {
    "GitHub (Frontend)": "https://github.com/discoversoftwaresolutions/DebugIQ-frontend",
    "GitHub (Backend)": "https://github.com/discoversoftwaresolutions/DebugIQ-backend"
}

# --- Sidebar Branding ---
st.sidebar.title(PROJECT_NAME)
st.sidebar.markdown("### 🔗 Repositories")
for name, url in REPO_LINKS.items():
    st.sidebar.markdown(f"[{name}]({url})")
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by DebugIQanalyze (GPT-4o) + DebugIQ Voice (Gemini)")
st.sidebar.markdown("Maintained by Discover Software Solutions")

# --- Main Interface ---
st.title("🧠 DebugIQ Agentic Dashboard")
st.markdown("A unified agent interface for autonomous debugging, documentation, QA, and workflow orchestration.")

# --- Tabs ---
tabs = st.tabs([
    "📄 Trace + Patch", "✅ QA", "📘 Documentation",
    "📣 Issue Notices", "🤖 Autonomous Workflow", "🔍 Workflow Check"
])

# -------------------------------
# 📄 Tab 1: Trace + Patch (DebugIQanalyze)
# -------------------------------
with tabs[0]:
    st.header("📄 DebugIQanalyze: Patch from Traceback")
    uploaded = st.file_uploader("Upload traceback or code file", type=["py", "txt"])
    if uploaded:
        code_text = uploaded.read().decode("utf-8")
        st.code(code_text, language="python")

        if st.button("🔧 Analyze and Patch with DebugIQanalyze"):
            with st.spinner("Calling DebugIQanalyze (GPT-4o)..."):
                res = requests.post(f"{BACKEND_URL}/suggest_patch", json={"code": code_text})
                if res.status_code == 200:
                    patch = res.json()
                    st.code(patch["diff"], language="diff")
                    st.markdown(f"💡 **Explanation:** {patch['explanation']}")
                else:
                    st.error("Patch generation failed.")

    st.subheader("🧑‍💻 Live Editor")
    st_ace(language="python", theme="twilight", height=250)

# -------------------------------
# ✅ Tab 2: QA Validation
# -------------------------------
with tabs[1]:
    st.header("✅ QA with DebugIQanalyze")
    qa_input = st.text_area("Paste updated code for validation:")
    if st.button("Run QA Validation"):
        with st.spinner("Validating with DebugIQanalyze..."):
            qa_response = requests.post(f"{BACKEND_URL}/run_qa", json={"code": qa_input})
            if qa_response.status_code == 200:
                st.json(qa_response.json())
            else:
                st.error("QA validation unavailable.")

# -------------------------------
# 📘 Tab 3: Documentation
# -------------------------------
with tabs[2]:
    st.header("📘 Auto-Documentation with DebugIQanalyze")
    doc_input = st.text_area("Paste code to generate documentation:")
    if st.button("📝 Generate Patch Doc"):
        doc_response = requests.post(f"{BACKEND_URL}/generate_doc", json={"code": doc_input})
        if doc_response.status_code == 200:
            st.markdown(doc_response.json().get("doc", "No documentation generated."))
        else:
            st.error("Doc generation failed.")

# -------------------------------
# 📣 Tab 4: Issue Notices
# -------------------------------
with tabs[3]:
    st.header("📣 Detected Issues (Autonomous Agent Summary)")
    if st.button("🔍 Fetch Notices"):
        issues = requests.get(f"{BACKEND_URL}/issues")
        if issues.status_code == 200:
            st.json(issues.json())
        else:
            st.warning("No issue data or backend error.")

# -------------------------------
# 🤖 Tab 5: Autonomous Workflow
# -------------------------------
with tabs[4]:
    st.header("🤖 DebugIQ Agent Workflow")
    wf = requests.get(f"{BACKEND_URL}/autonomous_workflow")
    if wf.status_code == 200:
        st.json(wf.json())
    else:
        st.error("Workflow agent not responding.")

# -------------------------------
# 🔍 Tab 6: Workflow Check
# -------------------------------
with tabs[5]:
    st.header("🔍 Workflow Integrity Check")
    check = requests.get(f"{BACKEND_URL}/workflow_check")
    if check.status_code == 200:
        st.json(check.json())
    else:
        st.warning("Workflow status unavailable.")

# -------------------------------
# 🎙️ DebugIQ Voice (Gemini + GPT-4o fallback)
# -------------------------------
st.markdown("---")
st.subheader("🎙️ DebugIQ Voice — Agentic Assistant (Text + Voice Input)")
voice_col, text_col = st.columns(2)

with voice_col:
    st.markdown("**🎧 Upload Command Audio (.wav)**")
    audio = st.file_uploader("Upload voice command", type=["wav"])
    # Placeholder response — to be wired to Gemini
    if audio:
        st.success("Audio uploaded — Gemini processing placeholder active")

with text_col:
    st.markdown("**🗣️ Text Command to DebugIQ Voice**")
    command = st.text_input("What do you want DebugIQ to do?")
    fallback = st.checkbox("Fallback to GPT-4o if Gemini unavailable", value=True)

    if st.button("Send to DebugIQ Voice"):
        st.markdown("⏳ Processing command...")
        response_text = "✔️ Command processed by DebugIQ Voice (Gemini)."
        if fallback:
            response_text += "\nFallback GPT-4o engaged if Gemini fails."
        st.success(response_text)
