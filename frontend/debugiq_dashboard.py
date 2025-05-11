# debugiq_dashboard.py
import streamlit as st
import requests
from streamlit_ace import st_ace
import difflib
from debugiq_gemini_voice import process_voice_file, process_text_command
import pandas as pd
import plotly.express as px# --- Branding + Config ---
from streamlit_autorefresh import st_autorefresh

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

tabs = st.tabs([
    "📄 Trace + Patch",
    "✅ QA",
    "📘 Docs",
    "📣 Issue Notices",
    "🤖 Autonomous Workflow",
    "🔍 Workflow Check",
    "📊 Metrics"
])
# --- Tabs ---
with tabs[0]:
    st.header("📄 Traceback + Patch (DebugIQanalyze)")
    uploaded_file = st.file_uploader("Upload traceback or .py file", type=["py", "txt"])

    if uploaded_file:
        original_code = uploaded_file.read().decode("utf-8")
        st.code(original_code, language="python")

        if st.button("🧠 Suggest Patch"):
            with st.spinner("Calling DebugIQanalyze..."):
                res = requests.post(f"{BACKEND_URL}/debugiq/suggest_patch", json={"code": original_code})
                if res.status_code == 200:
                    patch = res.json()
                    patched_code = patch.get("patched_code", "") or patch.get("suggested_patch", "")
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
                    
                    # Store edited code in session state for QA / Docs tabs
                    st.session_state.edited_patch = edited_code

                    st.success("✅ Patch displayed below. You can edit and pass it to QA.")
                else:
                    st.error("Failed to generate patch.")

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
    st.header("🤖 DebugIQ Autonomous Agent Workflow")
    st.markdown("""
    This agentic workflow runs the entire pipeline:
    - 🧾 Fetches issue details
    - 🕵️ Diagnoses root cause
    - 🛠 Suggests a patch
    - 🔬 Validates the patch
    - 📦 Creates a pull request
    - 🧠 Updates agent status live
    """)

    issue_id = st.text_input("Enter Issue ID", placeholder="e.g. ISSUE-101")

    # Visual progress UI map
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed",
        "📦 PR Created"
    ]
    progress_status_map = {
        "Fetching Details": 0,
        "Diagnosis in Progress": 1,
        "Patch Suggestion in Progress": 2,
        "Patch Validation in Progress": 3,
        "Patch Validated": 4,
        "PR Created - Awaiting Review/QA": 5
    }

    def show_agent_progress(status: str):
        step = progress_status_map.get(status, 0)
        st.progress((step + 1) / len(progress_labels))
        for i, label in enumerate(progress_labels):
            prefix = "✅" if i <= step else "🔄"
            st.markdown(f"{prefix} {label}")

    if st.button("▶️ Run DebugIQ Workflow"):
        if not issue_id:
            st.warning("Please enter a valid issue ID.")
        else:
            with st.spinner("Running DebugIQ agents..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/run_autonomous_workflow",
                        json={"issue_id": issue_id}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        status = result.get("status", "Diagnosis in Progress")
                        show_agent_progress(status)

                        if "pull_request" in result:
                            pr_url = result["pull_request"].get("url")
                            if pr_url:
                                st.success(f"✅ Pull Request Created: [View PR]({pr_url})")
                        st.subheader("🧾 Full Result")
                        st.json(result)

                    else:
                        st.error("❌ Workflow error from backend.")
                        st.json(response.json())

                except Exception as e:
                    st.exception(e)
# -------------------------------
# 🔍 Tab 6: Workflow Check
# -------------------------------
with tabs[5]:
    st.header("🔍 Workflow Integrity Check")
    check = requests.get(f"{BACKEND_URL}/workflow_check")
    if check.status_code == 200:
        st.json(check.json())
     
with tabs[6]:
    st_autorefresh(interval=30 * 1000, key="autorefresh_metrics")
    st.header("📊 Agent + Workflow Metrics")
    

    
   # (rest of your metrics logic indented here...)
    ...
    with st.spinner("Fetching metrics from backend..."):
        try:
            response = requests.get(METRICS_API_URL, timeout=5)
            data = response.json()

            # Agent Call Chart
            if "agent_calls" in data:
                agent_df = pd.DataFrame(
                    list(data["agent_calls"].items()), columns=["Agent Task", "Calls"]
                )
                st.subheader("Agent Call Volume")
                st.plotly_chart(px.bar(agent_df, x="Agent Task", y="Calls", color="Agent Task"))

            # Workflow Status Pie
            if "workflow_status" in data:
                status_df = pd.DataFrame(
                    list(data["workflow_status"].items()), columns=["Status", "Count"]
                )
                st.subheader("Workflow Status Distribution")
                st.plotly_chart(px.pie(status_df, names="Status", values="Count", hole=0.4))

            st.markdown(f"**Total Issues Tracked:** `{data.get('issue_count', 0)}`")
            st.markdown(f"**Last Updated:** `{data.get('last_updated')}`")

        except Exception as e:
            st.error(f"Failed to load metrics: {e}")

# 🎙️ DebugIQ Voice – Voice & Text Command Processing

st.markdown("---")
st.subheader("🎙️ DebugIQ Voice — Agentic Assistant")

voice_col, text_col = st.columns(2)

with voice_col:
    st.markdown("**🎧 Upload Command Audio (.wav)**")
    audio = st.file_uploader("Upload voice command", type=["wav"])
    if audio and st.button("Send Audio to DebugIQ Voice"):
        with st.spinner("Processing voice via Gemini..."):
            result = process_voice_file(audio.read())
            st.success("🔊 Voice processed")
            st.json(result)

with text_col:
    st.markdown("**🗣️ Text Command to DebugIQ Voice**")
    cmd = st.text_input("Enter your request...")
    fallback = st.checkbox("Fallback to GPT-4o", value=True)
    if st.button("Send Text to DebugIQ Voice"):
        with st.spinner("Processing text via Gemini..."):
            result = process_text_command(cmd, fallback_model="gpt-4o" if fallback else "gemini-pro")
            st.success("🧠 Command interpreted")
            st.json(result)
