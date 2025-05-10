# debugiq_dashboard_v2.py

import streamlit as st
import requests
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
PROJECT_NAME = "DebugIQ"
BACKEND_URL = "https://autonomous-debug.onrender.com"
REPO_LINKS = {
    "Frontend": "https://github.com/discoversoftwaresolutions/DebugIQ-frontend",
    "Backend": "https://github.com/discoversoftwaresolutions/DebugIQ-backend"
}

# --- Sidebar ---
st.sidebar.title(PROJECT_NAME)
st.sidebar.markdown("### 🔗 Repositories")
for name, link in REPO_LINKS.items():
    st.sidebar.markdown(f"[{name}]({link})")

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by DebugIQanalyze + DebugIQ Voice")
st.sidebar.markdown("Maintained by Discover Software Solutions")

# --- Main Dashboard UI ---
st.title("🧠 DebugIQ Autonomous Agent Dashboard")

# --- UI Tabs ---
tabs = st.tabs([
    "📄 Trace + Patch", "✅ QA", "📘 Documentation",
    "📣 Issue Notices", "🤖 Autonomous Workflow", "🔍 Workflow Check"
])

# -------------------------------
# 🤖 Tab 5: Autonomous Workflow
# -------------------------------
with tabs[4]:
    st.header("🤖 DebugIQ Autonomous Agent Workflow")

    # Session state for polling
    if "active_issue_id" not in st.session_state:
        st.session_state.active_issue_id = None
    if "last_status" not in st.session_state:
        st.session_state.last_status = None
    if "workflow_completed" not in st.session_state:
        st.session_state.workflow_completed = False

    # Progress tracker logic
    progress_labels = [
        "🧾 Fetching Details",
        "🕵️ Diagnosis",
        "🛠 Patch Suggestion",
        "🔬 Patch Validation",
        "✅ Patch Confirmed",
        "📦 PR Created"
    ]
    progress_map = {
        "Fetching Details": 0,
        "Diagnosis in Progress": 1,
        "Patch Suggestion in Progress": 2,
        "Patch Validation in Progress": 3,
        "Patch Validated": 4,
        "PR Created - Awaiting Review/QA": 5
    }
    terminal_status = "PR Created - Awaiting Review/QA"

    def show_agent_progress(status):
        step = progress_map.get(status, 0)
        st.progress((step + 1) / len(progress_labels))
        for i, label in enumerate(progress_labels):
            icon = "✅" if i <= step else "🔄"
            st.markdown(f"{icon} {label}")

    # Polling logic (autorefresh only if workflow not complete)
    if st.session_state.active_issue_id and not st.session_state.workflow_completed:
        st_autorefresh(interval=5000, key="workflow-refresh")

        # Poll backend for live issue status
        try:
            status_check = requests.get(
                f"{BACKEND_URL}/issues/{st.session_state.active_issue_id}/status"
            )
            if status_check.status_code == 200:
                current_status = status_check.json().get("status", "Unknown")
                st.session_state.last_status = current_status
                st.info(f"🔁 Live Status: **{current_status}**")
                show_agent_progress(current_status)

                if current_status == terminal_status:
                    st.session_state.workflow_completed = True
                    st.success("✅ DebugIQ agents completed full cycle.")
            else:
                st.warning("Could not retrieve status from backend.")
        except Exception as e:
            st.error(f"Status check failed: {e}")

    issue_id = st.text_input("Enter Issue ID to Run Workflow", placeholder="e.g. ISSUE-101")
    if st.button("▶️ Run DebugIQ Workflow"):
        if not issue_id:
            st.warning("Please provide a valid issue ID.")
        else:
            try:
                res = requests.post(
                    f"{BACKEND_URL}/run_autonomous_workflow",
                    json={"issue_id": issue_id}
                )
                if res.status_code == 200:
                    result = res.json()
                    st.session_state.active_issue_id = issue_id
                    st.session_state.workflow_completed = False
                    st.success("🚀 Workflow triggered.")
                    st.json(result)
                else:
                    st.error("Failed to trigger autonomous workflow.")
                    st.json(res.json())
            except Exception as e:
                st.exception(e)

# -------------------------------
# Other tabs (optional placeholders)
# -------------------------------

# 📄 Trace + Patch
with tabs[0]:
    st.header("📄 Traceback + Patch (DebugIQanalyze)")
    file = st.file_uploader("Upload Python traceback or file", type=["py", "txt"])
    if file:
        code = file.read().decode("utf-8")
        st.code(code, language="python")
        if st.button("🧠 Suggest Patch"):
            res = requests.post(f"{BACKEND_URL}/debugiq/suggest_patch", json={"code": code})
            if res.status_code == 200:
                patch = res.json()
                st.code(patch.get("diff", ""), language="diff")
                st.markdown(f"💡 Explanation: {patch.get('explanation', 'N/A')}")
            else:
                st.error("Patch generation failed.")

# ✅ QA
with tabs[1]:
    st.header("✅ QA Validator")
    qa_code = st.text_area("Paste your modified code for QA:")
    if st.button("Run QA"):
        res = requests.post(f"{BACKEND_URL}/qa/run", json={"code": qa_code})
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("QA validation failed.")

# 📘 Documentation
with tabs[2]:
    st.header("📘 Auto-Doc Generation")
    doc_code = st.text_area("Paste code to document:")
    if st.button("Generate Documentation"):
        res = requests.post(f"{BACKEND_URL}/doc/generate", json={"code": doc_code})
        if res.status_code == 200:
            st.markdown(res.json().get("doc", "No documentation found."))
        else:
            st.error("Documentation generation failed.")

# 📣 Notices
with tabs[3]:
    st.header("📣 Issue Notices")
    res = requests.get(f"{BACKEND_URL}/issues/attention-needed")
    if res.status_code == 200:
        st.json(res.json())
    else:
        st.warning("No issues needing attention found.")

# 🔍 Workflow Check
with tabs[5]:
    st.header("🔍 DebugIQ Workflow Integrity Check")
    try:
        res = requests.get(f"{BACKEND_URL}/workflow/check")
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("Workflow check unavailable.")
    except Exception as e:
        st.exception(e)
