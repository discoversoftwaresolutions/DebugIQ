# debugiq_dashboard_v2.py

import streamlit as st
import requests
import difflib
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
PROJECT_NAME = "DebugIQ"
# CORRECT THIS URL to your Railway backend's public HTTPS URL
BACKEND_URL=https://debugiq-backend-production.up.railway.app

# ... rest of imports and sidebar ...

# --- UI Tabs ---
tabs = st.tabs([
    "📄 Trace + Patch", "✅ QA", "📘 Documentation",
    "📣 Issue Notices", "🤖 Autonomous Workflow", "🔍 Workflow Check"
])

# -------------------------------
# 📄 Tab 0: Trace + Patch (DebugIQanalyze)
# -------------------------------
with tabs[0]:
    st.header("📄 Traceback + Patch (DebugIQanalyze)")
    uploaded_file = st.file_uploader("Upload Python traceback or file", type=["py", "txt", "java", "js", "cpp", "c"]) # Added more types based on backend model
    if uploaded_file:
        code = uploaded_file.read().decode("utf-8")
        # Reduced height
        # Using st.code with disabled=True is fine if you want syntax highlighting
        st.code(code, language=uploaded_file.type.split('/')[-1] if uploaded_file.type in ["py", "java", "js", "cpp", "c"] else "plaintext", height=300) # Adjusted height, added language detection based on type

        if st.button("🧠 Suggest Patch"):
            with st.spinner("Analyzing and suggesting patch..."):
                # Corrected Payload to match backend AnalyzeRequest model
                # Need to provide language
                detected_language = uploaded_file.type.split('/')[-1] if uploaded_file.type in ["py", "java", "js", "cpp", "c"] else "plaintext" # Basic language guess

                payload = {
                    "code": code,
                    "language": detected_language, # <--- ADDED REQUIRED FIELD
                    # "context": {} # <--- Add optional context if needed
                }
                # URL is correct here
                res = requests.post(f"{BACKEND_URL}/debugiq/suggest_patch", json=payload)

                if res.status_code == 200:
                    patch_data = res.json() # Renamed from 'patch' to avoid conflict with later usage
                    # Assumes backend returns diff and explanation keys
                    suggested_diff = patch_data.get("diff", "")
                    explanation = patch_data.get("explanation", "No explanation provided.")

                    st.markdown("### Suggested Patch (Read-Only)")
                    st.code(suggested_diff, language="diff", height=300) # Use diff language for highlighting
                    st.markdown(f"💡 Explanation: {explanation}")

                    # Code Editor for Editing Patch (using suggest_diff now)
                    st.markdown("### ✍️ Edit Suggested Patch")
                    # Reduced height
                    edited_patch = st_ace(
                        value=suggested_diff, # Use the suggested diff here
                        language="python", # Or auto-detect language based on file type
                        theme="monokai",
                        height=350, # Adjusted height
                        key="ace_editor_patch"
                    )
                    # TODO: Add a button here to DO something with the edited_patch

                else:
                    st.error("Patch suggestion failed.")
                    # Display error details from backend if available
                    try:
                        st.json(res.json())
                    except:
                        st.write(f"Status Code: {res.status_code}")
                        st.write(f"Response Text: {res.text}")



# -------------------------------
# 🤖 Tab 5: Autonomous Workflow (Corrected Trigger URL)
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

    # Progress tracker logic (kept as is)
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
        "Patch Validated": 4, # Check backend status names match these
        "PR Created - Awaiting Review/QA": 5 # Check backend status names match these
    }
    terminal_status = "PR Created - Awaiting Review/QA" # Check backend status names match these
    failed_status = "Workflow Failed" # Add failed status


    def show_agent_progress(status):
        step = progress_map.get(status, 0)
        st.progress((step + 1) / len(progress_labels))
        for i, label in enumerate(progress_labels):
            icon = "✅" if i <= step else "🔄"
            st.markdown(f"{icon} {label}")

    # Polling logic (autorefresh only if workflow not complete)
    if st.session_state.active_issue_id and not st.session_state.workflow_completed:
        st_autorefresh(interval=2000, key="workflow-refresh") # Reduced interval for faster updates

        # Poll backend for live issue status
        try:
            # This URL is correct
            status_check = requests.get(
                f"{BACKEND_URL}/issues/{st.session_state.active_issue_id}/status"
            )
            if status_check.status_code == 200:
                status_data = status_check.json() # Renamed to avoid conflict
                current_status = status_data.get("status", "Unknown")
                error_message = status_data.get("error_message") # Get error message
                st.session_state.last_status = current_status
                st.info(f"🔁 Live Status: **{current_status}**")

                if error_message:
                    st.error(f"Workflow Error: {error_message}")

                show_agent_progress(current_status)

                if current_status == terminal_status or current_status == failed_status: # Check for failed status too
                    st.session_state.workflow_completed = True
                    if current_status == terminal_status:
                        st.success("✅ DebugIQ agents completed full cycle.")
                    else: # Workflow Failed
                        st.error("❌ DebugIQ workflow failed.")

            else:
                st.warning(f"Could not retrieve status from backend. Status code: {status_check.status_code}")
                try:
                    st.json(status_check.json())
                except:
                    st.write(f"Response text: {status_check.text}")

        except Exception as e:
            st.error(f"Status check failed: {e}")
            st.session_state.workflow_completed = True # Stop polling on status check error


    issue_id = st.text_input("Enter Issue ID to Run Workflow", placeholder="e.g. ISSUE-101")
    if st.button("▶️ Run DebugIQ Workflow"):
        if not issue_id:
            st.warning("Please provide a valid issue ID.")
        else:
            try:
                res = requests.post(
                    # CORRECT THIS URL
                    f"{BACKEND_URL}/workflow/run_autonomous_workflow", # <--- CORRECT URL
                    json={"issue_id": issue_id}
                )
                if res.status_code == 200:
                    result = res.json()
                    st.session_state.active_issue_id = issue_id
                    st.session_state.workflow_completed = False # Reset completed state to start polling
                    st.success(f"🚀 Workflow triggered for issue {issue_id}.")
                    # st.json(result) # Optional: show trigger response
                else:
                    st.error("Failed to trigger autonomous workflow.")
                    # Display error details from backend
                    try:
                        st.json(res.json())
                    except:
                        st.write(f"Status Code: {res.status_code}")
                        st.write(f"Response Text: {res.text}")
            except Exception as e:
                st.exception(e)


# ... rest of other tabs (QA, Doc, Notices, Workflow Check) - URLs look correct relative to BACKEND_URL

# ✅ QA
with tabs[1]:
    st.header("✅ QA Validator")
    qa_code = st.text_area("Paste your modified code for QA:")
    if st.button("Run QA"):
        res = requests.post(f"{BACKEND_URL}/qa/run", json={"code": qa_code}) # Correct URL
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error("QA validation failed.")
            try: st.json(res.json())
            except: st.write(f"Status: {res.status_code}, Response: {res.text}")


# 📘 Documentation
with tabs[2]:
    st.header("📘 Auto-Doc Generation")
    doc_code = st.text_area("Paste code to document:")
    if st.button("Generate Documentation"):
        res = requests.post(f"{BACKEND_URL}/doc/generate", json={"code": doc_code}) # Correct URL
        if res.status_code == 200:
            st.markdown(res.json().get("doc", "No documentation found."))
        else:
            st.error("Documentation generation failed.")
            try: st.json(res.json())
            except: st.write(f"Status: {res.status_code}, Response: {res.text}")


# 📣 Notices
with tabs[3]:
    st.header("📣 Issue Notices")
    # This endpoint is called on page load, might fail if backend is down
    try:
        res = requests.get(f"{BACKEND_URL}/issues/attention-needed") # Correct URL
        if res.status_code == 200:
            issues_attention = res.json().get("issues", []) # Assume backend returns {"issues": [...]}
            if issues_attention:
                st.subheader("Issues Needing Attention:")
                for issue in issues_attention:
                     st.write(f"- **{issue.get('id', 'N/A')}**: {issue.get('status', 'Unknown Status')} - {issue.get('error_message', 'No error details')}")
            else:
                 st.info("No issues needing attention.")
        elif res.status_code == 404: # Specific handling for 404 on this path if needed
             st.warning("Issue Attention endpoint not found (404). Check backend routing.")
        else:
            st.warning(f"Failed to fetch issues needing attention. Status code: {res.status_code}")
             try: st.json(res.json())
             except: st.write(f"Response text: {res.text}")
    except requests.exceptions.ConnectionError:
         st.error("Could not connect to the backend API. Ensure the backend is running.")
    except Exception as e:
         st.exception(e)


# 🔍 Tab 5: Workflow Check (Corrected URL)
with tabs[5]:
    st.header("🔍 DebugIQ Workflow Integrity Check")
    try:
        # This URL is correct
        res = requests.get(f"{BACKEND_URL}/workflow/check") # Correct URL
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error(f"Workflow check unavailable. Status code: {res.status_code}")
            try: st.json(res.json())
            except: st.write(f"Response text: {res.text}")
    except requests.exceptions.ConnectionError:
         st.error("Could not connect to the backend API. Ensure the backend is running.")
    except Exception as e:
        st.exception(e)
