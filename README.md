# 🧠 DebugIQ Front-End

The official Streamlit-based front-end for **DebugIQ**, an open-source autonomous debugging agent powered by GPT-4o and voice interaction. This UI provides trace analysis, code patch generation, QA validation, documentation generation, and an optional voice assistant interface.

---

## 🚀 Features

- 📄 Upload Python tracebacks and source files
- 🔍 GPT-4o-powered patch generation and explanations
- ✅ LLM + static QA validation
- 🧾 HTML diff viewer with unified and visual modes
- 📘 Auto-generated patch documentation
- 🎙️ Voice assistant (mic + file upload) powered by Gemini or Cora
- 🧠 Connected to the autonomous backend (FastAPI) at `https://autonomous-debug.onrender.com`

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [OpenAI GPT-4o](https://platform.openai.com/docs/)
- [FastAPI backend](https://github.com/discoversoftwaresolutions/DebugIQ-backend)
- `av`, `numpy`, `soundfile`, `requests`, `streamlit-ace`

---

## 📦 Installation (Local)

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit-dashboard.py
🌐 Deployment (Render Setup)
Setting	Value
Root Directory	frontend
Build Command	pip install -r requirements.txt
Start Command	streamlit run streamlit-dashboard.py --server.port=10000

📁 Project Structure
bash
Always show details

Copy
frontend/
├── streamlit-dashboard.py      # Full UI with patch/QA/doc/voice
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Optional: helps build av bindings
🤖 Voice Assistant (Optional)
The DebugIQ voice assistant is powered by Gemini or Cora (WIP) and supports:

Uploading .wav voice files

Real-time mic recording and TTS feedback

Command parsing, patch application, and QA validation

Cora is an upcoming voice-native AGI assistant under development by Discover Software Solutions.

📄 License
This project is open-source under the Apache 2.0 license (unless otherwise specified).

🤝 Contributing
Want to improve DebugIQ? Open an issue or pull request!

🌍 Maintained by
Discover Software Solutions
