# 🧠 IBM Mental Health Assistant

> **Powered by IBM watsonx.ai · IBM Granite Models · Agentic AI Architecture**

[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0f62fe?style=for-the-badge&logo=ibm)](https://www.ibm.com/watsonx)
[![IBM Granite](https://img.shields.io/badge/IBM-Granite%2013B-8a3ffc?style=for-the-badge&logo=ibm)](https://www.ibm.com/granite)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)

---

## 🎯 Project Overview

An intelligent **Mental Health Assistant** built as a single-file Python Flask application demonstrating:

- 🤖 **IBM watsonx.ai Studio**
- 🧬 **IBM Granite Models** (granite-13b-instruct-v2)
- ⚙️ **Agentic AI Architecture**
- 🎯 **Agent Orchestration**
- 💙 **Mental Health Awareness & Suicide Prevention Support**

> Suitable for **IBM SkillsBuild**, Hackathons, Academic Projects, and AI Showcases.

---

## ⚙️ Five Specialized AI Agents

| # | Agent | Purpose |
|---|-------|---------|
| 🧠 1 | **Mental Health Awareness Agent** | Educates on anxiety, depression, stress, burnout, mindfulness & self-care |
| 💙 2 | **Emotional Support Agent** | Empathetic active listening, validation & compassionate companionship |
| 🔍 3 | **Distress Detection Agent** | Real-time 3-tier crisis signal recognition & emergency routing |
| 🌱 4 | **Coping Strategy Agent** | Personalised evidence-based coping plans & actionable techniques |
| 🏥 5 | **Risk Prediction & Resources Agent** | Mental health risk scoring (0–100) + professional support connections |

---

## 🏗️ Architecture

```
Browser (SPA)  →  Flask Routes  →  Orchestrator  →  5 AI Agents  →  IBM Granite 13B
                                        ↓
                              Intent Classification
                              (keyword routing)
                                        ↓
                    ┌──────────────────────────────────────┐
                    │  Always: Distress Detection Agent    │
                    │  Conditional: Awareness / Support /  │
                    │              Coping / Risk           │
                    └──────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
copy .env.example .env
# Edit .env with your IBM watsonx.ai credentials
```

### 3. Run
```bash
python app.py
```

### 4. Open browser
```
http://localhost:5000
```

> **Demo Mode:** If no credentials are set, the app runs with rich pre-written responses so the full UI can be showcased immediately.

---

## 🔧 Configuration

Edit `.env` file:

```env
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT=your-watsonx-project-id
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
```

---

## 🆘 Crisis Resources

| Resource | Contact |
|----------|---------|
| 988 Suicide & Crisis Lifeline | Call or text **988** (US) |
| Crisis Text Line | Text **HOME** to **741741** |
| NAMI Helpline | **1-800-950-6264** |
| International | [iasp.info](https://www.iasp.info/resources/Crisis_Centres/) |

---

## 📁 Project Structure

```
mental/
├── app.py            ← Complete application (single file)
├── requirements.txt  ← Python dependencies
├── .env.example      ← Credential template
└── README.md         ← This file
```

---

## ⚠️ Disclaimer

This AI assistant provides **educational information only** and is **not a substitute** for professional mental health care. Always consult a licensed mental health professional.

---

*Built with ❤️ for IBM SkillsBuild · Powered by IBM watsonx.ai & IBM Granite Models*
