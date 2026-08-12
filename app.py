"""
IBM watsonx.ai · Mental Health Assistant
Powered by IBM Granite Models · Agentic AI Architecture
IBM SkillsBuild · Hackathon · Academic · AI Showcase
"""

import os, json
from flask import Flask, request, jsonify, render_template_string

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

WATSONX_API_KEY  = os.getenv("WATSONX_API_KEY",  "your-ibm-watsonx-api-key")
WATSONX_URL      = os.getenv("WATSONX_URL",      "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT  = os.getenv("WATSONX_PROJECT",  "your-project-id")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")

_DEMO_MODE = (
    WATSONX_API_KEY in ("", "your-ibm-watsonx-api-key") or
    WATSONX_PROJECT in ("", "your-project-id")
)

DEMO_RESPONSES = {
    "awareness": (
        "Anxiety is a natural emotional response to stress or perceived threats — but it becomes a concern when it is persistent, excessive, or disrupts daily life.\n\n"
        "Common symptoms include:\n"
        "• Excessive worry or racing thoughts\n"
        "• Restlessness and irritability\n"
        "• Difficulty concentrating or sleeping\n"
        "• Physical symptoms: rapid heartbeat, sweating, trembling\n"
        "• Avoidance of anxiety-triggering situations\n\n"
        "Types include: Generalized Anxiety Disorder (GAD), Social Anxiety, Panic Disorder, and Phobias.\n\n"
        "Mindfulness, CBT therapy, exercise, and breathing techniques are proven effective. "
        "You are not alone — anxiety affects over 284 million people worldwide. Always consult a mental health professional for a proper assessment. 💙"
    ),
    "support": (
        "I hear you, and I want you to know your feelings are completely valid. "
        "It takes real courage to share what you're going through, and I'm truly glad you reached out.\n\n"
        "What you're experiencing sounds genuinely hard, and it's okay to not be okay sometimes. "
        "You don't have to carry this weight alone.\n\n"
        "I'm here with you right now. Would you like to tell me a little more about what's been "
        "happening? Sometimes just putting words to our feelings can bring a small sense of relief. 💙"
    ),
    "distress": (
        "Thank you for trusting me with something so personal. I want you to know that your pain is real and you deserve support.\n\n"
        "Please know that you are not alone in this moment. Reaching out is a meaningful and brave first step.\n\n"
        "If you are in crisis right now, please contact:\n"
        "• 988 Suicide & Crisis Lifeline — Call or text 988 (US, available 24/7)\n"
        "• Crisis Text Line — Text HOME to 741741\n"
        "• Emergency services — Call 911 (US) or 999 (UK)\n\n"
        "You matter. Your life has value. Please reach out to a professional who can truly help. 💙"
    ),
    "coping": (
        "Here are 5 evidence-based coping strategies you can try right now:\n\n"
        "1. 5-4-3-2-1 Grounding — Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Anchors you to the present.\n\n"
        "2. Box Breathing (4-4-4-4) — Inhale 4 counts, hold 4, exhale 4, hold 4. Repeat 4 times. Activates your parasympathetic nervous system.\n\n"
        "3. Journaling — Write freely for 10 minutes without judgment. Externalising thoughts reduces their power.\n\n"
        "4. Progressive Muscle Relaxation — Tense each muscle group for 5 seconds, then release. Work from feet to forehead.\n\n"
        "5. Reach Out — Call or text someone you trust. Social connection is one of the most powerful stress buffers we have. 🌱"
    ),
    "risk": (
        "I want to gently acknowledge how much you've shared today — that takes real strength.\n\n"
        "Based on our conversation, it may be helpful to speak with a licensed mental health professional. "
        "This isn't a sign of weakness — it's one of the most powerful acts of self-care you can take.\n\n"
        "Steps you can take today:\n"
        "• Contact your primary care doctor for a referral\n"
        "• Search for a therapist at psychologytoday.com\n"
        "• Try BetterHelp or 7 Cups for immediate online support\n"
        "• Call NAMI Helpline: 1-800-950-6264\n\n"
        "You deserve consistent, professional care. Taking this step is a sign of incredible self-awareness. 🏥"
    ),
    "general": (
        "Thank you for sharing that with me. Mental health is something that touches all of us, and conversations like this one matter deeply.\n\n"
        "Whether you're looking for information, emotional support, coping strategies, or professional resources — I'm here to help guide you.\n\n"
        "What would be most helpful for you right now? 💙"
    ),
}

def demo_granite_generate(prompt: str) -> str:
    pl = prompt.lower()
    if any(k in pl for k in ["suicide", "kill", "die", "hurt myself", "hopeless", "worthless", "can't go on"]):
        return DEMO_RESPONSES["distress"]
    if any(k in pl for k in ["cope", "coping", "strategy", "technique", "manage", "how can i"]):
        return DEMO_RESPONSES["coping"]
    if any(k in pl for k in ["feel", "feeling", "lonely", "sad", "upset", "overwhelm", "support", "listen"]):
        return DEMO_RESPONSES["support"]
    if any(k in pl for k in ["what is", "explain", "define", "tell me", "anxiety", "depression",
                              "burnout", "mindfulness", "stress", "mental health", "symptoms"]):
        return DEMO_RESPONSES["awareness"]
    if any(k in pl for k in ["risk", "resource", "professional", "help", "therapist"]):
        return DEMO_RESPONSES["risk"]
    return DEMO_RESPONSES["general"]


app = Flask(__name__)

def get_granite_client():
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=GRANITE_MODEL_ID, credentials=creds, project_id=WATSONX_PROJECT,
        params={Params.MAX_NEW_TOKENS:600, Params.MIN_NEW_TOKENS:30,
                Params.TEMPERATURE:0.7, Params.TOP_P:0.9, Params.REPETITION_PENALTY:1.1}
    )

def granite_generate(prompt: str) -> str:
    if _DEMO_MODE:
        return demo_granite_generate(prompt)
    try:
        client = get_granite_client()
        result = client.generate_text(prompt=prompt)
        return result.strip() if result else "I'm here to help. Could you share a bit more?"
    except:
        return demo_granite_generate(prompt)


# ── AGENTS ──────────────────────────────────────────────────────────────────

def agent_awareness(user_message):
    topics = ["anxiety","depression","stress","burnout","mindfulness","emotional wellness",
              "self-care","mental health","panic","trauma","ptsd","ocd","bipolar","phobia"]
    prompt = f"""You are the Mental Health Awareness Agent powered by IBM Granite.
Educate the user about: {user_message}
Provide clear, empathetic, evidence-based psychoeducation. Never diagnose — only educate."""
    return {"agent":"Mental Health Awareness Agent","agent_id":1,"icon":"🧠",
            "response":granite_generate(prompt),"topics_detected":[t for t in topics if t in user_message.lower()],"confidence":0.92}

def agent_emotional_support(user_message, context=None):
    ctx = ""
    if context:
        ctx = "\n".join([f"User: {c['user']}\nAgent: {c['agent']}" for c in context[-3:]])
    prompt = f"""You are the Emotional Support Agent powered by IBM Granite.
Listen with empathy, validate feelings, offer gentle encouragement.
{ctx}
Message: {user_message}
Respond with genuine empathy and warmth:"""
    tone = "distressed" if any(w in user_message.lower() for w in ["sad","lonely","hopeless","worthless","tired","overwhelmed"]) \
      else "positive" if any(w in user_message.lower() for w in ["happy","grateful","hopeful","better"]) else "reflective"
    return {"agent":"Emotional Support Agent","agent_id":2,"icon":"💙",
            "response":granite_generate(prompt),"emotional_tone":tone,"confidence":0.89}

def agent_distress_detection(user_message):
    ml = user_message.lower()
    tier = 3 if any(p in ml for p in ["suicide","kill myself","end my life","want to die","better off dead"]) \
      else 2 if any(p in ml for p in ["self-harm","hurt myself","cut myself","overdose","can't go on"]) \
      else 1 if any(p in ml for p in ["hopeless","worthless","giving up","breaking down","nothing matters","trapped"]) \
      else 0
    risk_level = {0:"low",1:"moderate",2:"high",3:"critical"}[tier]
    prompt = f"""You are the Distress Detection Agent powered by IBM Granite.
Risk level: {risk_level}. Message: {user_message}
{"Respond with IMMEDIATE compassion, provide 988 crisis resources, urge professional help." if tier>0 else "Acknowledge supportively."}"""
    flags = []
    if tier>=3: flags.append("⚠️ CRISIS: Immediate intervention recommended")
    if tier>=2: flags.append("🚨 Self-harm language detected")
    if tier>=1: flags.append("📍 Hopelessness indicators present")
    return {"agent":"Distress Detection Agent","agent_id":3,"icon":"🔍",
            "response":granite_generate(prompt),"risk_level":risk_level,"risk_tier":tier,
            "alert_flags":flags,"crisis_resources":tier>=2,"confidence":0.95}

def agent_coping_strategy(user_message, distress_tier=0):
    strategy_map = {
        "anxiety":["deep breathing","grounding (5-4-3-2-1)","progressive muscle relaxation"],
        "depression":["behavioural activation","journaling","social connection"],
        "stress":["mindfulness meditation","exercise","nature walks"],
        "burnout":["boundary setting","rest scheduling","digital detox"],
        "loneliness":["community groups","volunteering","online support communities"],
    }
    detected = []
    for k, s in strategy_map.items():
        if k in user_message.lower(): detected.extend(s)
    if not detected: detected = ["mindfulness","journaling","social support","exercise"]
    prompt = f"""You are the Coping Strategy Agent powered by IBM Granite.
Recommend 3-5 personalised evidence-based coping strategies.
Strategies: {', '.join(detected[:4])}. Distress: {"elevated" if distress_tier>=2 else "moderate" if distress_tier==1 else "mild"}.
Message: {user_message}"""
    return {"agent":"Coping Strategy Agent","agent_id":4,"icon":"🌱",
            "response":granite_generate(prompt),"strategies_identified":detected[:4],"confidence":0.88}

def agent_risk_resources(user_message, distress_tier=0, emotional_tone="neutral"):
    score = min(100, max(0, distress_tier*25 + {"distressed":15,"neutral":0,"reflective":5,"positive":-10}.get(emotional_tone,0)))
    cat = "Critical" if score>=75 else "High" if score>=50 else "Moderate" if score>=25 else "Low"
    resources = {
        "Critical":[{"name":"988 Suicide & Crisis Lifeline","contact":"Call or text 988","type":"Crisis Hotline"},
                    {"name":"Crisis Text Line","contact":"Text HOME to 741741","type":"Crisis Text"},
                    {"name":"Emergency Services","contact":"911 (US) / 999 (UK)","type":"Emergency"}],
        "High":[{"name":"NAMI Helpline","contact":"1-800-950-6264","type":"Mental Health"},
                {"name":"Crisis Text Line","contact":"Text HOME to 741741","type":"Crisis Text"},
                {"name":"SAMHSA Helpline","contact":"1-800-662-4357","type":"Substance & Mental Health"}],
        "Moderate":[{"name":"NAMI Helpline","contact":"1-800-950-6264","type":"Mental Health"},
                    {"name":"BetterHelp","contact":"betterhelp.com","type":"Online Therapy"},
                    {"name":"7 Cups","contact":"7cups.com","type":"Peer Support"}],
        "Low":[{"name":"Headspace","contact":"headspace.com","type":"Wellness App"},
               {"name":"Calm App","contact":"calm.com","type":"Wellness App"},
               {"name":"Mental Health America","contact":"mhanational.org","type":"Education"}],
    }
    prompt = f"""You are the Risk & Resources Agent powered by IBM Granite.
Risk: {cat} ({score}/100). Message: {user_message}
Write a compassionate message encouraging professional support appropriate to {cat.lower()} risk:"""
    return {"agent":"Risk Prediction & Resources Agent","agent_id":5,"icon":"🏥",
            "response":granite_generate(prompt),"risk_score":score,"risk_category":cat,
            "resources":resources[cat],"confidence":0.91}

def orchestrate(user_message, context=None):
    ml = user_message.lower()
    is_awareness = any(k in ml for k in ["what is","explain","tell me","define","anxiety","depression","burnout","mindfulness","stress","mental health","symptoms","signs of"])
    is_support   = any(k in ml for k in ["feel","feeling","i'm","im ","i am","help me","lonely","alone","sad","upset"])
    is_coping    = any(k in ml for k in ["cope","coping","strategy","tips","advice","how can i","manage","deal with"])
    is_crisis    = any(k in ml for k in ["suicide","kill","die","hurt myself","self-harm","hopeless","worthless","give up","want to die"])
    distress_result = agent_distress_detection(user_message)
    distress_tier   = distress_result["risk_tier"]
    if distress_tier >= 2: is_crisis = True
    agents_activated = ["Distress Detection"]
    results = {"distress": distress_result}
    if is_awareness or (not is_support and not is_coping and not is_crisis):
        results["awareness"] = agent_awareness(user_message)
        agents_activated.append("Awareness")
    if is_support or is_crisis or distress_tier >= 1:
        results["support"] = agent_emotional_support(user_message, context)
        agents_activated.append("Emotional Support")
    emotional_tone = results.get("support",{}).get("emotional_tone","neutral")
    if is_coping or distress_tier >= 1:
        results["coping"] = agent_coping_strategy(user_message, distress_tier)
        agents_activated.append("Coping Strategy")
    if distress_tier >= 1 or is_crisis:
        results["risk"] = agent_risk_resources(user_message, distress_tier, emotional_tone)
        agents_activated.append("Risk & Resources")
    if is_crisis or distress_tier >= 2:       primary = results["distress"]
    elif is_support:                           primary = results.get("support", results["distress"])
    elif is_awareness:                         primary = results.get("awareness", results["distress"])
    elif is_coping:                            primary = results.get("coping", results["distress"])
    else:                                      primary = results.get("awareness", results.get("support", results["distress"]))
    return {"primary_response":primary["response"],"primary_agent":primary["agent"],
            "primary_icon":primary["icon"],"agents_activated":agents_activated,
            "all_results":results,"distress_tier":distress_tier,
            "risk_level":distress_result["risk_level"],"alert_flags":distress_result["alert_flags"],
            "show_resources":distress_tier>=1}


# ── FLASK ROUTES ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message","").strip()
    if not message:
        return jsonify({"error":"Message is required"}), 400
    return jsonify(orchestrate(message, data.get("context",[])))

@app.route("/api/agent/<int:agent_id>", methods=["POST"])
def single_agent(agent_id):
    data = request.get_json(silent=True) or {}
    message = data.get("message","").strip()
    if not message:
        return jsonify({"error":"Message is required"}), 400
    fn = {1:agent_awareness,2:agent_emotional_support,3:agent_distress_detection,
          4:agent_coping_strategy,5:lambda m:agent_risk_resources(m)}.get(agent_id)
    if not fn: return jsonify({"error":"Unknown agent ID"}), 404
    return jsonify(fn(message))

@app.route("/api/health")
def health():
    return jsonify({"status":"online","app":"IBM Mental Health Assistant",
                    "model":GRANITE_MODEL_ID,"agents":5,"version":"1.0.0","demo_mode":_DEMO_MODE})


# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>IBM Mental Health Assistant · Granite AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --blue:#0f62fe;--blue-d:#0043ce;--purple:#8a3ffc;--teal:#009d9a;
  --green:#24a148;--orange:#ff832b;--red:#da1e28;--yellow:#f1c21b;
  --bg:#060910;--s1:#0d1117;--s2:#111827;--s3:#1a2233;
  --b1:#1f2937;--b2:#374151;--t1:#f0f6fc;--t2:#c9d1d9;--t3:#8b949e;--t4:#484f58;
  --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
  --r1:20px;--r2:12px;--r3:8px;
}
html{scroll-behavior:smooth}
body{font-family:var(--font);background:var(--bg);color:var(--t1);min-height:100vh;overflow-x:hidden}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}

/* NAV */
nav{position:sticky;top:0;z-index:100;height:58px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:rgba(6,9,16,.9);backdrop-filter:blur(20px);border-bottom:1px solid var(--b1)}
.nb{display:flex;align-items:center;gap:10px}
.ibm-chip{background:var(--blue);color:#fff;font-weight:800;font-size:12px;letter-spacing:2px;padding:3px 9px;border-radius:4px}
.nav-t{font-size:14px;font-weight:700}
.nav-sep{width:1px;height:28px;background:var(--b1)}
.mode-pill{display:flex;align-items:center;gap:5px;background:rgba(36,161,72,.12);border:1px solid rgba(36,161,72,.25);color:#3fb950;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600}
.live-dot{width:6px;height:6px;background:#3fb950;border-radius:50%;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* HERO */
.hero{position:relative;padding:52px 24px 40px;text-align:center;overflow:hidden;border-bottom:1px solid var(--b1)}
.hero-glow{position:absolute;inset:0;background:radial-gradient(ellipse 70% 60% at 50% 0%,rgba(15,98,254,.15),transparent 70%),radial-gradient(ellipse 40% 40% at 15% 80%,rgba(138,63,252,.08),transparent),radial-gradient(ellipse 40% 40% at 85% 80%,rgba(0,157,154,.07),transparent);pointer-events:none}
.hero-dots{position:absolute;inset:0;background-image:radial-gradient(circle,rgba(255,255,255,.04) 1px,transparent 1px);background-size:28px 28px;pointer-events:none}
.hero-inner{position:relative;z-index:1;max-width:700px;margin:auto}
.eyebrow{display:inline-flex;align-items:center;gap:7px;background:rgba(15,98,254,.1);border:1px solid rgba(15,98,254,.25);color:#79c0ff;padding:5px 14px;border-radius:30px;font-size:11px;font-weight:700;letter-spacing:.5px;margin-bottom:20px;text-transform:uppercase}
.hero h1{font-size:clamp(28px,5vw,52px);font-weight:800;line-height:1.1;letter-spacing:-1.5px;margin-bottom:14px}
.g1{background:linear-gradient(135deg,#fff,#93c5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.g2{background:linear-gradient(135deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:15px;color:var(--t3);max-width:500px;margin:0 auto 28px;line-height:1.7}
.hero-pills{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-bottom:32px}
.hpill{background:rgba(255,255,255,.04);border:1px solid var(--b1);color:var(--t3);padding:4px 13px;border-radius:20px;font-size:11px;font-weight:500}

/* AGENT CHIPS */
.agent-strip{display:flex;overflow-x:auto;gap:10px;padding:0 24px 16px;margin-top:-4px;scrollbar-width:none}
.agent-strip::-webkit-scrollbar{display:none}
.achip{display:flex;align-items:center;gap:7px;background:var(--s2);border:1px solid var(--b1);border-radius:30px;padding:7px 14px;cursor:pointer;transition:.15s;white-space:nowrap;flex-shrink:0}
.achip:hover,.achip.active{border-color:var(--chip-c,var(--blue));background:rgba(15,98,254,.08)}
.achip:nth-child(1){--chip-c:#0f62fe}.achip:nth-child(2){--chip-c:#8a3ffc}.achip:nth-child(3){--chip-c:#009d9a}.achip:nth-child(4){--chip-c:#24a148}.achip:nth-child(5){--chip-c:#ff832b}
.achip-dot{width:7px;height:7px;border-radius:50%;background:var(--chip-c,var(--blue))}
.achip-name{font-size:12px;font-weight:600;color:var(--t2)}

/* MAIN LAYOUT */
.layout{display:grid;grid-template-columns:1fr 300px;gap:16px;padding:16px 24px 32px;max-width:1300px;margin:auto}
@media(max-width:900px){.layout{grid-template-columns:1fr}.sidebar{display:none}}

/* CHAT */
.chat-wrap{background:var(--s1);border:1px solid var(--b1);border-radius:var(--r1);overflow:hidden;display:flex;flex-direction:column;height:580px}
.chat-top{padding:14px 18px;border-bottom:1px solid var(--b1);display:flex;align-items:center;justify-content:space-between;background:linear-gradient(90deg,var(--bg),var(--s1))}
.chat-top-l{display:flex;align-items:center;gap:8px}
.chat-top-icon{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:14px}
.chat-top-name{font-size:13px;font-weight:700}
.chat-top-sub{font-size:10px;color:var(--t3);margin-top:1px}
.model-tag{font-size:10px;font-family:var(--mono);color:var(--t4);background:rgba(255,255,255,.04);border:1px solid var(--b1);padding:3px 10px;border-radius:8px}

.msgs{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px}
/* Welcome */
.welcome{background:linear-gradient(135deg,rgba(15,98,254,.07),rgba(138,63,252,.07));border:1px solid rgba(15,98,254,.15);border-radius:var(--r1);padding:24px;text-align:center}
.welcome h3{font-size:16px;font-weight:700;margin-bottom:8px}
.welcome p{font-size:13px;color:var(--t3);line-height:1.6;margin-bottom:18px}
.starters{display:flex;flex-wrap:wrap;gap:7px;justify-content:center}
.starter{background:rgba(15,98,254,.1);border:1px solid rgba(15,98,254,.2);color:#79c0ff;padding:5px 13px;border-radius:20px;font-size:12px;cursor:pointer;transition:.15s;font-family:var(--font)}
.starter:hover{background:rgba(15,98,254,.2)}

/* Message bubbles */
.msg{display:flex;gap:8px;max-width:86%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.msg.user .bubble{background:linear-gradient(135deg,var(--blue),var(--blue-d));color:#fff;border-radius:var(--r1) var(--r1) 4px var(--r1)}
.msg.bot  .bubble{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r1) var(--r1) var(--r1) 4px}
.bubble{padding:11px 15px;font-size:13px;line-height:1.65;word-wrap:break-word;white-space:pre-wrap}
.avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px}
.msg.user .avatar{background:var(--blue)}
.msg.bot  .avatar{background:var(--s2);border:1px solid var(--b1)}
.msg-meta{font-size:10px;color:var(--t4);margin-top:4px;display:flex;align-items:center;gap:5px}
.msg.user .msg-meta{flex-direction:row-reverse}
.agent-chip{background:rgba(15,98,254,.1);color:#79c0ff;border:1px solid rgba(15,98,254,.15);padding:1px 7px;border-radius:8px;font-size:10px}
.risk-tag{padding:2px 8px;border-radius:8px;font-size:10px;font-weight:600}
.risk-tag.low{background:rgba(36,161,72,.15);color:#3fb950}
.risk-tag.moderate{background:rgba(241,194,27,.12);color:#e3b341}
.risk-tag.high{background:rgba(255,131,43,.12);color:#ffa657}
.risk-tag.critical{background:rgba(218,30,40,.15);color:#f85149}

/* Typing */
.typing-wrap{display:flex;gap:8px}
.typing-bubble{background:var(--s2);border:1px solid var(--b1);border-radius:var(--r1) var(--r1) var(--r1) 4px;padding:12px 16px;display:flex;gap:4px;align-items:center}
.typing-bubble span{width:6px;height:6px;border-radius:50%;background:var(--t4);animation:tb .8s infinite}
.typing-bubble span:nth-child(2){animation-delay:.15s}
.typing-bubble span:nth-child(3){animation-delay:.3s}
@keyframes tb{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}

/* Input */
.input-area{border-top:1px solid var(--b1);padding:14px 16px;background:var(--s1)}
.input-row{display:flex;gap:8px;align-items:flex-end}
.msg-input{flex:1;background:var(--bg);border:1px solid var(--b1);border-radius:var(--r2);color:var(--t1);font-family:var(--font);font-size:13px;padding:9px 13px;resize:none;outline:none;line-height:1.5;max-height:100px;transition:border-color .15s}
.msg-input:focus{border-color:var(--blue)}
.msg-input::placeholder{color:var(--t4)}
.send-btn{width:38px;height:38px;background:var(--blue);border:none;border-radius:var(--r3);color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.15s;flex-shrink:0}
.send-btn:hover{background:var(--blue-d);transform:scale(1.05)}
.send-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}
.input-note{font-size:10px;color:var(--t4);margin-top:8px;text-align:center}

/* SIDEBAR */
.sidebar{display:flex;flex-direction:column;gap:12px}
.scard{background:var(--s1);border:1px solid var(--b1);border-radius:var(--r1);overflow:hidden}
.scard-h{padding:10px 14px;background:linear-gradient(90deg,var(--bg),var(--s1));border-bottom:1px solid var(--b1);font-size:10px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.8px}
.scard-b{padding:14px}

/* Pipeline */
.pipe-item{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:var(--r3);font-size:11px;color:var(--t4);margin-bottom:4px;transition:.15s}
.pipe-item.active{background:rgba(15,98,254,.08);color:var(--t2);border:1px solid rgba(15,98,254,.15)}
.pipe-item.done{background:rgba(36,161,72,.07);color:#3fb950;border:1px solid rgba(36,161,72,.12)}
.pdot{width:7px;height:7px;border-radius:50%;background:var(--b2);flex-shrink:0;transition:.3s}
.pipe-item.active .pdot{background:var(--blue);animation:blink 1.2s infinite}
.pipe-item.done  .pdot{background:var(--green)}

/* Gauge */
.gauge-wrap{text-align:center;padding:4px 0}
.gauge-svg{display:block;margin:0 auto 8px}
.gauge-val{font-size:22px;font-weight:800}
.gauge-cat{font-size:13px;font-weight:700;margin-bottom:3px}
.gauge-note{font-size:11px;color:var(--t3)}

/* Resources */
.res-item{border:1px solid var(--b1);border-radius:var(--r3);padding:10px 11px;margin-bottom:7px;transition:.15s}
.res-item:hover{border-color:var(--b2)}
.res-name{font-size:12px;font-weight:700;margin-bottom:2px}
.res-contact{font-size:11px;color:#79c0ff;margin-bottom:3px}
.res-type{font-size:10px;color:var(--t4);background:rgba(255,255,255,.04);padding:1px 7px;border-radius:7px;display:inline-block}

/* Stats */
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid var(--b1);font-size:12px}
.stat-row:last-child{border:none}
.stat-l{color:var(--t3)}.stat-v{font-weight:600;font-family:var(--mono);color:var(--t2)}
.stat-v.blue{color:#79c0ff}.stat-v.green{color:#3fb950}.stat-v.yellow{color:#e3b341}.stat-v.red{color:#f85149}

/* Alerts */
.alert-item{font-size:11px;color:#f85149;padding:4px 0;border-top:1px solid rgba(248,81,73,.1);margin-top:5px}

/* FOOTER */
footer{background:var(--bg);border-top:1px solid var(--b1);padding:32px 24px;text-align:center}
.footer-grid{max-width:960px;margin:0 auto 24px;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:20px;text-align:left}
.ft{font-size:11px;font-weight:700;color:var(--t2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}
.fp{font-size:12px;color:var(--t3);line-height:1.7}
.ftag{display:inline-block;background:rgba(255,255,255,.04);border:1px solid var(--b1);color:var(--t3);padding:3px 10px;border-radius:10px;font-size:11px;margin:2px}
.fc{font-size:11px;color:var(--t4);border-top:1px solid var(--b1);padding-top:20px}
.fc a{color:#79c0ff;text-decoration:none}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="nb">
    <span class="ibm-chip">IBM</span>
    <span style="color:var(--t3);font-size:13px">watsonx<span style="color:var(--t2)">.ai</span></span>
    <div class="nav-sep"></div>
    <span class="nav-t">🧠 Mental Health Assistant</span>
  </div>
  <div class="mode-pill"><div class="live-dot"></div>5 Agents Active</div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-dots"></div>
  <div class="hero-inner">
    <div class="eyebrow">⚡ IBM Granite · Agentic AI Architecture</div>
    <h1><span class="g1">Mental Health</span><br/><span class="g2">AI Companion</span></h1>
    <p class="hero-sub">A safe, confidential space powered by <strong style="color:#93c5fd">IBM Granite 13B</strong> — mental health awareness, empathetic support &amp; crisis prevention.</p>
    <div class="hero-pills">
      <span class="hpill">🏆 IBM SkillsBuild</span>
      <span class="hpill">💡 Hackathon</span>
      <span class="hpill">🎓 Academic</span>
      <span class="hpill">🤖 AI Showcase</span>
      <span class="hpill">💙 Suicide Prevention</span>
    </div>
  </div>
</div>

<!-- AGENT CHIPS -->
<div class="agent-strip">
  <div class="achip active" id="ac0" onclick="switchAgent(0)"><div class="achip-dot" style="background:#f0f6fc"></div><span class="achip-name">🤖 All Agents</span></div>
  <div class="achip" id="ac1" onclick="switchAgent(1)"><div class="achip-dot"></div><span class="achip-name">🧠 Awareness</span></div>
  <div class="achip" id="ac2" onclick="switchAgent(2)"><div class="achip-dot"></div><span class="achip-name">💙 Support</span></div>
  <div class="achip" id="ac3" onclick="switchAgent(3)"><div class="achip-dot"></div><span class="achip-name">🔍 Distress Check</span></div>
  <div class="achip" id="ac4" onclick="switchAgent(4)"><div class="achip-dot"></div><span class="achip-name">🌱 Coping</span></div>
  <div class="achip" id="ac5" onclick="switchAgent(5)"><div class="achip-dot"></div><span class="achip-name">🏥 Resources</span></div>
</div>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- CHAT -->
  <div class="chat-wrap">
    <div class="chat-top">
      <div class="chat-top-l">
        <div class="chat-top-icon" id="chat-top-icon">🤖</div>
        <div>
         <div class="chat-top-name" id="chat-top-name">AI Assistant</div>
          <div class="chat-top-sub" id="active-agent">Orchestrated · All 5 agents ready</div>
        </div>
      </div>
      <span class="model-tag" id="model-tag">granite-13b</span>
    </div>
    <div class="msgs" id="msgs">
      <div class="welcome">
        <h3>👋 Welcome — You're in a safe space</h3>
        <p>I'm your AI mental health companion, powered by IBM Granite. Share anything on your mind — I'm here to listen, support, and guide you without judgment.</p>
        <div class="starters">
          <button class="starter" onclick="qs('What is anxiety?')">What is anxiety?</button>
          <button class="starter" onclick="qs('I feel overwhelmed and stressed')">I feel overwhelmed</button>
          <button class="starter" onclick="qs('What are coping strategies for depression?')">Coping strategies</button>
          <button class="starter" onclick="qs('How can mindfulness help with stress?')">Mindfulness tips</button>
          <button class="starter" onclick="qs('What are signs of burnout?')">Signs of burnout</button>
          <button class="starter" onclick="qs('I feel lonely and no one understands me')">I feel lonely</button>
        </div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-row">
        <textarea class="msg-input" id="inp" placeholder="Share what's on your mind…" rows="1" onkeydown="hk(event)" oninput="ar(this)"></textarea>
        <button class="send-btn" id="sbtn" onclick="send()">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <div class="input-note">⚠️ Not a substitute for professional care. Crisis? Call or text <strong>988</strong>.</div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- Pipeline -->
    <div class="scard">
      <div class="scard-h">⚡ Agent Pipeline</div>
      <div class="scard-b">
        <div class="pipe-item" id="p1"><div class="pdot"></div>Awareness Agent</div>
        <div class="pipe-item" id="p2"><div class="pdot"></div>Emotional Support</div>
        <div class="pipe-item" id="p3"><div class="pdot"></div>Distress Detection</div>
        <div class="pipe-item" id="p4"><div class="pdot"></div>Coping Strategy</div>
        <div class="pipe-item" id="p5"><div class="pdot"></div>Risk &amp; Resources</div>
      </div>
    </div>

    <!-- Gauge -->
    <div class="scard">
      <div class="scard-h">📊 Risk Assessment</div>
      <div class="scard-b">
        <div class="gauge-wrap">
          <svg class="gauge-svg" width="110" height="110" viewBox="0 0 110 110">
            <circle cx="55" cy="55" r="44" fill="none" stroke="#1f2937" stroke-width="10"/>
            <circle id="gauge-arc" cx="55" cy="55" r="44" fill="none" stroke="#1f2937" stroke-width="10"
              stroke-dasharray="276.46" stroke-dashoffset="276.46" stroke-linecap="round"
              transform="rotate(-90 55 55)" style="transition:stroke-dashoffset .6s ease,stroke .4s ease"/>
            <text x="55" y="51" text-anchor="middle" fill="#f0f6fc" font-size="18" font-weight="800" font-family="Inter,sans-serif" id="gauge-num">—</text>
            <text x="55" y="66" text-anchor="middle" fill="#8b949e" font-size="9" font-family="Inter,sans-serif">RISK SCORE</text>
          </svg>
          <div class="gauge-cat" id="gauge-cat" style="color:var(--t3)">Awaiting input</div>
          <div class="gauge-note" id="gauge-note">Send a message to begin</div>
        </div>
        <div id="alerts"></div>
      </div>
    </div>

    <!-- Resources -->
    <div class="scard" id="res-card" style="display:none">
      <div class="scard-h">🏥 Support Resources</div>
      <div class="scard-b" id="res-list"></div>
    </div>

    <!-- Session -->
    <div class="scard">
      <div class="scard-h">📈 Session</div>
      <div class="scard-b">
        <div class="stat-row"><span class="stat-l">Messages</span><span class="stat-v blue" id="s-msgs">0</span></div>
        <div class="stat-row"><span class="stat-l">Agents Used</span><span class="stat-v blue" id="s-agents">0</span></div>
        <div class="stat-row"><span class="stat-l">Peak Risk</span><span class="stat-v green" id="s-peak">Low</span></div>
        <div class="stat-row"><span class="stat-l">Model</span><span class="stat-v" style="font-size:10px;color:var(--t4)">Granite 13B</span></div>
      </div>
    </div>

  </div>
</div>

<!-- FOOTER -->
<footer>
  <div class="footer-grid">
    <div><div class="ft">About</div><p class="fp">IBM Mental Health Assistant demonstrates Agentic AI using IBM watsonx.ai and IBM Granite Models for mental health awareness and suicide prevention support.</p></div>
    <div><div class="ft">Technology</div><span class="ftag">IBM watsonx.ai</span><span class="ftag">IBM Granite 13B</span><span class="ftag">Agentic AI</span><span class="ftag">Python Flask</span></div>
    <div><div class="ft">Crisis Contacts</div><p class="fp">🆘 <strong style="color:#f85149">988</strong> — Suicide &amp; Crisis Lifeline<br/>📱 Text <strong style="color:#e3b341">HOME</strong> → 741741<br/>🌍 iasp.info — International</p></div>
    <div><div class="ft">Disclaimer</div><p class="fp">Educational information only — not a substitute for professional mental health care. Always consult a licensed professional.</p></div>
  </div>
  <div class="fc">Built with ❤️ for <strong>IBM SkillsBuild</strong> · Powered by <a href="https://www.ibm.com/watsonx" target="_blank">IBM watsonx.ai</a> &amp; IBM Granite Models · © 2025</div>
</footer>

<script>
/* ── Agent personas ── */
const AGENTS = {
  0: { name:'All Agents',       sub:'Orchestrated · All 5 agents ready',       icon:'🤖', color:'#f0f6fc',
       greeting:"Hi! I'm your Mental Health AI Companion, powered by IBM Granite. All five specialized agents are standing by.\n\nYou can talk to me about anything — mental health topics, how you're feeling, coping strategies, or getting professional help. What's on your mind today?",
       placeholder:"Ask me anything about mental health…" },
  1: { name:'Awareness Agent',  sub:'Mental Health Awareness · IBM Granite',    icon:'🧠', color:'#0f62fe',
       greeting:"Hello! I'm your Mental Health Awareness Agent, powered by IBM Granite 🧠\n\nI'm here to help you understand mental health topics clearly and compassionately — things like anxiety, depression, stress, burnout, mindfulness, and more.\n\nWhat would you like to learn about today?",
       placeholder:"Ask about anxiety, depression, burnout, mindfulness…" },
  2: { name:'Support Agent',    sub:'Emotional Support · Active Listening',     icon:'💙', color:'#8a3ffc',
       greeting:"Hey, I'm really glad you're here 💙\n\nI'm your Emotional Support Agent. This is a completely safe, judgment-free space. You don't have to explain yourself or have the right words — just share what's going on, and I'll listen.\n\nHow are you feeling right now?",
       placeholder:"Share how you're feeling, I'm listening…" },
  3: { name:'Distress Check',   sub:'Crisis Detection · Safety First',          icon:'🔍', color:'#009d9a',
       greeting:"Hi, I'm here with you 🔍\n\nI'm the Distress Detection Agent. My job is to check in with you and make sure you're safe and supported.\n\nSometimes things get really hard and it helps to talk. You can be completely honest with me — nothing you say will be judged.\n\nHow are you doing today, really?",
       placeholder:"Tell me honestly how you're doing…" },
  4: { name:'Coping Agent',     sub:'Coping Strategies · Evidence-Based',       icon:'🌱', color:'#24a148',
       greeting:"Hi there! I'm your Coping Strategy Agent 🌱\n\nI'm powered by IBM Granite and I specialise in practical, evidence-based techniques to help you manage stress, anxiety, low mood, and difficult emotions.\n\nTell me what you're currently struggling with and I'll suggest strategies that actually work.",
       placeholder:"Tell me what you're struggling with…" },
  5: { name:'Resources Agent',  sub:'Professional Support · Risk Assessment',   icon:'🏥', color:'#ff832b',
       greeting:"Hello! I'm your Professional Resources Agent 🏥\n\nI can help you find the right mental health support — whether that's crisis hotlines, therapists, online counselling, or wellness apps.\n\nTell me a bit about what you're going through and I'll point you toward the most helpful resources.",
       placeholder:"Tell me what kind of support you need…" },
};

const S = { msgs:0, agents:0, peakTier:0, ctx:[], activeAgent:0, busy:false };
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br/>');
const now = () => new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});

/* ── Resize textarea ── */
function ar(el){ el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,100)+'px'; }
function hk(ev){ if(ev.key==='Enter'&&!ev.shiftKey){ ev.preventDefault(); send(); } }

/* ── Switch active agent ── */
function switchAgent(id) {
  if(S.busy) return;
  S.activeAgent = id;
  S.ctx = []; // fresh context per agent
  const a = AGENTS[id];

  // highlight chip
  document.querySelectorAll('.achip').forEach(c => c.classList.remove('active'));
  const chip = document.getElementById('ac'+id);
  if(chip) chip.classList.add('active');

  // update header
  document.getElementById('chat-top-icon').textContent = a.icon;
  document.getElementById('chat-top-icon').style.background = `linear-gradient(135deg, ${a.color}99, ${a.color}44)`;
  document.getElementById('chat-top-name').textContent = a.name;
  document.getElementById('active-agent').textContent  = a.sub;
  document.getElementById('inp').placeholder = a.placeholder;

  // clear messages and show greeting with typewriter
  const msgs = document.getElementById('msgs');
  msgs.innerHTML = '';
  typewriterMsg(a.icon, a.greeting, a.name);
}

/* ── Typewriter effect for bot messages ── */
function typewriterMsg(icon, fullText, agentName) {
  const msgs = document.getElementById('msgs');
  const wrapper = document.createElement('div');
  wrapper.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.innerHTML = `<span>${now()}</span><span class="agent-chip">${esc(agentName)}</span>`;
  const inner = document.createElement('div');
  wrapper.innerHTML = `<div class="avatar">${icon}</div>`;
  wrapper.appendChild(Object.assign(document.createElement('div'), {
    innerHTML: ''
  }));
  wrapper.lastChild.appendChild(bubble);
  wrapper.lastChild.appendChild(meta);
  msgs.appendChild(wrapper);

  // Stream text character by character (fast)
  const lines = fullText.split('\n');
  let lineIdx=0, charIdx=0;
  let displayed = '';
  function step() {
    if(lineIdx >= lines.length){ msgs.scrollTop=msgs.scrollHeight; return; }
    const line = lines[lineIdx];
    if(charIdx < line.length){
      displayed += esc(line[charIdx]);
      charIdx++;
      bubble.innerHTML = displayed.replace(/\n/g,'<br/>');
      msgs.scrollTop = msgs.scrollHeight;
      setTimeout(step, 8); // fast typewriter
    } else {
      displayed += '<br/>';
      lineIdx++; charIdx=0;
      bubble.innerHTML = displayed;
      msgs.scrollTop = msgs.scrollHeight;
      setTimeout(step, charIdx===0&&line===''?30:50);
    }
  }
  step();
}

/* ── Add a message bubble instantly ── */
function addMsg(role, text, meta={}) {
  const msgs = document.getElementById('msgs');
  const d = document.createElement('div');
  d.className = 'msg ' + (role==='user' ? 'user' : 'bot');
  const icon = role==='user' ? '👤' : (meta.icon || AGENTS[S.activeAgent].icon);
  const riskHtml = meta.risk && role!=='user'
    ? `<span class="risk-tag ${meta.risk}" style="margin-left:6px">${meta.risk.toUpperCase()}</span>` : '';
  d.innerHTML = `<div class="avatar">${icon}</div><div>
    <div class="bubble">${esc(text)}</div>
    <div class="msg-meta"><span>${now()}</span>
      ${meta.agent?`<span class="agent-chip">${esc(meta.agent)}</span>`:''}
      ${riskHtml}
    </div></div>`;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}

/* ── Typing indicator ── */
function showTyping() {
  const msgs = document.getElementById('msgs');
  const d = document.createElement('div');
  d.id='typing'; d.className='msg bot';
  const icon = AGENTS[S.activeAgent].icon;
  d.innerHTML=`<div class="avatar">${icon}</div><div class="typing-bubble"><span></span><span></span><span></span></div>`;
  msgs.appendChild(d); msgs.scrollTop=msgs.scrollHeight;
}
function hideTyping(){ const t=document.getElementById('typing'); if(t) t.remove(); }

/* ── Pipeline animation ── */
function animatePipe(agents) {
  const m={'Awareness':'p1','Emotional Support':'p2','Distress Detection':'p3','Coping Strategy':'p4','Risk & Resources':'p5'};
  ['p1','p2','p3','p4','p5'].forEach(id=>{document.getElementById(id).classList.remove('active','done')});
  let delay=0;
  agents.forEach(a=>{
    const id=m[a]; if(!id) return;
    setTimeout(()=>document.getElementById(id).classList.add('active'), delay);
    setTimeout(()=>{ document.getElementById(id).classList.remove('active'); document.getElementById(id).classList.add('done'); }, delay+600);
    delay+=300;
  });
}

/* ── Risk gauge ── */
function updateGauge(level, tier) {
  const colors={low:'#3fb950',moderate:'#e3b341',high:'#ffa657',critical:'#f85149'};
  const scores={low:12,moderate:40,high:68,critical:92};
  const sc=scores[level]||0, color=colors[level]||'#374151', circ=276.46;
  const arc=document.getElementById('gauge-arc');
  arc.style.strokeDashoffset = circ-(sc/100)*circ;
  arc.style.stroke = color;
  document.getElementById('gauge-num').textContent = sc||'—';
  document.getElementById('gauge-num').setAttribute('fill', sc?color:'#f0f6fc');
  const cats={low:'Low Risk',moderate:'Moderate',high:'High Risk',critical:'Critical'};
  const catEl=document.getElementById('gauge-cat');
  catEl.textContent=cats[level]||'—'; catEl.style.color=color;
  const notes={low:'No immediate concern',moderate:'Monitor & consider support',high:'Seek professional help soon',critical:'⚠️ Immediate help needed'};
  document.getElementById('gauge-note').textContent = notes[level]||'';
  if(tier>S.peakTier){ S.peakTier=tier; const el=document.getElementById('s-peak'); el.textContent=cats[level]||'Low'; el.className='stat-v '+(tier===0?'green':tier===1?'yellow':'red'); }
}

/* ── Show alerts ── */
function showAlerts(flags) {
  document.getElementById('alerts').innerHTML = flags.map(f=>`<div class="alert-item">${esc(f)}</div>`).join('');
}

/* ── Show resources ── */
function showResources(resources) {
  if(!resources||!resources.length) return;
  document.getElementById('res-card').style.display='block';
  document.getElementById('res-list').innerHTML = resources.map(r=>`
    <div class="res-item">
      <div class="res-name">${esc(r.name)}</div>
      <div class="res-contact">${esc(r.contact)}</div>
      <span class="res-type">${esc(r.type)}</span>
    </div>`).join('');
}

/* ── SEND ── */
async function send() {
  if(S.busy) return;
  const inp = document.getElementById('inp');
  const btn = document.getElementById('sbtn');
  const text = inp.value.trim();
  if(!text) return;

  addMsg('user', text);
  inp.value=''; inp.style.height='auto';
  btn.disabled=true; S.busy=true;
  S.msgs++; document.getElementById('s-msgs').textContent=S.msgs;
  showTyping();

  try {
    // If a specific agent is selected (1-5), call that agent directly
    const endpoint = S.activeAgent > 0 ? `/api/agent/${S.activeAgent}` : '/api/chat';
    const body = S.activeAgent > 0
      ? JSON.stringify({message:text})
      : JSON.stringify({message:text, context:S.ctx});

    const r = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body});
    const d = await r.json();
    hideTyping();

    if(d.error){ addMsg('bot','⚠️ '+d.error); return; }

    // For orchestrated mode
    if(S.activeAgent === 0) {
      addMsg('bot', d.primary_response, {icon:d.primary_icon, agent:d.primary_agent, risk:d.risk_level});
      animatePipe(d.agents_activated||[]);
      updateGauge(d.risk_level||'low', d.distress_tier||0);
      showAlerts(d.alert_flags||[]);
      if(d.show_resources && d.all_results?.risk) showResources(d.all_results.risk.resources);
      // Show coping/risk follow-up only for elevated distress
      const res = d.all_results||{};
      if(d.distress_tier>=2 && res.coping)
        setTimeout(()=>addMsg('bot', res.coping.response, {icon:'🌱', agent:'Coping Strategy Agent'}), 400);
      if(d.distress_tier>=2 && res.risk)
        setTimeout(()=>addMsg('bot', res.risk.response, {icon:'🏥', agent:'Risk & Resources Agent'}), 800);
      S.ctx.push({user:text, agent:d.primary_response});
      if(S.ctx.length>10) S.ctx.shift();
      S.agents += (d.agents_activated||[]).length;
      document.getElementById('s-agents').textContent = S.agents;
    } else {
      // Single agent mode — just show the response naturally
      const a = AGENTS[S.activeAgent];
      addMsg('bot', d.response, {icon:a.icon, agent:a.name, risk:d.risk_level});
      animatePipe(['Distress Detection', ['Awareness','Emotional Support','Distress Detection','Coping Strategy','Risk & Resources'][S.activeAgent-1]]);
      updateGauge(d.risk_level||'low', d.risk_tier||0);
      showAlerts(d.alert_flags||[]);
      if(d.resources) showResources(d.resources);
      S.agents++;
      document.getElementById('s-agents').textContent = S.agents;
      // Keep context for this agent conversation
      S.ctx.push({user:text, agent:d.response});
      if(S.ctx.length>10) S.ctx.shift();
    }

  } catch(err) {
    hideTyping();
    addMsg('bot','⚠️ Connection error — make sure the server is running on http://localhost:5000');
    console.error(err);
  }

  btn.disabled=false; S.busy=false; inp.focus();
}

/* ── Init: show welcome from All Agents ── */
window.addEventListener('DOMContentLoaded', ()=>{ switchAgent(0); });
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n IBM Mental Health Assistant · ibm/granite-13b-instruct-v2")
    print(f" Demo mode: {_DEMO_MODE}")
    print(" Open: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
