"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          IBM watsonx.ai  ·  Mental Health Assistant                         ║
║          Powered by IBM Granite Models  ·  Agentic AI Architecture          ║
║          IBM SkillsBuild · Hackathon · Academic · AI Showcase               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Five Specialized Agents:
  1. Mental Health Awareness Agent   – Education & psychoeducation
  2. Emotional Support Agent         – Empathetic active listening
  3. Distress Detection Agent        – Crisis signal recognition
  4. Coping Strategy Agent           – Personalised coping plans
  5. Risk Prediction & Resources     – Risk scoring + professional help
"""

import os
import json
import re
from flask import Flask, request, jsonify, render_template_string

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  –  set your IBM watsonx.ai credentials via environment vars
# or fill in your values directly below.
# ─────────────────────────────────────────────────────────────────────────────
WATSONX_API_KEY  = os.getenv("WATSONX_API_KEY",  "your-ibm-watsonx-api-key")
WATSONX_URL      = os.getenv("WATSONX_URL",      "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT  = os.getenv("WATSONX_PROJECT",  "your-project-id")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE – when real credentials are not set, rich pre-written responses
# simulate IBM Granite output so the full UI can be showcased immediately.
# ─────────────────────────────────────────────────────────────────────────────
_DEMO_MODE = (
    WATSONX_API_KEY in ("", "your-ibm-watsonx-api-key") or
    WATSONX_PROJECT in ("", "your-project-id")
)

DEMO_RESPONSES = {
    "awareness": (
        "Great question! Anxiety is a natural emotional response to stress or perceived threats. "
        "It becomes a concern when it is persistent, excessive, or interferes with daily life.\n\n"
        "**Common symptoms include:**\n"
        "• Excessive worry or fear\n"
        "• Restlessness and irritability\n"
        "• Difficulty concentrating\n"
        "• Physical symptoms: rapid heartbeat, sweating, trembling\n"
        "• Avoidance of anxiety-triggering situations\n\n"
        "**Types include:** Generalized Anxiety Disorder (GAD), Social Anxiety, Panic Disorder, and Phobias.\n\n"
        "Mindfulness, therapy (especially CBT), exercise, and breathing techniques are proven effective. "
        "Always consult a mental health professional for a proper assessment. You are not alone — "
        "anxiety affects over 284 million people worldwide. 💙"
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
        "Thank you for trusting me with something so personal. I want you to know that your pain "
        "is real and you deserve support.\n\n"
        "Please know that you are not alone in this moment. Reaching out — even to an AI — "
        "is a meaningful and brave first step.\n\n"
        "🆘 **If you are in crisis right now, please contact:**\n"
        "• **988 Suicide & Crisis Lifeline** — Call or text **988** (US, available 24/7)\n"
        "• **Crisis Text Line** — Text **HOME** to **741741**\n"
        "• **Emergency services** — Call **911** (US) or **999** (UK)\n\n"
        "You matter. Your life has value. Please reach out to a professional who can truly help. 💙"
    ),
    "coping": (
        "Here are 5 evidence-based coping strategies you can try right now:\n\n"
        "**1. 5-4-3-2-1 Grounding Technique**\n"
        "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. "
        "This anchors you to the present moment.\n\n"
        "**2. Box Breathing (4-4-4-4)**\n"
        "Inhale for 4 counts → Hold for 4 → Exhale for 4 → Hold for 4. Repeat 4 times. "
        "This activates your parasympathetic nervous system.\n\n"
        "**3. Journaling**\n"
        "Write freely for 10 minutes without judgment. Externalising thoughts reduces their power.\n\n"
        "**4. Progressive Muscle Relaxation**\n"
        "Tense each muscle group for 5 seconds, then release. Work from feet to forehead.\n\n"
        "**5. Reach Out**\n"
        "Call or text someone you trust. Social connection is one of the most powerful stress buffers we have. 🌱"
    ),
    "risk": (
        "I want to gently acknowledge how much you've shared today — that takes real strength.\n\n"
        "Based on our conversation, it may be helpful to speak with a licensed mental health "
        "professional. This isn't a sign of weakness — it's one of the most powerful acts of "
        "self-care you can take.\n\n"
        "**Steps you can take today:**\n"
        "• Contact your primary care doctor for a referral\n"
        "• Search for a therapist at **psychologytoday.com**\n"
        "• Try **BetterHelp** or **7 Cups** for immediate online support\n"
        "• Call **NAMI Helpline: 1-800-950-6264**\n\n"
        "You deserve consistent, professional care. Taking this step is a sign of incredible self-awareness. 🏥"
    ),
    "general": (
        "Thank you for sharing that with me. Mental health is something that touches all of us, "
        "and conversations like this one matter deeply.\n\n"
        "Whether you're looking for information, emotional support, coping strategies, or "
        "professional resources — I'm here to help guide you.\n\n"
        "What would be most helpful for you right now? You can ask me about specific mental "
        "health topics, share how you're feeling, or ask for practical coping techniques. 💙"
    ),
}

def demo_granite_generate(prompt: str) -> str:
    """Return rich demo responses based on prompt content when credentials are not set."""
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

# ─────────────────────────────────────────────────────────────────────────────
# Shared IBM Granite Model Client
# ─────────────────────────────────────────────────────────────────────────────
def get_granite_client():
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT,
        params={
            Params.MAX_NEW_TOKENS: 600,
            Params.MIN_NEW_TOKENS: 30,
            Params.TEMPERATURE:    0.7,
            Params.TOP_P:          0.9,
            Params.REPETITION_PENALTY: 1.1,
        }
    )

def granite_generate(prompt: str) -> str:
    """Call IBM Granite (or demo fallback) and return the generated text."""
    if _DEMO_MODE:
        return demo_granite_generate(prompt)
    try:
        client = get_granite_client()
        result = client.generate_text(prompt=prompt)
        return result.strip() if result else "I'm here to help. Could you share a bit more?"
    except Exception as exc:
        # Fallback to demo responses if live call fails
        return demo_granite_generate(prompt)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Mental Health Awareness Agent
# ═════════════════════════════════════════════════════════════════════════════
def agent_awareness(user_message: str) -> dict:
    """
    Educates users on anxiety, depression, stress, burnout, mindfulness,
    emotional wellness, and self-care using IBM Granite.
    """
    topics = ["anxiety", "depression", "stress", "burnout",
              "mindfulness", "emotional wellness", "self-care",
              "mental health", "panic", "trauma", "ptsd", "ocd",
              "bipolar", "schizophrenia", "eating disorder", "phobia"]

    prompt = f"""You are the Mental Health Awareness Agent — a compassionate, 
evidence-based psychoeducation assistant powered by IBM Granite.

Your role: Educate users about mental health topics including anxiety, 
depression, stress, burnout, mindfulness, emotional wellness, and self-care.

Guidelines:
- Provide clear, accurate, and empathetic educational information
- Use plain language accessible to non-medical audiences
- Always remind users that professional help is available when appropriate
- Never diagnose — only educate

User Question: {user_message}

Provide a warm, informative, and structured educational response:"""

    response = granite_generate(prompt)
    detected_topics = [t for t in topics if t in user_message.lower()]

    return {
        "agent": "Mental Health Awareness Agent",
        "agent_id": 1,
        "icon": "🧠",
        "response": response,
        "topics_detected": detected_topics,
        "confidence": 0.92
    }


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Emotional Support Agent
# ═════════════════════════════════════════════════════════════════════════════
def agent_emotional_support(user_message: str, context: list = None) -> dict:
    """
    Provides empathetic, non-judgmental emotional support and active listening.
    """
    context_str = ""
    if context:
        context_str = "\n".join([f"User: {c['user']}\nAgent: {c['agent']}" for c in context[-3:]])
        context_str = f"\nConversation history:\n{context_str}\n"

    prompt = f"""You are the Emotional Support Agent — a warm, empathetic AI companion 
powered by IBM Granite. You practice active listening and provide compassionate support.

Your role:
- Listen with empathy and without judgment
- Validate the user's feelings
- Offer gentle encouragement and emotional comfort
- Ask thoughtful follow-up questions to understand their experience
- Remind users they are not alone
{context_str}
Current message: {user_message}

Respond with genuine empathy, warmth, and emotional intelligence:"""

    response = granite_generate(prompt)

    # Detect emotional tone
    positive_words = ["happy", "grateful", "hopeful", "better", "good", "okay"]
    negative_words = ["sad", "lonely", "hopeless", "worthless", "empty", "numb",
                      "tired", "exhausted", "overwhelmed", "scared", "anxious"]
    neutral_words  = ["confused", "unsure", "wondering", "thinking", "curious"]

    msg_lower = user_message.lower()
    if any(w in msg_lower for w in negative_words):
        tone = "distressed"
    elif any(w in msg_lower for w in positive_words):
        tone = "positive"
    elif any(w in msg_lower for w in neutral_words):
        tone = "neutral"
    else:
        tone = "reflective"

    return {
        "agent": "Emotional Support Agent",
        "agent_id": 2,
        "icon": "💙",
        "response": response,
        "emotional_tone": tone,
        "confidence": 0.89
    }


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Distress Detection Agent
# ═════════════════════════════════════════════════════════════════════════════
def agent_distress_detection(user_message: str) -> dict:
    """
    Detects distress signals, crisis indicators, and emotional red flags.
    Returns a risk level and appropriate response.
    """
    # Tiered keyword detection
    crisis_tier1 = [
        "suicide", "kill myself", "end my life", "don't want to live",
        "want to die", "better off dead", "take my life", "no reason to live"
    ]
    crisis_tier2 = [
        "self-harm", "hurt myself", "cut myself", "harm myself",
        "overdose", "disappear forever", "can't go on"
    ]
    distress_tier3 = [
        "hopeless", "worthless", "no one cares", "giving up", "can't cope",
        "breaking down", "falling apart", "nothing matters", "trapped"
    ]

    msg_lower = user_message.lower()
    tier = 0
    if any(phrase in msg_lower for phrase in crisis_tier1):
        tier = 3
    elif any(phrase in msg_lower for phrase in crisis_tier2):
        tier = 2
    elif any(phrase in msg_lower for phrase in distress_tier3):
        tier = 1

    risk_levels = {0: "low", 1: "moderate", 2: "high", 3: "critical"}
    risk_level  = risk_levels[tier]

    if tier == 0:
        prompt = f"""You are the Distress Detection Agent powered by IBM Granite.
Analyze this message for any subtle emotional distress indicators and respond supportively.
Message: {user_message}
Provide a brief supportive acknowledgment:"""
    else:
        prompt = f"""You are the Distress Detection Agent powered by IBM Granite.
This message shows signs of {risk_level} distress. Respond with IMMEDIATE compassion,
validate their pain, encourage them to reach out for professional help urgently,
and provide crisis hotline information (988 Suicide & Crisis Lifeline in the US).

Message: {user_message}

Respond with urgency, warmth, and clear guidance to seek immediate help:"""

    response = granite_generate(prompt)

    alert_flags = []
    if tier >= 3:
        alert_flags.append("⚠️ CRISIS: Immediate intervention recommended")
    if tier >= 2:
        alert_flags.append("🚨 Self-harm language detected")
    if tier >= 1:
        alert_flags.append("📍 Hopelessness indicators present")

    return {
        "agent": "Distress Detection Agent",
        "agent_id": 3,
        "icon": "🔍",
        "response": response,
        "risk_level": risk_level,
        "risk_tier": tier,
        "alert_flags": alert_flags,
        "crisis_resources": tier >= 2,
        "confidence": 0.95
    }


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Coping Strategy Agent
# ═════════════════════════════════════════════════════════════════════════════
def agent_coping_strategy(user_message: str, distress_tier: int = 0) -> dict:
    """
    Recommends personalised, evidence-based coping strategies.
    """
    # Map keywords to strategy categories
    strategy_map = {
        "anxiety":     ["deep breathing", "grounding (5-4-3-2-1)", "progressive muscle relaxation"],
        "depression":  ["behavioural activation", "journaling", "social connection", "sunlight exposure"],
        "stress":      ["time management", "mindfulness meditation", "exercise", "nature walks"],
        "burnout":     ["boundary setting", "rest scheduling", "hobby re-engagement", "digital detox"],
        "sleep":       ["sleep hygiene", "bedtime routine", "screen-free wind-down"],
        "anger":       ["box breathing", "physical exercise", "cool-down space"],
        "loneliness":  ["community groups", "volunteering", "online support communities"],
        "grief":       ["grief journaling", "memory rituals", "bereavement support groups"],
    }

    detected_strategies = []
    msg_lower = user_message.lower()
    for keyword, strategies in strategy_map.items():
        if keyword in msg_lower:
            detected_strategies.extend(strategies)

    if not detected_strategies:
        detected_strategies = ["mindfulness", "journaling", "social support", "exercise"]

    strategies_str = ", ".join(detected_strategies[:4])

    prompt = f"""You are the Coping Strategy Agent powered by IBM Granite.
Your role: Recommend personalised, evidence-based coping strategies that are 
practical, actionable, and tailored to the user's situation.

Relevant strategies to consider: {strategies_str}
Distress level: {"elevated" if distress_tier >= 2 else "moderate" if distress_tier == 1 else "mild"}

User message: {user_message}

Provide 3-5 concrete, step-by-step coping strategies with brief explanations.
Make them immediately actionable and compassionate in tone:"""

    response = granite_generate(prompt)

    return {
        "agent": "Coping Strategy Agent",
        "agent_id": 4,
        "icon": "🌱",
        "response": response,
        "strategies_identified": detected_strategies[:4],
        "confidence": 0.88
    }


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Risk Prediction & Professional Resources Agent
# ═════════════════════════════════════════════════════════════════════════════
def agent_risk_resources(user_message: str, distress_tier: int = 0,
                          emotional_tone: str = "neutral") -> dict:
    """
    Predicts potential mental health risk level and connects users with
    professional support resources.
    """
    # Compute composite risk score (0–100)
    base_score = distress_tier * 25
    tone_bonus  = {"distressed": 15, "neutral": 0, "reflective": 5, "positive": -10}.get(emotional_tone, 0)
    risk_score  = min(100, max(0, base_score + tone_bonus))

    risk_category = (
        "Critical"  if risk_score >= 75 else
        "High"      if risk_score >= 50 else
        "Moderate"  if risk_score >= 25 else
        "Low"
    )

    resources = {
        "Critical": [
            {"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988 (US)",     "type": "Crisis Hotline"},
            {"name": "Crisis Text Line",               "contact": "Text HOME to 741741",        "type": "Crisis Text"},
            {"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "type": "International"},
            {"name": "Emergency Services",             "contact": "Call 911 (US) / 999 (UK)",   "type": "Emergency"},
        ],
        "High": [
            {"name": "NAMI Helpline",                  "contact": "1-800-950-NAMI (6264)",      "type": "Mental Health"},
            {"name": "Crisis Text Line",               "contact": "Text HOME to 741741",        "type": "Crisis Text"},
            {"name": "SAMHSA National Helpline",       "contact": "1-800-662-4357",             "type": "Substance & Mental Health"},
            {"name": "Psychology Today Therapist Finder", "contact": "https://www.psychologytoday.com/us/therapists", "type": "Therapy"},
        ],
        "Moderate": [
            {"name": "NAMI Helpline",                  "contact": "1-800-950-NAMI (6264)",      "type": "Mental Health"},
            {"name": "BetterHelp",                     "contact": "https://www.betterhelp.com", "type": "Online Therapy"},
            {"name": "7 Cups",                         "contact": "https://www.7cups.com",      "type": "Peer Support"},
            {"name": "Headspace (Meditation)",         "contact": "https://www.headspace.com",  "type": "Wellness App"},
        ],
        "Low": [
            {"name": "Headspace (Meditation)",         "contact": "https://www.headspace.com",  "type": "Wellness App"},
            {"name": "Calm App",                       "contact": "https://www.calm.com",       "type": "Wellness App"},
            {"name": "Mental Health America",          "contact": "https://mhanational.org",    "type": "Education"},
            {"name": "NAMI Education Programs",        "contact": "https://www.nami.org/Support-Education", "type": "Education"},
        ]
    }

    prompt = f"""You are the Risk Prediction & Professional Resources Agent powered by IBM Granite.
Based on the conversation, the assessed risk level is: {risk_category} (score: {risk_score}/100).

Your role:
- Acknowledge the user's situation with compassion
- Explain the importance of professional support without alarming them
- Encourage them to reach out to mental health professionals
- Remind them that seeking help is a sign of strength

User message: {user_message}

Write a compassionate, encouraging message about seeking professional support
appropriate to a {risk_category.lower()} risk level:"""

    response = granite_generate(prompt)

    return {
        "agent": "Risk Prediction & Resources Agent",
        "agent_id": 5,
        "icon": "🏥",
        "response": response,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "resources": resources[risk_category],
        "confidence": 0.91
    }


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Routes messages to the right agent(s)
# ═════════════════════════════════════════════════════════════════════════════
def orchestrate(user_message: str, context: list = None) -> dict:
    """
    Intelligent multi-agent orchestrator.
    Analyses the message, activates relevant agents, and synthesises results.
    """
    msg_lower = user_message.lower()

    # Intent classification
    awareness_keywords  = ["what is", "explain", "tell me about", "how does", "define",
                           "anxiety", "depression", "burnout", "mindfulness", "stress",
                           "mental health", "bipolar", "ptsd", "ocd", "symptoms", "signs of"]
    support_keywords    = ["feel", "feeling", "i'm", "im ", "i am", "help me", "support",
                           "listen", "talk", "need", "lonely", "alone", "sad", "upset"]
    coping_keywords     = ["cope", "coping", "strategy", "strategies", "tips", "advice",
                           "how can i", "what can i do", "manage", "deal with", "handle"]
    crisis_keywords     = ["suicide", "kill", "die", "hurt myself", "self-harm", "hopeless",
                           "worthless", "no point", "give up", "can't go on", "want to die"]

    is_awareness = any(k in msg_lower for k in awareness_keywords)
    is_support   = any(k in msg_lower for k in support_keywords)
    is_coping    = any(k in msg_lower for k in coping_keywords)
    is_crisis    = any(k in msg_lower for k in crisis_keywords)

    # Always run distress detection
    distress_result = agent_distress_detection(user_message)
    distress_tier   = distress_result["risk_tier"]

    # Override flags if crisis detected
    if distress_tier >= 2:
        is_crisis = True

    agents_activated = ["Distress Detection"]
    results = {"distress": distress_result}

    # Activate relevant agents based on intent
    if is_awareness or (not is_support and not is_coping and not is_crisis):
        awareness_result = agent_awareness(user_message)
        results["awareness"] = awareness_result
        agents_activated.append("Awareness")

    if is_support or is_crisis or distress_tier >= 1:
        support_result = agent_emotional_support(user_message, context)
        results["support"] = support_result
        agents_activated.append("Emotional Support")
        emotional_tone = support_result.get("emotional_tone", "neutral")
    else:
        emotional_tone = "neutral"

    if is_coping or distress_tier >= 1:
        coping_result = agent_coping_strategy(user_message, distress_tier)
        results["coping"] = coping_result
        agents_activated.append("Coping Strategy")

    # Always run risk/resources for elevated distress
    if distress_tier >= 1 or is_crisis:
        risk_result = agent_risk_resources(user_message, distress_tier, emotional_tone)
        results["risk"] = risk_result
        agents_activated.append("Risk & Resources")

    # Determine primary agent response to surface to user
    if is_crisis or distress_tier >= 2:
        primary = results["distress"]
    elif is_support:
        primary = results.get("support", results["distress"])
    elif is_awareness:
        primary = results.get("awareness", results["distress"])
    elif is_coping:
        primary = results.get("coping", results["distress"])
    else:
        primary = results.get("awareness", results.get("support", results["distress"]))

    return {
        "primary_response": primary["response"],
        "primary_agent":    primary["agent"],
        "primary_icon":     primary["icon"],
        "agents_activated": agents_activated,
        "all_results":      results,
        "distress_tier":    distress_tier,
        "risk_level":       distress_result["risk_level"],
        "alert_flags":      distress_result["alert_flags"],
        "show_resources":   distress_tier >= 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    context = data.get("context", [])

    if not message:
        return jsonify({"error": "Message is required"}), 400

    result = orchestrate(message, context)
    return jsonify(result)


@app.route("/api/agent/<int:agent_id>", methods=["POST"])
def single_agent(agent_id: int):
    """Direct endpoint to query a specific agent."""
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    agents = {
        1: agent_awareness,
        2: agent_emotional_support,
        3: agent_distress_detection,
        4: agent_coping_strategy,
        5: lambda m: agent_risk_resources(m),
    }

    fn = agents.get(agent_id)
    if not fn:
        return jsonify({"error": "Unknown agent ID"}), 404

    return jsonify(fn(message))


@app.route("/api/health")
def health():
    return jsonify({
        "status":    "online",
        "app":       "IBM Mental Health Assistant",
        "model":     GRANITE_MODEL_ID,
        "agents":    5,
        "version":   "1.0.0",
        "demo_mode": _DEMO_MODE,
    })


# ─────────────────────────────────────────────────────────────────────────────
# HTML Template  –  Full Single-Page App
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>IBM Mental Health Assistant · Powered by Granite</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  /* ── Reset & Base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --ibm-blue:       #0f62fe;
    --ibm-blue-dark:  #0043ce;
    --ibm-blue-light: #d0e2ff;
    --ibm-purple:     #8a3ffc;
    --ibm-teal:       #009d9a;
    --ibm-green:      #24a148;
    --ibm-yellow:     #f1c21b;
    --ibm-red:        #da1e28;
    --ibm-orange:     #ff832b;
    --cool-gray-10:   #f2f4f8;
    --cool-gray-20:   #dde1e7;
    --cool-gray-60:   #697077;
    --cool-gray-90:   #21272a;
    --white:          #ffffff;
    --gradient-hero:  linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    --gradient-card:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --shadow-sm:      0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.08);
    --shadow-md:      0 4px 16px rgba(0,0,0,.15);
    --shadow-lg:      0 16px 48px rgba(0,0,0,.22);
    --radius:         16px;
    --radius-sm:      8px;
    --font:           'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    --mono:           'IBM Plex Mono', 'Fira Code', monospace;
  }
  html { scroll-behavior: smooth; }
  body {
    font-family: var(--font);
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #161b22; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

  /* ── Header ── */
  .header {
    background: linear-gradient(90deg, #010409 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
    padding: 0 24px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(10px);
  }
  .header-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .ibm-logo {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.5px;
  }
  .ibm-logo .logo-box {
    background: var(--ibm-blue);
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
  }
  .header-title {
    font-size: 15px;
    font-weight: 600;
    color: #e6edf3;
  }
  .header-subtitle {
    font-size: 11px;
    color: #8b949e;
    margin-top: 1px;
  }
  .header-divider {
    width: 1px;
    height: 32px;
    background: #30363d;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(36, 161, 72, .15);
    border: 1px solid rgba(36, 161, 72, .3);
    color: #3fb950;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    background: #3fb950;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: .4; }
  }

  /* ── Hero Banner ── */
  .hero {
    background: var(--gradient-hero);
    padding: 56px 24px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 60% 50% at 20% 50%, rgba(15,98,254,.25) 0%, transparent 70%),
      radial-gradient(ellipse 60% 50% at 80% 50%, rgba(138,63,252,.2)  0%, transparent 70%);
  }
  .hero-content { position: relative; z-index: 1; max-width: 800px; margin: auto; }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(15,98,254,.2);
    border: 1px solid rgba(15,98,254,.4);
    color: #74b9ff;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 20px;
    letter-spacing: 0.5px;
  }
  .hero h1 {
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #74b9ff 50%, #a29bfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 16px;
  }
  .hero p {
    font-size: 16px;
    color: #8b949e;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 auto 32px;
  }
  .hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }
  .hero-tag {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.12);
    color: #c9d1d9;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
  }

  /* ── Agents Grid ── */
  .agents-section {
    padding: 48px 24px 32px;
    max-width: 1200px;
    margin: auto;
  }
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
  }
  .section-title {
    font-size: 20px;
    font-weight: 600;
    color: #e6edf3;
  }
  .section-subtitle {
    font-size: 13px;
    color: #8b949e;
    margin-top: 2px;
  }
  .agents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 16px;
  }
  .agent-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: var(--radius);
    padding: 20px;
    cursor: pointer;
    transition: all .2s ease;
    position: relative;
    overflow: hidden;
  }
  .agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, var(--ibm-blue));
  }
  .agent-card:hover {
    border-color: var(--card-accent, var(--ibm-blue));
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,.3);
  }
  .agent-card.active {
    border-color: var(--card-accent, var(--ibm-blue));
    background: rgba(15,98,254,.08);
  }
  .agent-card:nth-child(1) { --card-accent: #0f62fe; }
  .agent-card:nth-child(2) { --card-accent: #8a3ffc; }
  .agent-card:nth-child(3) { --card-accent: #009d9a; }
  .agent-card:nth-child(4) { --card-accent: #24a148; }
  .agent-card:nth-child(5) { --card-accent: #ff832b; }
  .agent-icon {
    font-size: 28px;
    margin-bottom: 10px;
  }
  .agent-name {
    font-size: 13px;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 6px;
  }
  .agent-desc {
    font-size: 11px;
    color: #8b949e;
    line-height: 1.5;
  }
  .agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(15,98,254,.15);
    color: #79c0ff;
    border: 1px solid rgba(15,98,254,.2);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    margin-top: 8px;
  }

  /* ── Chat Interface ── */
  .chat-section {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px 48px;
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 900px) {
    .chat-section { grid-template-columns: 1fr; }
    .sidebar { order: -1; }
  }
  .chat-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 620px;
  }
  .chat-header {
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .chat-title {
    font-size: 14px;
    font-weight: 600;
    color: #e6edf3;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .chat-model-badge {
    font-size: 11px;
    color: #8b949e;
    background: rgba(255,255,255,.05);
    border: 1px solid #30363d;
    padding: 2px 10px;
    border-radius: 10px;
    font-family: var(--mono);
  }
  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  /* Welcome */
  .welcome-msg {
    background: linear-gradient(135deg, rgba(15,98,254,.1) 0%, rgba(138,63,252,.1) 100%);
    border: 1px solid rgba(15,98,254,.2);
    border-radius: var(--radius);
    padding: 24px;
    text-align: center;
  }
  .welcome-msg h3 { font-size: 16px; font-weight: 600; color: #e6edf3; margin-bottom: 8px; }
  .welcome-msg p  { font-size: 13px; color: #8b949e; line-height: 1.6; margin-bottom: 16px; }
  .quick-starters {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }
  .quick-btn {
    background: rgba(15,98,254,.15);
    border: 1px solid rgba(15,98,254,.3);
    color: #79c0ff;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    cursor: pointer;
    transition: all .15s;
    font-family: var(--font);
  }
  .quick-btn:hover {
    background: rgba(15,98,254,.3);
    border-color: rgba(15,98,254,.5);
  }
  /* Messages */
  .msg { display: flex; gap: 10px; max-width: 88%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.user .bubble {
    background: linear-gradient(135deg, #0f62fe 0%, #0043ce 100%);
    color: white;
    border-radius: var(--radius) var(--radius) 4px var(--radius);
  }
  .msg.assistant .bubble {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
  }
  .bubble {
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 100%;
    word-wrap: break-word;
  }
  .msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg.user .msg-avatar     { background: #0f62fe; }
  .msg.assistant .msg-avatar { background: #21262d; border: 1px solid #30363d; }
  .msg-meta {
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .msg.user .msg-meta { flex-direction: row-reverse; }
  .agent-tag {
    background: rgba(15,98,254,.15);
    color: #79c0ff;
    border: 1px solid rgba(15,98,254,.2);
    padding: 1px 7px;
    border-radius: 8px;
    font-size: 10px;
  }
  /* Risk alert */
  .risk-banner {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    margin-top: 6px;
  }
  .risk-banner.low      { background: rgba(36,161,72,.1);  border: 1px solid rgba(36,161,72,.25);  color: #56d364; }
  .risk-banner.moderate { background: rgba(241,194,27,.1); border: 1px solid rgba(241,194,27,.25); color: #e3b341; }
  .risk-banner.high     { background: rgba(255,131,43,.1); border: 1px solid rgba(255,131,43,.25); color: #ffa657; }
  .risk-banner.critical { background: rgba(218,30,40,.12); border: 1px solid rgba(218,30,40,.35);  color: #f85149; }
  /* Typing indicator */
  .typing { display: flex; align-items: center; gap: 4px; padding: 12px 16px; }
  .typing span {
    width: 6px; height: 6px;
    background: #8b949e;
    border-radius: 50%;
    animation: typing-bounce .8s infinite;
  }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes typing-bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%           { transform: translateY(-6px); }
  }
  /* Input area */
  .chat-input-area {
    border-top: 1px solid #21262d;
    padding: 16px 20px;
    background: #161b22;
  }
  .input-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  .chat-input {
    flex: 1;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: var(--radius-sm);
    color: #e6edf3;
    font-family: var(--font);
    font-size: 14px;
    padding: 10px 14px;
    resize: none;
    outline: none;
    line-height: 1.5;
    max-height: 120px;
    transition: border-color .15s;
  }
  .chat-input:focus { border-color: var(--ibm-blue); }
  .chat-input::placeholder { color: #484f58; }
  .send-btn {
    width: 40px; height: 40px;
    background: var(--ibm-blue);
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all .15s;
    flex-shrink: 0;
  }
  .send-btn:hover   { background: var(--ibm-blue-dark); transform: scale(1.05); }
  .send-btn:active  { transform: scale(.95); }
  .send-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
  .input-hint {
    font-size: 11px;
    color: #484f58;
    margin-top: 8px;
    text-align: center;
  }

  /* ── Sidebar ── */
  .sidebar { display: flex; flex-direction: column; gap: 16px; }
  .sidebar-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: var(--radius);
    overflow: hidden;
  }
  .sidebar-card-header {
    padding: 14px 16px;
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    font-size: 12px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sidebar-card-body { padding: 16px; }
  /* Agent pipeline */
  .pipeline-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 12px;
    color: #8b949e;
    transition: all .15s;
  }
  .pipeline-item.active {
    background: rgba(15,98,254,.1);
    color: #e6edf3;
    border: 1px solid rgba(15,98,254,.2);
  }
  .pipeline-item.done {
    background: rgba(36,161,72,.08);
    color: #56d364;
    border: 1px solid rgba(36,161,72,.15);
  }
  .pipeline-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #30363d;
    flex-shrink: 0;
  }
  .pipeline-item.active .pipeline-dot { background: var(--ibm-blue); animation: pulse-dot 1.5s infinite; }
  .pipeline-item.done  .pipeline-dot  { background: var(--ibm-green); }
  /* Risk gauge */
  .risk-gauge {
    text-align: center;
    padding: 8px 0;
  }
  .gauge-circle {
    width: 90px; height: 90px;
    border-radius: 50%;
    background: conic-gradient(var(--gauge-color, #30363d) var(--gauge-pct, 0%), #21262d 0%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px;
    position: relative;
  }
  .gauge-inner {
    width: 68px; height: 68px;
    background: #161b22;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  .gauge-value { font-size: 20px; font-weight: 700; color: #e6edf3; }
  .gauge-label { font-size: 9px; color: #8b949e; text-transform: uppercase; }
  .gauge-category { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .gauge-note { font-size: 11px; color: #8b949e; }
  /* Resources */
  .resource-item {
    display: flex;
    flex-direction: column;
    padding: 10px 12px;
    border: 1px solid #21262d;
    border-radius: var(--radius-sm);
    margin-bottom: 8px;
    transition: border-color .15s;
  }
  .resource-item:hover { border-color: #30363d; }
  .resource-name    { font-size: 12px; font-weight: 600; color: #e6edf3; margin-bottom: 2px; }
  .resource-contact { font-size: 11px; color: #79c0ff; margin-bottom: 3px; }
  .resource-type    { font-size: 10px; color: #8b949e; background: rgba(255,255,255,.05); padding: 1px 7px; border-radius: 8px; align-self: flex-start; }
  /* Stats */
  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    font-size: 12px;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: #8b949e; }
  .stat-value { color: #e6edf3; font-weight: 500; font-family: var(--mono); }
  .stat-value.blue   { color: #79c0ff; }
  .stat-value.green  { color: #56d364; }
  .stat-value.yellow { color: #e3b341; }
  .stat-value.red    { color: #f85149; }

  /* ── Footer ── */
  .footer {
    background: #010409;
    border-top: 1px solid #21262d;
    padding: 32px 24px;
    text-align: center;
  }
  .footer-grid {
    max-width: 1000px;
    margin: 0 auto 24px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 24px;
    text-align: left;
  }
  .footer-col-title { font-size: 12px; font-weight: 600; color: #e6edf3; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .footer-col p { font-size: 12px; color: #8b949e; line-height: 1.7; }
  .footer-tag {
    display: inline-block;
    background: rgba(255,255,255,.06);
    border: 1px solid #30363d;
    color: #8b949e;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    margin: 3px;
  }
  .footer-copy { font-size: 11px; color: #484f58; border-top: 1px solid #21262d; padding-top: 20px; margin-top: 8px; }
  .footer-copy a { color: #79c0ff; text-decoration: none; }

  /* ── Utility ── */
  .hidden { display: none !important; }
  .shimmer {
    background: linear-gradient(90deg, #21262d 25%, #2d333b 50%, #21262d 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
    height: 14px;
  }
  @keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
  /* Responsive */
  @media (max-width: 600px) {
    .hero { padding: 36px 16px 32px; }
    .agents-section, .chat-section { padding-left: 12px; padding-right: 12px; }
    .chat-container { height: 520px; }
  }
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header class="header">
  <div class="header-brand">
    <div class="ibm-logo">
      <div class="logo-box">IBM</div>
      <span style="color:#8b949e;">watsonx</span><span style="color:#e6edf3;">.ai</span>
    </div>
    <div class="header-divider"></div>
    <div>
      <div class="header-title">🧠 Mental Health Assistant</div>
      <div class="header-subtitle">Powered by IBM Granite Models · Agentic AI</div>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>
    5 Agents Online
  </div>
</header>

<!-- ── HERO ── -->
<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">⚡ IBM Granite · Agentic AI Architecture</div>
    <h1>Your AI-Powered<br/>Mental Health Companion</h1>
    <p>A safe, confidential space for mental health awareness, emotional support,
       and crisis prevention — built on IBM watsonx.ai and IBM Granite Models.</p>
    <div class="hero-tags">
      <span class="hero-tag">🏆 IBM SkillsBuild</span>
      <span class="hero-tag">💡 Hackathon Ready</span>
      <span class="hero-tag">🎓 Academic Project</span>
      <span class="hero-tag">🤖 AI Showcase</span>
      <span class="hero-tag">💙 Suicide Prevention</span>
      <span class="hero-tag">🧬 Granite Models</span>
    </div>
  </div>
</section>

<!-- ── AGENTS GRID ── -->
<section class="agents-section">
  <div class="section-header">
    <div>
      <div class="section-title">⚙️ Specialized AI Agents</div>
      <div class="section-subtitle">Five IBM Granite-powered agents orchestrated intelligently</div>
    </div>
  </div>
  <div class="agents-grid">
    <div class="agent-card" onclick="askAgent(1)">
      <div class="agent-icon">🧠</div>
      <div class="agent-name">Mental Health Awareness</div>
      <div class="agent-desc">Educates on anxiety, depression, stress, burnout, mindfulness & self-care</div>
      <span class="agent-badge">Agent 1</span>
    </div>
    <div class="agent-card" onclick="askAgent(2)">
      <div class="agent-icon">💙</div>
      <div class="agent-name">Emotional Support</div>
      <div class="agent-desc">Empathetic active listening, validation & compassionate companionship</div>
      <span class="agent-badge">Agent 2</span>
    </div>
    <div class="agent-card" onclick="askAgent(3)">
      <div class="agent-icon">🔍</div>
      <div class="agent-name">Distress Detection</div>
      <div class="agent-desc">Real-time crisis signal recognition & emergency response routing</div>
      <span class="agent-badge">Agent 3</span>
    </div>
    <div class="agent-card" onclick="askAgent(4)">
      <div class="agent-icon">🌱</div>
      <div class="agent-name">Coping Strategy</div>
      <div class="agent-desc">Personalised evidence-based coping plans & actionable techniques</div>
      <span class="agent-badge">Agent 4</span>
    </div>
    <div class="agent-card" onclick="askAgent(5)">
      <div class="agent-icon">🏥</div>
      <div class="agent-name">Risk & Resources</div>
      <div class="agent-desc">Mental health risk prediction & professional support connections</div>
      <span class="agent-badge">Agent 5</span>
    </div>
  </div>
</section>

<!-- ── CHAT + SIDEBAR ── -->
<section class="chat-section">
  <!-- Chat Window -->
  <div class="chat-container">
    <div class="chat-header">
      <div class="chat-title">
        💬 AI Assistant
        <span id="active-agent-name" style="font-size:11px;color:#8b949e;font-weight:400;">Orchestrated</span>
      </div>
      <span class="chat-model-badge" id="model-badge">ibm/granite-13b</span>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="welcome-msg">
        <h3>👋 Welcome to your Mental Health Assistant</h3>
        <p>I'm here to listen, support, and guide you. This is a safe, judgment-free space.
           You can ask about mental health topics, share how you're feeling, or explore coping strategies.</p>
        <div class="quick-starters">
          <button class="quick-btn" onclick="sendQuick('What is anxiety and what are its symptoms?')">What is anxiety?</button>
          <button class="quick-btn" onclick="sendQuick('I have been feeling really stressed and overwhelmed lately')">I feel overwhelmed</button>
          <button class="quick-btn" onclick="sendQuick('What are some coping strategies for depression?')">Coping strategies</button>
          <button class="quick-btn" onclick="sendQuick('How can mindfulness help with stress?')">Mindfulness tips</button>
          <button class="quick-btn" onclick="sendQuick('What are the signs of burnout?')">Signs of burnout</button>
          <button class="quick-btn" onclick="sendQuick('I feel lonely and no one understands me')">I feel lonely</button>
        </div>
      </div>
    </div>
    <div class="chat-input-area">
      <div class="input-row">
        <textarea class="chat-input" id="chat-input" placeholder="Share what's on your mind…" rows="1"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="send-btn" onclick="sendMessage()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="input-hint">⚠️ This AI assistant is not a substitute for professional medical advice. In crisis, call 988.</div>
    </div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">
    <!-- Agent Pipeline -->
    <div class="sidebar-card">
      <div class="sidebar-card-header">⚡ Agent Pipeline</div>
      <div class="sidebar-card-body" id="pipeline-container">
        <div class="pipeline-item" id="pipe-1"><div class="pipeline-dot"></div>Awareness Agent</div>
        <div class="pipeline-item" id="pipe-2"><div class="pipeline-dot"></div>Emotional Support Agent</div>
        <div class="pipeline-item" id="pipe-3"><div class="pipeline-dot"></div>Distress Detection Agent</div>
        <div class="pipeline-item" id="pipe-4"><div class="pipeline-dot"></div>Coping Strategy Agent</div>
        <div class="pipeline-item" id="pipe-5"><div class="pipeline-dot"></div>Risk & Resources Agent</div>
      </div>
    </div>

    <!-- Risk Gauge -->
    <div class="sidebar-card">
      <div class="sidebar-card-header">📊 Risk Assessment</div>
      <div class="sidebar-card-body">
        <div class="risk-gauge">
          <div class="gauge-circle" id="gauge-circle" style="--gauge-color:#30363d;--gauge-pct:0%">
            <div class="gauge-inner">
              <span class="gauge-value" id="gauge-value">—</span>
              <span class="gauge-label">Risk</span>
            </div>
          </div>
          <div class="gauge-category" id="gauge-category" style="color:#8b949e">Awaiting input</div>
          <div class="gauge-note" id="gauge-note">Send a message to begin assessment</div>
        </div>
        <div id="alert-flags"></div>
      </div>
    </div>

    <!-- Resources -->
    <div class="sidebar-card" id="resources-card" style="display:none">
      <div class="sidebar-card-header">🏥 Support Resources</div>
      <div class="sidebar-card-body" id="resources-list"></div>
    </div>

    <!-- Session Stats -->
    <div class="sidebar-card">
      <div class="sidebar-card-header">📈 Session Stats</div>
      <div class="sidebar-card-body">
        <div class="stat-row"><span class="stat-label">Messages</span><span class="stat-value blue" id="stat-msgs">0</span></div>
        <div class="stat-row"><span class="stat-label">Agents Invoked</span><span class="stat-value blue" id="stat-agents">0</span></div>
        <div class="stat-row"><span class="stat-label">Peak Risk</span><span class="stat-value" id="stat-peak">Low</span></div>
        <div class="stat-row"><span class="stat-label">Model</span><span class="stat-value" style="font-size:10px;color:#8b949e">Granite 13B</span></div>
      </div>
    </div>
  </div>
</section>

<!-- ── FOOTER ── -->
<footer class="footer">
  <div class="footer-grid">
    <div class="footer-col">
      <div class="footer-col-title">About</div>
      <p>IBM Mental Health Assistant demonstrates Agentic AI Architecture using IBM watsonx.ai and IBM Granite Models for mental health awareness and suicide prevention.</p>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Technology Stack</div>
      <span class="footer-tag">IBM watsonx.ai</span>
      <span class="footer-tag">IBM Granite</span>
      <span class="footer-tag">Agentic AI</span>
      <span class="footer-tag">Python Flask</span>
      <span class="footer-tag">Agent Orchestration</span>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Crisis Contacts</div>
      <p>🆘 <strong style="color:#f85149;">988</strong> — Suicide & Crisis Lifeline (US)<br/>
         📱 Text <strong style="color:#e3b341;">HOME</strong> to 741741 — Crisis Text<br/>
         🌍 <strong>iasp.info</strong> — International</p>
    </div>
    <div class="footer-col">
      <div class="footer-col-title">Disclaimer</div>
      <p>This AI assistant provides educational information only and is not a substitute for professional mental health care. Always consult a licensed professional.</p>
    </div>
  </div>
  <div class="footer-copy">
    Built with ❤️ for <strong>IBM SkillsBuild</strong> · Powered by <a href="https://www.ibm.com/watsonx" target="_blank">IBM watsonx.ai</a> &amp; IBM Granite Models · © 2024
  </div>
</footer>

<script>
/* ── State ── */
const state = {
  messages: [],
  context:  [],
  msgCount: 0,
  agentCount: 0,
  peakTier: 0,
};

/* ── Utilities ── */
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/\\n/g,'<br/>');
}
function now() {
  return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

/* ── Agent card highlight ── */
function askAgent(id) {
  const questions = {
    1: 'What is anxiety and how does it affect mental health?',
    2: 'I have been feeling really overwhelmed and sad lately and need someone to talk to',
    3: 'I am feeling hopeless and like nothing will ever get better',
    4: 'What are some coping strategies I can use when I feel anxious?',
    5: 'I am worried about my mental health and want to know what professional resources are available',
  };
  const input = document.getElementById('chat-input');
  input.value = questions[id] || '';
  autoResize(input);
  input.focus();
  // highlight card
  document.querySelectorAll('.agent-card').forEach((c,i) => {
    c.classList.toggle('active', i === id - 1);
  });
}

/* ── Quick starters ── */
function sendQuick(text) {
  const input = document.getElementById('chat-input');
  input.value = text;
  autoResize(input);
  sendMessage();
}

/* ── Append message ── */
function appendMsg(role, text, meta = {}) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;

  const avatarIcon = role === 'user' ? '👤' : (meta.icon || '🤖');

  let riskHtml = '';
  if (meta.risk_level && role === 'assistant') {
    const rl = meta.risk_level;
    const icons = { low:'✅', moderate:'⚠️', high:'🚨', critical:'🆘' };
    riskHtml = `<div class="risk-banner ${rl}">${icons[rl] || '•'} Risk level: <strong>${rl.toUpperCase()}</strong></div>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">${avatarIcon}</div>
    <div>
      <div class="bubble">${esc(text)}${riskHtml}</div>
      <div class="msg-meta">
        <span>${now()}</span>
        ${meta.agent ? `<span class="agent-tag">${meta.agent}</span>` : ''}
      </div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

/* ── Typing indicator ── */
function showTyping() {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.id = 'typing-indicator';
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="bubble">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}
function hideTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

/* ── Pipeline animation ── */
function animatePipeline(agents) {
  const map = {
    'Awareness':          'pipe-1',
    'Emotional Support':  'pipe-2',
    'Distress Detection': 'pipe-3',
    'Coping Strategy':    'pipe-4',
    'Risk & Resources':   'pipe-5',
  };
  // reset
  Object.values(map).forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
  });
  let delay = 0;
  agents.forEach(agent => {
    const id = map[agent];
    if (!id) return;
    setTimeout(() => {
      document.getElementById(id).classList.add('active');
    }, delay);
    setTimeout(() => {
      const el = document.getElementById(id);
      el.classList.remove('active');
      el.classList.add('done');
    }, delay + 800);
    delay += 400;
  });
}

/* ── Risk gauge update ── */
function updateGauge(riskLevel, tier) {
  const colors = { low:'#56d364', moderate:'#e3b341', high:'#ffa657', critical:'#f85149' };
  const scores  = { low:15, moderate:45, high:70, critical:95 };
  const color   = colors[riskLevel] || '#30363d';
  const pct     = (scores[riskLevel] || 0) / 100 * 360;

  const gauge = document.getElementById('gauge-circle');
  gauge.style.setProperty('--gauge-color', color);
  gauge.style.setProperty('--gauge-pct', `${pct}deg`);

  document.getElementById('gauge-value').textContent    = scores[riskLevel] || '—';
  document.getElementById('gauge-value').style.color    = color;
  document.getElementById('gauge-category').textContent = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1);
  document.getElementById('gauge-category').style.color = color;

  const notes = {
    low:      'No immediate concern detected',
    moderate: 'Monitor and consider support',
    high:     'Encourage professional help soon',
    critical: '⚠️ Immediate intervention needed'
  };
  document.getElementById('gauge-note').textContent = notes[riskLevel] || '';

  // Peak
  if (tier > state.peakTier) {
    state.peakTier = tier;
    const peakEl = document.getElementById('stat-peak');
    peakEl.textContent = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1);
    peakEl.className   = 'stat-value ' + (tier===0?'green':tier===1?'yellow':tier===2?'':' red');
  }
}

/* ── Alert flags ── */
function showAlerts(flags) {
  const el = document.getElementById('alert-flags');
  if (!flags || flags.length === 0) { el.innerHTML = ''; return; }
  el.innerHTML = flags.map(f =>
    `<div style="font-size:12px;color:#f85149;padding:4px 0;border-top:1px solid rgba(248,81,73,.15);margin-top:6px;">${esc(f)}</div>`
  ).join('');
}

/* ── Resources panel ── */
function showResources(resources) {
  if (!resources || resources.length === 0) return;
  const card = document.getElementById('resources-card');
  const list = document.getElementById('resources-list');
  card.style.display = 'block';
  list.innerHTML = resources.map(r => `
    <div class="resource-item">
      <div class="resource-name">${esc(r.name)}</div>
      <div class="resource-contact">${esc(r.contact)}</div>
      <span class="resource-type">${esc(r.type)}</span>
    </div>`).join('');
}

/* ── Send message ── */
async function sendMessage() {
  const input   = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const text    = input.value.trim();
  if (!text) return;

  // Append user message
  appendMsg('user', text);
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  state.msgCount++;
  document.getElementById('stat-msgs').textContent = state.msgCount;

  showTyping();

  try {
    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, context: state.context }),
    });
    const data = await resp.json();
    hideTyping();

    if (data.error) {
      appendMsg('assistant', '⚠️ ' + data.error);
    } else {
      appendMsg('assistant', data.primary_response, {
        icon:       data.primary_icon,
        agent:      data.primary_agent,
        risk_level: data.risk_level,
      });

      // Update sidebar
      animatePipeline(data.agents_activated || []);
      updateGauge(data.risk_level || 'low', data.distress_tier || 0);
      showAlerts(data.alert_flags || []);

      // Show resources if risk elevated
      if (data.show_resources && data.all_results && data.all_results.risk) {
        showResources(data.all_results.risk.resources);
      }

      // Show extra agent responses
      const results = data.all_results || {};
      if (data.risk_level !== 'low' && results.coping) {
        setTimeout(() => appendMsg('assistant', results.coping.response, {
          icon: '🌱', agent: 'Coping Strategy Agent'
        }), 600);
      }
      if ((data.distress_tier >= 2) && results.risk) {
        setTimeout(() => appendMsg('assistant', results.risk.response, {
          icon: '🏥', agent: 'Risk & Resources Agent'
        }), 1200);
      }

      // Update context & stats
      state.context.push({ user: text, agent: data.primary_response });
      if (state.context.length > 10) state.context.shift();

      state.agentCount += (data.agents_activated || []).length;
      document.getElementById('stat-agents').textContent = state.agentCount;
      document.getElementById('active-agent-name').textContent = data.primary_agent || 'Orchestrated';
    }
  } catch (err) {
    hideTyping();
    appendMsg('assistant', '⚠️ Connection error. Please check your watsonx.ai credentials and try again.');
    console.error(err);
  }

  sendBtn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   IBM Mental Health Assistant · Agentic AI · IBM Granite Models            ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Configure your credentials:
    export WATSONX_API_KEY="your-api-key"
    export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
    export WATSONX_PROJECT="your-project-id"

  Then visit:  http://localhost:5000
""")
    app.run(debug=True, host="0.0.0.0", port=5000)
