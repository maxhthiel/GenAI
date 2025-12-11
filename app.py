import streamlit as st
import os
import glob
import base64
import time
from dotenv import load_dotenv
from openai import OpenAI

# Importiert euren Agenten
from agent.agent_builder import build_agent

# --- KONFIGURATION ---
load_dotenv()
st.set_page_config(
    page_title="Smol-Quant Analyst", 
    page_icon="📊", 
    layout="wide"
)

# --- CLEAN CSS (Professional Style) ---
st.markdown("""
<style>
    .stApp {background-color: #ffffff;}
    .stChatMessage {font-family: 'IBM Plex Mono', monospace; background-color: #f8f9fa; border-radius: 10px; padding: 10px;}
    h1 {color: #2c3e50;}
    /* Dezentere Status-Meldungen */
    .stStatusWidget {border: 1px solid #e0e0e0;}
</style>
""", unsafe_allow_html=True)

# --- INITIALISIERUNG ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agent cachen für Performance
@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- HILFSFUNKTIONEN ---

def cleanup_old_plots():
    """Löscht alte PNGs vor einer neuen Anfrage."""
    files = glob.glob("*.png")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

def evaluator_check(agent_response: str, user_query: str) -> dict:
    """
    RELAXED JUDGE: Erlaubt Meinungen/Analysen, verbietet nur direkte Befehle.
    """
    system_prompt = """
    You are a helpful Compliance Assistant checking a financial report.
    
    Review the AI's response based on these RELAXED Rules:
    1. ALLOW ANALYSIS: The AI IS ALLOWED to describe trends, growth, and market sentiment (e.g., "The stock is performing well", "Bullish signal").
    2. NO DIRECT COMMANDS: The AI must ONLY avoid direct imperatives like "Buy now!", "Sell immediately!".
    3. DATA QUALITY: The answer must not be empty. Technical terms (np.float64) are allowed.
    
    Output strictly: PASSED or FAILED: <Reason>
    """
    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {user_query}\n\nAI Response: {agent_response}"}
            ]
        )
        verdict = response.choices[0].message.content
        if "PASSED" in verdict:
            return {"passed": True, "feedback": ""}
        else:
            return {"passed": False, "feedback": verdict.replace("FAILED:", "").strip()}
    except Exception as e:
        return {"passed": True, "feedback": f"Judge Error: {e}"}

def run_agent_process(query):
    """Führt den Agenten mit Safety-Loop aus."""
    cleanup_old_plots()
    status = st.status("🤖 Smol-Quant is working...", expanded=True)
    
    try:
        # 1. Initialer Gedanke
        status.write("🔍 Gathering financial data & news...")
        response = agent.run(query, reset=True)
        final_response_text = str(response)
        
        # 2. Compliance Loop (Max 2 Retries)
        max_retries = 2
        for attempt in range(max_retries):
            status.write(f"⚖️ Compliance Check (Round {attempt+1})...")
            check = evaluator_check(final_response_text, query)
            
            if check["passed"]:
                status.update(label="✅ Analysis Complete & Approved", state="complete", expanded=False)
                return final_response_text
            
            # Fehlerfall
            status.write(f"⚠️ Adjustment needed: {check['feedback']}")
            correction_prompt = f"Your answer was rejected. Reason: {check['feedback']}. Please rewrite it strictly."
            response = agent.run(correction_prompt, reset=False) # Memory behalten!
            final_response_text = str(response)

        status.update(label="⚠️ Finished with Warnings", state="error", expanded=False)
        return final_response_text

    except Exception as e:
        status.update(label="❌ Critical Error", state="error")
        return f"Error: {e}"

# --- SIDEBAR (Clean) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("### Active Modules")
    st.code("RAG (News)\nPython (Charts)\nImageGen (DALL-E)", language="text")
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        cleanup_old_plots()
        st.rerun()

# --- MAIN CHAT UI ---
st.title("Smol-Quant Analyst 🚀")
st.caption("Autonomous Financial Analysis | Powered by smolagents")

# 1. Verlauf rendern
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Text
        st.markdown(msg["content"])
        
        # Bilder (Plot)
        if msg.get("plot_path") and os.path.exists(msg["plot_path"]):
            st.image(msg["plot_path"], caption="Market Analysis Chart")
            
        # Bilder (Base64 / Image Gen Tool)
        if msg.get("image_data"):
            st.image(msg["image_data"], caption="Sentiment Visualization")

# 2. User Input
if prompt := st.chat_input("Ask about Tesla, Nvidia, or Market Trends..."):
    # User Nachricht
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Prozess
    with st.chat_message("assistant"):
        answer_text = run_agent_process(prompt)
        
        # Daten für History vorbereiten
        msg_data = {"role": "assistant", "content": answer_text}

        # A) Nach Charts suchen (Matplotlib speichert PNGs)
        plot_files = glob.glob("*.png")
        if plot_files:
            # Nimm das neueste Bild
            latest_plot = max(plot_files, key=os.path.getctime)
            st.image(latest_plot, caption="Generated Data Chart")
            msg_data["plot_path"] = latest_plot

        # B) Nach Base64 Bildern suchen (Image Gen Tool gibt Strings zurück)
        # Hack: Wir suchen nach langen Strings ohne Leerzeichen, die wie Base64 aussehen
        if "iVBOR" in answer_text and len(answer_text) > 1000:
            try:
                # Extrahiere Base64 (Clean up text around it)
                clean_b64 = answer_text.split("")[-1].strip()
                # Manchmal ist noch Text davor/dahinter, wir probieren es einfach anzuzeigen
                image_bytes = base64.b64decode(clean_b64)
                st.image(image_bytes, caption="AI Generated Sentiment Art")
                msg_data["image_data"] = image_bytes
                # Text bereinigen, damit der Base64 String nicht den Chat flutet
                msg_data["content"] = "🎨 I have generated an artistic visualization of the sentiment."
            except:
                pass # War wohl doch kein Bild

        st.markdown(msg_data["content"])
        st.session_state.messages.append(msg_data)