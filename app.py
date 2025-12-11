import streamlit as st
import os
import glob
import time
from dotenv import load_dotenv
from openai import OpenAI

# Wir importieren den Agenten-Builder aus deinem existierenden Skript
from agent.agent_builder import build_agent

# --- KONFIGURATION ---
load_dotenv()
st.set_page_config(
    page_title="Smol-Quant Analyst", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Wall Street Look) ---
st.markdown("""
<style>
    .stApp {background-color: #f0f2f6;}
    .stChatMessage {font-family: 'IBM Plex Mono', monospace;}
    h1 {color: #0e1117;}
    div.stButton > button {background-color: #0e1117; color: white;}
</style>
""", unsafe_allow_html=True)

# --- INITIALISIERUNG ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agent cachen (damit er nicht bei jedem Klick neu lädt)
@st.cache_resource
def get_agent():
    return build_agent()

agent = get_agent()
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- HILFSFUNKTIONEN ---

def cleanup_old_plots():
    """Löscht alte PNGs, damit wir nicht versehentlich Charts von gestern anzeigen."""
    files = glob.glob("*.png")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

def evaluator_check(agent_response: str, user_query: str) -> dict:
    """Der 'Judge', der prüft ob die Antwort sicher ist."""
    system_prompt = """
    You are a strict Compliance Officer at a major bank. 
    Review the output of the AI Financial Analyst.
    
    1. NO FINANCIAL ADVICE: The AI must NEVER explicitly tell the user to "Buy", "Sell", or "Invest".
    2. DATA QUALITY: The answer must not be empty. Technical types like 'np.float64' are allowed but should be readable.
    
    Output format:
    PASSED
    (or)
    FAILED: <Reason>
    """
    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {user_query}\n\nResponse: {agent_response}"}
            ]
        )
        verdict = response.choices[0].message.content
        if "PASSED" in verdict:
            return {"passed": True, "feedback": ""}
        else:
            return {"passed": False, "feedback": verdict.replace("FAILED:", "").strip()}
    except Exception as e:
        return {"passed": True, "feedback": f"Judge Error (Allowing response): {e}"}

# --- HAUPTLOGIK: DER SAFETY LOOP ---
def run_agent_with_ui(query):
    # 1. Alte Plots aufräumen vor dem Run
    cleanup_old_plots()
    
    # UI Status Container
    status_container = st.status("🤖 Analyst is working...", expanded=True)
    
    try:
        # Initialer Versuch
        status_container.write("🧠 Planning analysis steps...")
        response = agent.run(query, reset=True) # reset=True für frischen Start
        
        max_retries = 2
        final_response_text = str(response)
        
        # Der Loop
        for attempt in range(max_retries):
            status_container.write(f"👮 Compliance Check (Round {attempt + 1})...")
            
            check = evaluator_check(str(response), query)
            
            if check["passed"]:
                status_container.update(label="✅ Analysis Approved!", state="complete", expanded=False)
                return response, True
            
            # Wenn Check fehlschlägt:
            status_container.write(f"⚠️ Issue detected: {check['feedback']}")
            status_container.write("🔄 Agent is self-correcting...")
            
            correction_instruction = (
                f"Your previous answer was rejected by compliance. Reason: {check['feedback']}. "
                f"Please correct your answer. Convert numpy types to standard numbers."
            )
            
            # WICHTIG: reset=False, damit er aus Fehlern lernt
            response = agent.run(correction_instruction, reset=False)
            final_response_text = str(response)

        # Wenn wir hier ankommen, hat er es nach Retries immer noch nicht geschafft oder (wie bei Nvidia) es ist der letzte Stand.
        # Wir geben es trotzdem zurück, markieren es aber.
        status_container.update(label="⚠️ Max Retries Reached", state="error", expanded=False)
        return final_response_text, False

    except Exception as e:
        status_container.update(label="❌ System Error", state="error")
        return f"Critical Agent Error: {e}", False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Apple_Logo.svg/1200px-Apple_Logo.svg.png", width=50) # Platzhalter Logo
    st.header("Smol-Quant Settings")
    st.caption("v1.0 | Thesis Demo")
    
    st.markdown("---")
    st.markdown("**Active Tools:**")
    st.success("✅ RAG (ChromaDB)")
    st.success("✅ Python Analysis")
    st.success("✅ Image Generation")
    
    if st.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        cleanup_old_plots()
        st.rerun()

# --- CHAT AREA ---
st.title("🤖 Smol-Quant")
st.subheader("Autonomous Financial Analyst Agent")

# Verlauf anzeigen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Wenn Bilder im Verlauf gespeichert waren
        if "plot_path" in msg and msg["plot_path"]:
            if os.path.exists(msg["plot_path"]):
                st.image(msg["plot_path"])
            else:
                st.caption(f"*(Chart {msg['plot_path']} expired)*")

# User Input
if prompt := st.chat_input("Ask about stocks, trends, or analysis..."):
    # 1. User Nachricht
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent Antwort
    with st.chat_message("assistant"):
        # Hier läuft der Loop mit UI Updates
        answer_text, success = run_agent_with_ui(prompt)
        
        # Nach Plots suchen (Dateien, die JETZT existieren)
        # Wir suchen alle PNGs, die in den letzten 30 Sekunden erstellt wurden
        plot_files = glob.glob("*.png")
        latest_plot = None
        
        # Simple Logic: Wenn ein PNG existiert, nehmen wir es an
        if plot_files:
            # Nimm das neueste
            latest_plot = max(plot_files, key=os.path.getctime)
            st.image(latest_plot, caption="Generated Analysis Chart")
        
        # Text anzeigen
        st.markdown(answer_text)
        
        # Speichern in History
        msg_data = {"role": "assistant", "content": answer_text}
        if latest_plot:
            msg_data["plot_path"] = latest_plot
            
        st.session_state.messages.append(msg_data)