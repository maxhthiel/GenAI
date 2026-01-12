import streamlit as st
import os
import shutil
import time
import glob
import ast
from dotenv import load_dotenv
from PIL import Image

# Deine Module
from agent.agent_builder import build_agent
from main import evaluator_check

# --- 1. KONFIGURATION & CSS ---
load_dotenv()
st.set_page_config(page_title="Smol-Quant Analyst", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #ffffff;}
    .stChatMessage {
        font-family: 'IBM Plex Mono', monospace; 
        background-color: #f0f2f6; 
        border-radius: 10px; 
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    h1 {color: #1f2937;}
    .stStatusWidget {border: 1px solid #dfe6e9;}
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if "agent" not in st.session_state:
    with st.spinner("🤖 Booting Autonomous Agent..."):
        try:
            st.session_state.agent = build_agent()
        except Exception as e:
            st.error(f"Critical Error building agent: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

# --- 3. HELPER FUNCTIONS (DIE FIXES) ---

def find_latest_plot():
    """
    FIX FÜR DAS PLOT PROBLEM:
    Sucht die NEUESTE .png Datei im Ordner, egal wie sie heißt.
    """
    # Alle PNGs finden
    list_of_files = glob.glob('*.png') 
    if not list_of_files:
        return None
    
    # Die neueste Datei finden
    latest_file = max(list_of_files, key=os.path.getctime)
    
    # Prüfen, ob die Datei 'frisch' ist (letzte 30 Sekunden)
    # Damit wir keine alten Charts von gestern laden
    if time.time() - os.path.getctime(latest_file) < 30:
        return latest_file
    return None

def save_plot_to_history(source_path):
    """Sichert den gefundenen Plot."""
    if source_path and os.path.exists(source_path):
        timestamp = int(time.time())
        filename = f"history_plot_{timestamp}.png"
        shutil.copy(source_path, filename)
        time.sleep(0.1) 
        return filename
    return None

def cleanup_temp_files():
    """Löscht alle nicht-historischen PNGs, um sauber zu starten."""
    for f in glob.glob("*.png"):
        if not f.startswith("history_plot_"):
            try:
                os.remove(f)
            except:
                pass

def format_final_answer(text):
    """
    FIX FÜR DIE SCHLECHTE ANTWORT:
    Versucht, Dictionary-Strings in schöne Tabellen zu wandeln.
    """
    try:
        # Prüfen ob es wie ein Dict aussieht: {'Key': ...}
        if text.strip().startswith("{") and "Analysis" in text:
            data = ast.literal_eval(text)
            # Wenn erfolgreich geparst, bauen wir Markdown manuell
            md = "### 📊 Structured Analysis\n"
            if 'Analysis' in data:
                for company, metrics in data['Analysis'].items():
                    if isinstance(metrics, dict):
                        md += f"**{company}**\n"
                        for k, v in metrics.items():
                            md += f"- {k}: {v}\n"
                        md += "\n"
            if 'Observations' in data:
                 md += "\n**📝 Observations:**\n"
                 for k, v in data['Observations'].items():
                     md += f"- **{k}**: {v}\n"
            return md
    except:
        pass # Falls parsing schief geht, einfach original Text zurückgeben
    return text

def parse_step_content(step):
    """
    FIX FÜR DEN TEXT BLOB:
    Extrahiert sauber Thought, Code und Observation aus dem Step-Objekt.
    """
    output = []
    
    # 1. Gedanken (Thought)
    # Verschiedene Versionen von smolagents speichern das an verschiedenen Orten
    thought = getattr(step, 'thought', None) 
    if not thought and hasattr(step, 'model_output'):
        thought = step.model_output
    
    if thought:
        output.append(f"**🤔 Thought:**\n{thought.strip()}")

    # 2. Code (Tool Calls)
    if hasattr(step, 'tool_calls') and step.tool_calls:
        for call in step.tool_calls:
            # Code steckt oft in 'arguments' oder direkt im Call-Objekt
            code = ""
            if hasattr(call, 'arguments'):
                code = call.arguments
            else:
                code = str(call)
            
            # Nur anzeigen wenn es Code ist (Python)
            if code:
                output.append(f"**🛠️ Tool Code:**\n```python\n{code}\n```")

    # 3. Ergebnis (Observation)
    if hasattr(step, 'observations') and step.observations:
        obs = str(step.observations)
        # Wenn die Observation riesig ist (z.B. ganzer CSV Inhalt), kürzen
        if len(obs) > 500:
            obs = obs[:500] + "... [truncated]"
        output.append(f"**👀 Observation:**\n_{obs}_")

    return "\n\n".join(output)

def get_agent_steps(agent_obj):
    if hasattr(agent_obj, "memory") and hasattr(agent_obj.memory, "steps"):
        return agent_obj.memory.steps
    elif hasattr(agent_obj, "logs"):
        return agent_obj.logs
    return []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info("**Active Modules:**\n- CodeAgent\n- RAG\n- Matplotlib")
    if st.button("🗑️ Reset All", use_container_width=True):
        del st.session_state.agent
        st.session_state.messages = []
        # Lösche auch alle History Bilder
        for f in glob.glob("history_plot_*.png"):
            os.remove(f)
        cleanup_temp_files()
        st.rerun()

# --- 5. CHAT UI ---
st.title("Smol-Quant Analyst 🚀")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Sauberer CoT Display
        if "steps_display" in msg and msg["steps_display"]:
            with st.expander("🧠 View Thought Process"):
                st.markdown(msg["steps_display"])

        # Bild Display
        if "plot_path" in msg and os.path.exists(msg["plot_path"]):
            try:
                img = Image.open(msg["plot_path"])
                st.image(img, caption="Analysis Chart")
            except:
                st.error("Image file missing.")

# --- 6. MAIN LOOP ---
if prompt := st.chat_input("Ask about stocks..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        status = st.status("🔍 Analyst is working...", expanded=True)
        
        cleanup_temp_files()
        
        # Step Zähler
        all_steps_before = get_agent_steps(agent)
        start_step_count = len(all_steps_before)
        
        try:
            status.write("🧠 Planning & Coding...")
            response = agent.run(prompt, reset=False) 
            
            # FIX: Formatierung anwenden falls Dictionary
            response_text = format_final_answer(str(response))
            
            # Compliance Check
            compliance = evaluator_check(str(response), prompt)
            if not compliance["passed"]:
                status.write(f"⚠️ Adjustment: {compliance['feedback']}")
                correction = f"Rewrite strictly: {compliance['feedback']}"
                response = agent.run(correction, reset=False)
                response_text = format_final_answer(str(response))

            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"Error: {e}")
            st.stop()

        # --- DATEN VERARBEITUNG ---
        
        # 1. Neue Steps holen & formatieren (Text-Blob Killer)
        all_steps_now = get_agent_steps(agent)
        raw_new_steps = all_steps_now[start_step_count:]
        
        formatted_steps_text = ""
        for step in raw_new_steps:
            formatted_steps_text += parse_step_content(step) + "\n\n---\n\n"

        # 2. Smart Plot Suche (Egal wie der Agent das Bild nennt)
        found_plot = find_latest_plot()
        saved_plot_path = save_plot_to_history(found_plot)
        
        # 3. Anzeige
        container.markdown(response_text)
        
        with st.expander("🧠 View Live Thought Process"):
             st.markdown(formatted_steps_text)

        if saved_plot_path:
            try:
                img = Image.open(saved_plot_path)
                st.image(img, caption="Generated Data Chart")
            except:
                pass

        # 4. Speichern
        msg_entry = {
            "role": "assistant", 
            "content": response_text,
            "steps_display": formatted_steps_text # Wir speichern den formatierten Text!
        }
        if saved_plot_path:
            msg_entry["plot_path"] = saved_plot_path
            
        st.session_state.messages.append(msg_entry)