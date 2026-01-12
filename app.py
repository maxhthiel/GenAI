import streamlit as st
import os
import shutil
import time
import glob
import ast
from dotenv import load_dotenv
from PIL import Image
from agent.agent_builder import build_agent
from main import evaluator_check

# --- 1. SYSTEM CONFIGURATION & UI STYLING ---
# Load environment variables (API keys) and configure the Streamlit page layout
load_dotenv()
st.set_page_config(page_title="Smol-Quant Analyst", page_icon="📊", layout="wide")

# Inject custom CSS to override Streamlit's default styling for a cleaner, professional look
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

# --- UI HEADER & EINFÜHRUNG ---
st.title("Smol-Quant: Autonomer Finanzanalyst")

st.markdown("""
Dieses System wurde im Rahmen eines Abschlussprojekts im Kurs ""Generative AI" entwickelt. Es agiert als autonome Einheit, um die Lücke zwischen quantitativer Datenanalyse und qualitativer Marktrecherche zu schließen.

* Projektziel: Entwicklung eines Agenten, der über einfache Textvorhersagen hinausgeht und aktiv Werkzeuge zur Problemlösung nutzt.
* Methodik: Das System basiert auf dem ReAct-Paradigma (Reasoning and Acting). Der Agent plant logische Schritte, schreibt eigenständig Python-Code und führt diesen aus, um verifizierbare Ergebnisse zu liefern.
* Datenbasis: Die Analyse stützt sich auf einen kuratierten Datensatz des NASDAQ-100 für fundamentale Kennzahlen sowie eine Vektordatenbank (RAG) für die semantische Suche in aktuellen Finanznachrichten.
* Fähigkeiten: Der Agent kann komplexe Markttrends kontextualisieren, Finanzkennzahlen vergleichen und dynamisch Visualisierungen generieren.
            
* Beispiel Anfragen:
    - "Was weißt du über Tesla?"
    - "Vergleiche Nvidia mit Meta."
    - "Analysiere die Volatilität der Top 5 Aktien im letzten Quartal."
""")

st.markdown("---")

# --- 2. SESSION STATE INITIALIZATION ---
# Initialize the autonomous agent only once per session to persist memory and avoid reloading overhead
if "agent" not in st.session_state:
    with st.spinner("🤖 Booting Autonomous Agent..."):
        try:
            st.session_state.agent = build_agent()
        except Exception as e:
            st.error(f"Critical Error building agent: {e}")
            st.stop()

# Initialize the chat history list if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a local reference to the agent for easier access
agent = st.session_state.agent

# --- 3. UTILITY FUNCTIONS (ROBUSTNESS & PARSING) ---

def find_latest_plot():
    """
    Robustly identifies the most recently generated plot file in the working directory.
    
    This solves the issue where agents might name files unpredictably (e.g., 'chart.png' vs 'plot.png').
    It includes a recency check (< 60s) to ensure we don't accidentally display outdated charts.
    """
    # Find all PNG files in the current directory
    list_of_files = glob.glob('*.png') 
    
    # Filter out files that belong to the chat history (archived plots)
    candidate_files = [f for f in list_of_files if not f.startswith("history_plot_")]
    
    if not candidate_files:
        return None
    
    # Identify the file with the most recent creation timestamp
    latest_file = max(candidate_files, key=os.path.getctime)
    
    # Verify that the file was created during the current interaction (within the last 30s)
    if time.time() - os.path.getctime(latest_file) < 30:
        return latest_file
    return None

def save_plot_to_history(source_path):
    """
    Persists the temporary plot file to the session history.
    
    It creates a timestamped copy of the generated image. This ensures that the chart
    remains visible in the chat history even if the agent overwrites the original file
    in subsequent runs.
    """
    if source_path and os.path.exists(source_path):
        timestamp = int(time.time())
        filename = f"history_plot_{timestamp}.png"
        shutil.copy(source_path, filename)
        time.sleep(0.1)  # Brief pause to ensure the OS completes the file write operation
        return filename
    return None

def cleanup_temp_files():
    """
    Cleans up temporary artifacts before a new analysis run.
    This ensures a sterile execution environment and prevents the display of stale data.
    """
    for f in glob.glob("*.png"):
        # We preserve files starting with 'history_' as they are part of the chat log
        if not f.startswith("history_plot_"):
            try:
                os.remove(f)
            except:
                pass

def format_final_answer(text):
    """
    Parses and formats the agent's final response.
    
    Sometimes, the LLM returns a structured dictionary/JSON instead of natural language.
    This function detects such cases and converts the data structure into a readable
    Markdown report for the user.
    """
    try:
        # Check if the text looks like a dictionary structure
        if text.strip().startswith("{") and "Analysis" in text:
            data = ast.literal_eval(text)
            # If parsing succeeds, manually construct a Markdown representation
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
        pass # If parsing fails, fall back to returning the original raw text
    return text

def parse_step_content(step):
    """
    Parses the raw 'ActionStep' object from the agent's memory.
    
    The raw step object contains extensive metadata and system prompts. This function
    extracts only the relevant components (Thoughts, Code, Observations) to display
    a clean 'Chain of Thought' to the user.
    """
    output = []
    
    # 1. Extract the Agent's Thought Process
    # Handling different internal attribute names across smolagents versions
    thought = getattr(step, 'thought', None) 
    if not thought and hasattr(step, 'model_output'):
        thought = step.model_output
    
    if thought:
        output.append(f"**🤔 Thought:**\n{thought.strip()}")

    # 2. Extract Executed Tool Code
    if hasattr(step, 'tool_calls') and step.tool_calls:
        for call in step.tool_calls:
            # Code might be in 'arguments' or directly in the call object
            code = ""
            if hasattr(call, 'arguments'):
                code = call.arguments
            else:
                code = str(call)
            
            # Display only if valid code is found
            if code:
                output.append(f"**🛠️ Tool Code:**\n```python\n{code}\n```")

    # 3. Extract the Result/Observation from the Tool
    if hasattr(step, 'observations') and step.observations:
        obs = str(step.observations)
        # Truncate extremely long outputs (e.g., entire DataFrames) to keep UI clean
        if len(obs) > 500:
            obs = obs[:500] + "... [truncated]"
        output.append(f"**👀 Observation:**\n_{obs}_")

    return "\n\n".join(output)

def get_agent_steps(agent_obj):
    """
    Retrieves the execution history safely.
    Acts as a compatibility layer for different versions of the 'smolagents' library,
    checking both 'memory.steps' and 'logs'.
    """
    if hasattr(agent_obj, "memory") and hasattr(agent_obj.memory, "steps"):
        return agent_obj.memory.steps
    elif hasattr(agent_obj, "logs"):
        return agent_obj.logs
    return []

# --- 4. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    if st.button("🗑️ Reset All", use_container_width=True):
        # Hard reset: Delete the agent instance to force a fresh rebuild
        del st.session_state.agent
        st.session_state.messages = []
        # Clean up historical plots
        for f in glob.glob("history_plot_*.png"):
            os.remove(f)
        cleanup_temp_files()
        st.rerun()

# --- 5. CHAT INTERFACE RENDERING ---
# Iterate through session history to render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Chain-of-Thought logs if available
        if "steps_display" in msg and msg["steps_display"]:
            with st.expander("🧠 View Thought Process"):
                st.markdown(msg["steps_display"])

        # Display associated charts if they exist on disk
        if "plot_path" in msg and os.path.exists(msg["plot_path"]):
            try:
                # Load image into memory to avoid file lock issues
                img = Image.open(msg["plot_path"])
                st.image(img, caption="Analysis Chart")
            except:
                st.error("Image file missing.")

# --- 6. MAIN INTERACTION LOOP ---
if prompt := st.chat_input("Ask about stocks..."):
    
    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process the request with the Agent
    with st.chat_message("assistant"):
        container = st.empty()
        status = st.status("🔍 Analyst is working...", expanded=True)
        
        # Prepare environment
        cleanup_temp_files()
        
        # Capture the state of steps BEFORE execution to calculate the delta later
        all_steps_before = get_agent_steps(agent)
        start_step_count = len(all_steps_before)
        
        try:
            status.write("🧠 Planning & Coding...")
            # Execute the agent. 'reset=False' is critical to maintain conversational context (memory) across turns.
            response = agent.run(prompt, reset=False) 
            
            # Apply formatting to handle cases where the LLM returns structured data (JSON/Dict)
            response_text = format_final_answer(str(response))
            
            # Optional: Compliance Check Layer (Evaluates if the response contains financial advice)
            compliance = evaluator_check(str(response), prompt)
            if not compliance["passed"]:
                status.write(f"⚠️ Adjustment: {compliance['feedback']}")
                correction = f"Rewrite strictly: {compliance['feedback']}"
                # Recursive call for self-correction
                response = agent.run(correction, reset=False)
                response_text = format_final_answer(str(response))

            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"Error: {e}")
            st.stop()

        # --- POST-PROCESSING & ARTIFACT HANDLING ---
        
        # 1. Process Execution Logs: Extract and format the Chain-of-Thought (CoT) steps for transparency
        all_steps_now = get_agent_steps(agent)
        raw_new_steps = all_steps_now[start_step_count:]
        
        formatted_steps_text = ""
        for step in raw_new_steps:
            formatted_steps_text += parse_step_content(step) + "\n\n---\n\n"

        # 2. Artifact Discovery: Automatically detect and secure any visual outputs generated during the run
        found_plot = find_latest_plot()
        saved_plot_path = save_plot_to_history(found_plot)
        
        # 3. UI Update: Render the final response and any discovered artifacts
        container.markdown(response_text)
        
        with st.expander("🧠 View Live Thought Process"):
             st.markdown(formatted_steps_text)

        if saved_plot_path:
            try:
                img = Image.open(saved_plot_path)
                st.image(img, caption="Generated Data Chart")
            except:
                pass

        # 4. Session Persistence: Append the full interaction data to the chat history
        msg_entry = {
            "role": "assistant", 
            "content": response_text,
            "steps_display": formatted_steps_text # Persist the formatted logs
        }
        if saved_plot_path:
            msg_entry["plot_path"] = saved_plot_path
            
        st.session_state.messages.append(msg_entry)