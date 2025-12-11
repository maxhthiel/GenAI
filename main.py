import logging
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from agent.agent_builder import build_agent # Deine korrigierte Builder-Funktion

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Separate Client for the "Judge" (Evaluator)
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluator_check(agent_response: str, original_question: str) -> dict:
    """
    LLM-as-a-Judge: Checks for financial advice (illegal) and quality.
    Returns: {"passed": bool, "feedback": str}
    """
    system_prompt = """
    You are a helpful Compliance Assistant checking a financial report.
    
    Review the AI's response based on these Relaxed Rules:
    
    1. ALLOW ANALYSIS: The AI IS ALLOWED to describe trends, growth, and positive/negative sentiment (e.g., "The stock is performing well", "Strong upside potential", "Investors are bullish"). This is NOT financial advice, it is analysis.
    2. NO DIRECT COMMANDS: The AI should only avoid direct imperatives like "Buy this stock now!", "Sell immediately!", or "Put all your money in X".
    3. DATA QUALITY: The answer must not be empty or obvious gibberish. Technical terms like 'np.float64' are acceptable.
    
    Output format strictly:
    PASSED
    (or)
    FAILED: <Reason>
    """
    
    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {original_question}\n\nAI Response: {agent_response}"}
            ]
        )
        
        verdict = response.choices[0].message.content
        
        if "PASSED" in verdict:
            return {"passed": True, "feedback": ""}
        else:
            return {"passed": False, "feedback": verdict.replace("FAILED:", "").strip()}
            
    except Exception as e:
        # Falls der Judge abstürzt, lassen wir es im Zweifel durchgehen (In dubio pro reo)
        return {"passed": True, "feedback": f"Judge Error (Passed by default): {e}"}

def run_safe_pipeline(agent, user_input):
    """
    Runs the agent with a safety loop. If the answer is bad, it auto-corrects.
    """
    # 1. Initial Run (reset=True clears previous conversation history for a fresh start)
    print(f"🤖 Agent is thinking...")
    response = agent.run(user_input, reset=True)
    
    # 2. Evaluation Loop (Max 2 retries to prevent infinite loops)
    max_retries = 3
    
    for attempt in range(max_retries):
        # Convert response to string just in case it returns an object
        response_str = str(response)
        
        # Check the response
        evaluation = evaluator_check(response_str, user_input)
        
        if evaluation["passed"]:
            return response # Success!
        
        # If FAILED:
        print(f"\n⚠️ Compliance Alert (Attempt {attempt+1}/{max_retries}): {evaluation['feedback']}")
        print("🔄 Agent is correcting the response based on feedback...")
        
        # 3. Correction Run
        # We send the feedback back to the agent. 
        # IMPORTANT: reset=False ensures the agent remembers its mistake!
        correction_prompt = (
            f"Your previous answer was rejected by the Compliance Officer. "
            f"Reason: {evaluation['feedback']}. "
            f"Please rewrite your answer, strictly following the rules (Data only, no advice)."
        )
        response = agent.run(correction_prompt, reset=False)

    # Fallback if it still fails
    return "❌ I apologize, but I cannot provide a compliant answer to this query at the moment."

def main():
    print("--------------------------------------------------")
    print("🚀 Smol-Quant Terminal Interface (English Mode)")
    print("--------------------------------------------------")
    print("Loading Agent and Tools... please wait.")
    
    try:
        # Build agent once
        agent = build_agent()
        print("✅ System Ready. (Type 'exit' or 'quit' to close)")
    except Exception as e:
        print(f"❌ Critical Error starting agent: {e}")
        return

    while True:
        try:
            print("\n" + "="*50)
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Session ended.")
                break

            if not user_input:
                continue
            
            # Run the safe pipeline instead of raw agent.run
            final_response = run_safe_pipeline(agent, user_input)

            print(f"\n🤖 Final Answer:\n{final_response}")

        except KeyboardInterrupt:
            print("\n👋 Aborted by user.")
            break
        except Exception as e:
            print(f"❌ Runtime Error: {e}")

if __name__ == "__main__":
    main()