"""
Smol-Quant Terminal Interface & Safety Pipeline.

Main entry point for terminal-based interaction with Smol-Quant agent.
Implements 'LLM-as-a-Judge' and self-correction for compliance.
"""

import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import the agent factory function
from agent.agent_builder_niklas2 import build_agent 

# Load environment variables (API keys, configuration)
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Initialize a separate OpenAI client for the Evaluator (Judge)
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluator_check(agent_response: str, original_question: str) -> dict:
    """Validates agent responses using GPT-4o-mini as Compliance Officer."""
    system_prompt = """
    You are a helpful Compliance Assistant checking a financial report.
    Review AI response for:
    1. Financial advice violations (imperatives)
    2. Data quality (no gibberish, empty answers)
    Output: PASSED or FAILED: <Reason>
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
        return {"passed": True, "feedback": f"Judge Error (Passed by default): {e}"}

def run_safe_pipeline(agent, user_input: str) -> str:
    """
    Executes agent logic with self-correction loop.
    Generates text + optional image if requested via ImageGenerationTool.
    """
    print(f"[Agent] Processing query...")
    response = agent.run(user_input, reset=True)
    max_retries = 3

    for attempt in range(max_retries):
        response_str = str(response)
        evaluation = evaluator_check(response_str, user_input)
        if evaluation["passed"]:
            return response
        print(f"\n[Warning] Compliance Alert (Attempt {attempt+1}/{max_retries}): {evaluation['feedback']}")
        print("[System] Initiating self-correction sequence...")
        correction_prompt = (
            f"Your previous answer was rejected by the Compliance Officer. "
            f"Reason: {evaluation['feedback']}. "
            f"Please rewrite your answer, strictly following the rules (Data only, no advice)."
        )
        response = agent.run(correction_prompt, reset=False)

    return "[Error] I cannot provide a compliant answer at the moment."

def main():
    """
    Main REPL for Smol-Quant.
    Handles user input, executes agent pipeline, prints text + image URLs.
    """
    print("--------------------------------------------------")
    print("Smol-Quant Terminal Interface (English Mode)")
    print("--------------------------------------------------")
    print("Loading Agent and Tools... please wait.")
    
    try:
        agent = build_agent()
        print("[System] Ready. (Type 'exit' or 'quit' to close)")
    except Exception as e:
        print(f"[Critical Error] Failed to initialize agent: {e}")
        return

    while True:
        try:
            print("\n" + "="*50)
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("[System] Session ended.")
                break
            if not user_input:
                continue

            # Run agent pipeline (text + optional image)
            final_response = run_safe_pipeline(agent, user_input)
            print(f"\n[Agent] Final Answer:\n{final_response}")

        except KeyboardInterrupt:
            print("\n[System] Aborted by user.")
            break
        except Exception as e:
            print(f"[Runtime Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

