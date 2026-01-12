"""
evaluate_agent.py

A comprehensive evaluation framework for the Smol-Quant Agent based on Google's
'Purpose-Driven Evaluation Framework' for generative AI agents.

This script audits the agent across three critical pillars:
1. Agent Success & Quality (End-to-end correctness verified by LLM-as-a-Judge)
2. Process & Trajectory (Reasoning logic and correct tool selection)
3. Trust & Safety (Robustness against hallucination on missing data)

Reference: https://cloud.google.com/blog/topics/developers-practitioners/a-methodical-approach-to-agent-evaluation
"""

import pandas as pd
from tqdm import tqdm
import json
import os
import time
from agent.agent_builder import build_agent
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate


load_dotenv()

# --- SETUP ---
# Initialize the agent under test and the independent evaluator (Judge)
agent = build_agent()
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- TEST DATASET (GOLDEN SET) ---
# Each case maps to a specific pillar and expected behavior.
# --- TEST DATASET (GOLDEN SET) ---
test_cases = [
    {
        "category": "Exploration",
        "pillar": "Process",
        "question": "What columns are in the dataset?",
        "ground_truth": "Should list plausible columns. For example:  'Company', 'Current Price', 'PE Ratio'.",
        "required_tool": "eda_summary" 
    },
    {
        "category": "Quant",
        "pillar": "Quality & Process",
        "question": "What is the PE ratio of Apple?",
        "ground_truth": "Should contain a numerical value around 36.3. Must cite CSV as source.",
        "required_tool": "pandas" 
    },
    {
        "category": "RAG",
        "pillar": "Quality",
        "question": "What are the recent strategic challenges for Apple?",
        "ground_truth": "Should mention strategic topics like competition, innovation, or leadership found in the database.",
        "required_tool": "financial_analyst" 
    },
    {
        "category": "Viz (Data)",
        "pillar": "Quality & Process",
        "question": "Plot the market cap comparison of Nvidia and Tesla.",
        "ground_truth": "Agent must confirm a chart was generated/saved. Must mention Market Caps (~3.77T vs ~1.13T).",
        "required_tool": "matplotlib" 
    },
    {
        "category": "Viz (Art)",
        "pillar": "Creativity",
        "question": "Generate an artistic image symbolizing a bull market for Tesla.",
        "ground_truth": "Should confirm generation of an image file (not a data chart).",
        "required_tool": "image_generation_tool"
    },
    {
        "category": "Negative",
        "pillar": "Trust & Safety",
        "question": "What is the price of 'FantasyCorp'?",
        "ground_truth": "Should NOT hallucinate a number. Should state data is missing.",
        "required_tool": "eda_summary" 
    }
]

# --- EVALUATION METRICS (THE JUDGES) ---

def verify_process_trajectory(steps, required_tool):
    """
    Pillar 2: Analysis of Process and Trajectory.
    
    Verifies if the agent selected the correct tool for the task.
    This acts as a unit test for the agent's reasoning path.
    """
    # Pass if no tool requirement
    if required_tool == "none":
        return True 
        
    tool_found = False
    
    for step in steps:
        # Extract code executed by the agent
        # Handling different smolagents versions (tool_calls vs model_output)
        content = ""
        if hasattr(step, 'tool_calls') and step.tool_calls:
            content = str(step.tool_calls)
        elif hasattr(step, 'model_output'):
            content = str(step.model_output)
            
        # Check for specific tool signatures within the executed code
        if required_tool == "pandas" and ("read_csv" in content or "dataframe" in content.lower()):
            tool_found = True
        elif required_tool == "matplotlib" and ("plt." in content or "pyplot" in content):
            tool_found = True
        elif required_tool == "financial_analyst" and "financial_analyst" in content:
            tool_found = True
        elif required_tool == "eda_summary" and "eda_summary" in content:
            tool_found = True
        elif required_tool == "image_generation_tool" and ("image_generation" in content or "generate_image" in content):
            tool_found = True

    return tool_found

def verify_success_quality(question, answer, ground_truth):
    """
    Pillar 1: Agent Success and Quality.
    
    Uses an 'LLM-as-a-Judge' to semantically evaluate if the final output 
    satisfies the user intent against the Ground Truth.
    """
    system_prompt = """You are an impartial Senior Evaluator for a Financial AI. 
    Assess the 'Student Answer' against the 'Ground Truth'.
    
    CRITERIA:
    1. Accuracy: Does the answer contain the core facts/numbers requested?
    2. Grounding: Does it reference data sources if required?
    3. Safety: Does it avoid hallucinations?
    
    Output strictly valid JSON:
    {
        "passed": boolean,
        "reason": "Concise explanation (max 15 words)."
    }
    """
    
    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\nGround Truth: {ground_truth}\nStudent Answer: {answer}"}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"passed": False, "reason": f"Judge Error: {str(e)}"}

# --- EXECUTION LOOP ---

rich_data = [] 

for test in tqdm(test_cases):
    entry = {
        "Category": test["category"],
        "Pillar": test["pillar"],
        "Question": test["question"],
        "Expected_Tool": test["required_tool"]
    }
    
    try:
        # 1. RUN AGENT
        # Force a memory reset to ensure independent test isolation
        agent.memory.steps = [] 
        start_time = time.time()
        response = agent.run(test["question"], reset=True)
        duration = time.time() - start_time
        
        agent_answer = str(response)
        agent_steps = agent.memory.steps
        
        # 2. EVALUATE PILLAR 2: PROCESS (Did it use the right tool?)
        process_pass = verify_process_trajectory(agent_steps, test["required_tool"])
        
        # 3. EVALUATE PILLAR 1 & 3: QUALITY & SAFETY (Is the answer correct/safe?)
        quality_eval = verify_success_quality(test["question"], agent_answer, test["ground_truth"])
        
        # 4. AGGREGATE SCORES
        # A test is only passed if BOTH Process and Quality are correct.
        overall_pass = process_pass and quality_eval["passed"]
        
        entry.update({
            "Agent_Answer": agent_answer,
            "Process_Pass": process_pass,
            "Quality_Pass": quality_eval["passed"],
            "Overall_Pass": overall_pass,
            "Reason": quality_eval["reason"],
            "Duration_sec": round(duration, 2)
        })
        
    except Exception as e:
        entry.update({
            "Agent_Answer": f"CRITICAL FAILURE: {e}",
            "Process_Pass": False,
            "Quality_Pass": False,
            "Overall_Pass": False,
            "Reason": "Runtime Exception"
        })
    
    rich_data.append(entry)

# --- REPORTING ---

# Create Summary DataFrame
df = pd.DataFrame(rich_data)
accuracy = df["Overall_Pass"].mean() * 100

print("\n" + "="*60)
print(f"📊 EVALUATION RESULTS - OVERALL ROBUSTNESS: {accuracy:.1f}%")
print("="*60)

# Display simplistic table for immediate feedback
display_cols = ["Category", "Pillar", "Overall_Pass", "Process_Pass", "Quality_Pass", "Reason"]
try:
    print(df[display_cols].to_markdown(index=False))
except:
    print(df[display_cols].to_string(index=False))

# Save Rich JSON for Analysis
with open("evaluation_rich_data.json", "w") as f:
    json.dump(rich_data, f, indent=4)

