from agent.agent import build_agent

agent = build_agent()

# Testfrage
q = "Give me a summary of the CSV"
answer = agent.run(q)
print(answer)
