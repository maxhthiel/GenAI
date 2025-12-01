from agent.agent import build_agent

if __name__ == "__main__":
    agent = build_agent()

    print("SmolAgent gestartet 🚀")
    while True:
        q = input("\nFrage: ")
        answer = agent.run(q)
        print("\nAntwort:", answer)
