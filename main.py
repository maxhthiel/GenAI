from agent.agent_builder import build_agent
import logging
import sys

# Logging konfigurieren
# Wenn es dir zu viel Text ist, ändere INFO zu WARNING
logging.basicConfig(level=logging.INFO)

def main():
    print("--------------------------------------------------")
    print("🚀 Smol-Quant Terminal Interface gestartet")
    print("--------------------------------------------------")
    print("Lade Agenten und Werkzeuge... bitte warten.")
    
    try:
        # Agent nur EINMAL initialisieren (spart Zeit)
        agent = build_agent()
        print("✅ System bereit. (Schreibe 'exit' oder 'quit' zum Beenden)")
    except Exception as e:
        print(f"❌ Fehler beim Starten des Agenten: {e}")
        return

    while True:
        try:
            print("\n" + "="*50)
            # Input vom User abfragen
            user_input = input("Du: ").strip()

            # Abbruchbedingung
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Beende Sitzung.")
                break

            if not user_input:
                continue

            print(f"🤖 Agent denkt nach...")
            
            # Agent ausführen
            # Der Agent behält in dieser Session (normalerweise) kein Gedächtnis über 
            # vorherige Fragen, es sei denn, man baut Memory explizit ein. 
            # Er behandelt jede Frage als neu.
            response = agent.run(user_input)

            print(f"\n🤖 Antwort:\n{response}")

        except KeyboardInterrupt:
            # Damit man mit CTRL+C sauber rauskommt
            print("\n👋 Abbruch durch User.")
            break
        except Exception as e:
            print(f"❌ Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    main()