import chromadb

# Pfad zu deiner Datenbank (wie in main.py)
DB_PATH = "agent/data/nasdaq_chroma_db"

def list_collections():
    print(f"🔍 Untersuche Datenbank unter: {DB_PATH}")
    
    try:
        # Verbindung herstellen
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Alle Collections abrufen
        collections = client.list_collections()
        
        if not collections:
            print("❌ Keine Collections gefunden (Datenbank ist leer).")
            return

        print(f"✅ {len(collections)} Collection(s) gefunden:\n")
        
        for col in collections:
            count = col.count()
            print(f"📂 Name: '{col.name}'")
            print(f"   📊 Anzahl Dokumente: {count}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Fehler beim Zugriff auf die DB: {e}")

if __name__ == "__main__":
    list_collections()