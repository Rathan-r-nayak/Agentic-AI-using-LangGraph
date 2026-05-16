import os
import sqlite3
from Utils.Logger import get_logger

logger = get_logger("DATABASE_MANAGER")

# Get the absolute path of the project root (one level up from Utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "DataStore")
DB_PATH = os.path.join(DB_DIR, "long_term_memory.db")

def setup_memory_db():
    """Ensure the database directory and table exist."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # UNIQUE ensures we don't save duplicate facts for the same user
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (user_id TEXT, fact TEXT, UNIQUE(user_id, fact))"
    )
    conn.commit()
    conn.close()
    logger.info(f"Long-Term Memory DB initialized at: {DB_PATH}")

# Run setup immediately when this file is imported
setup_memory_db()

def save_user_fact(user_id: str, fact: str):
    """Save a single new fact to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # INSERT OR IGNORE silently skips if the exact fact already exists
        conn.execute(
            "INSERT OR IGNORE INTO facts (user_id, fact) VALUES (?, ?)", 
            (user_id, fact)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving fact: {e}")

def get_user_facts(user_id: str) -> str:
    """Retrieve all facts for a specific user and format them as a string."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT fact FROM facts WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            # Returns a clean bulleted list for the LLM to read
            return "\n".join([f"- {row[0]}" for row in rows])
    except Exception as e:
        logger.error(f"Error retrieving facts: {e}")
    
    return "No known facts about user."