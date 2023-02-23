import sqlite3


class LangChainDB:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS conversation_history (user_input TEXT, response TEXT, history_key TEXT)"
        )
        self.conn.commit()

    def store(self, history_key, conversation_history):
        for user_input, response in conversation_history:
            self.cursor.execute(
                "INSERT INTO conversation_history (user_input, response, history_key) VALUES (?, ?, ?)",
                (user_input, response, history_key),
            )
            self.conn.commit()

    def retrieve(self, history_key):
        self.cursor.execute(
            "SELECT user_input, response FROM conversation_history WHERE history_key=?", (history_key,)
        )
        rows = self.cursor.fetchall()
        return rows



    def delete_all_conversations(self) -> None:
        """Delete all conversations from the table."""
        self.cursor.execute("""
            DELETE FROM conversations
        """)
        self.conn.commit()

    def close_connection(self) -> None:
        """Close the connection to the database."""
        self.conn.close()
