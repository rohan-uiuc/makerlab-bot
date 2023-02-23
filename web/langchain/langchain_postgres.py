# Code for the LangChainDB class, which handles the storage and retrieval of conversation history.

import psycopg2

class LangChainDB:
    def __init__(self, host, port, name, user, password):
        self.connection = psycopg2.connect(
            host=host,
            port=port,
            database=name,
            user=user,
            password=password
        )
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def insert_conversation(self, input_text, output_text):
        self.cursor.execute(
            """
            INSERT INTO conversations (input_text, output_text) VALUES (%s, %s);
            """,
            (input_text, output_text)
        )
        self.connection.commit()

    def get_conversations(self):
        self.cursor.execute(
            """
            SELECT input_text, output_text FROM conversations;
            """
        )
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.connection.close()
