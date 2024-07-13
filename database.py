import sqlite3
import numpy as np
import logging


class FaceDatabase:
    def __init__(self, db_name="face_recognition.db"):
        self.conn = sqlite3.connect(db_name)
        self.c = self.conn.cursor()
        self._create_tables()
        logging.info("Database initialized.")

    def open_connection(self):
        self.conn = sqlite3.connect(self.db_name)
        self.c = self.conn.cursor()

    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.c = None

    def _create_tables(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS persons
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT)"""
        )
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS embeddings
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          person_id INTEGER,
                          embedding BLOB,
                          FOREIGN KEY(person_id) REFERENCES persons(id))"""
        )
        self.conn.commit()

    def merge_persons(self, person_id1, person_id2):
        try:
            name1 = self.get_person_name(person_id1)
            name2 = self.get_person_name(person_id2)
            if name1 and name2:
                return
            if name1 or name2:
                name = name1 or name2
                self.c.execute(
                    "UPDATE persons SET name = ? WHERE id = ?",
                    (name, person_id1),
                )
            self.c.execute(
                "UPDATE embeddings SET person_id = ? WHERE person_id = ?",
                (person_id1, person_id2),
            )
            self.c.execute("DELETE FROM persons WHERE id = ?", (person_id2,))

            self.conn.commit()
            logging.info(f"Persons {person_id1} and {person_id2} merged successfully.")
        except Exception as e:
            logging.error(f"Error merging persons: {e}")

    def insert_person(self, embedding, name=None):
        try:
            self.c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            person_id = self.c.lastrowid
            self.add_embedding(person_id, embedding)
            self.conn.commit()
            return person_id
        except Exception as e:
            logging.error(f"Error inserting person: {e}")
            return None

    def add_embedding(self, person_id, embedding):
        try:
            embedding_bytes = embedding.astype(np.float32).tobytes()
            self.c.execute(
                "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                (person_id, sqlite3.Binary(embedding_bytes)),
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error adding embedding: {e}")

    def update_embedding(self, person_id, embedding_index, new_embedding):
        try:
            embedding_bytes = new_embedding.astype(np.float32).tobytes()
            self.c.execute(
                """UPDATE embeddings SET embedding = ? 
                              WHERE person_id = ? AND id IN 
                              (SELECT id FROM embeddings WHERE person_id = ? 
                               ORDER BY id LIMIT 1 OFFSET ?)""",
                (
                    sqlite3.Binary(embedding_bytes),
                    person_id,
                    person_id,
                    embedding_index,
                ),
            )
            self.conn.commit()
            logging.info(f"Updated embedding for person ID: {person_id}")
        except Exception as e:
            logging.error(f"Error updating embedding: {e}")

    def get_person_embeddings(self, id):
        try:
            self.c.execute(
                "SELECT embedding FROM embeddings WHERE person_id = ?", (id,)
            )
            embeddings = []
            for embedding in self.c.fetchall():
                try:
                    e = np.frombuffer(embedding[0], dtype=np.float32)
                    if e.shape == (512,):
                        embeddings.append(e)
                    else:
                        logging.warning(
                            f"Invalid embedding shape {e.shape} for person {id}"
                        )
                except Exception as e:
                    logging.error(f"Error processing embedding for person {id}: {e}")
            return embeddings
        except Exception as e:
            logging.error(f"Error fetching person embeddings: {e}")
            return []

    def get_all_persons(self):
        try:
            self.c.execute("SELECT id, name FROM persons")
            return self.c.fetchall()
        except Exception as e:
            logging.error(f"Error fetching all persons: {e}")
            return []

    def get_person_name(self, person_id):
        try:
            self.c.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
            result = self.c.fetchone()
            return result[0] if result else None
        except Exception as e:
            logging.error(f"Error fetching person name: {e}")
            return None

    def update_person_name(self, id, name):
        try:
            self.c.execute("UPDATE persons SET name = ? WHERE id = ?", (name, id))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error updating person name: {e}")

    def delete_person(self, id):
        try:
            self.c.execute("DELETE FROM embeddings WHERE person_id = ?", (id,))
            self.c.execute("DELETE FROM persons WHERE id = ?", (id,))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error deleting person: {e}")

    def close(self):
        self.close_connection()
        logging.info("Database connection closed.")

    def clean_database(self):
        try:
            # Remove embeddings that can't be converted to numpy arrays
            self.c.execute("SELECT id, embedding FROM embeddings")
            invalid_ids = []
            for row in self.c.fetchall():
                embedding_id, embedding_bytes = row
                try:
                    np.frombuffer(embedding_bytes, dtype=np.float32)
                except:
                    invalid_ids.append(embedding_id)

            if invalid_ids:
                placeholders = ",".join("?" for _ in invalid_ids)
                self.c.execute(
                    f"DELETE FROM embeddings WHERE id IN ({placeholders})", invalid_ids
                )

            # Remove persons with no associated embeddings
            self.c.execute(
                """
                DELETE FROM persons 
                WHERE id NOT IN (SELECT DISTINCT person_id FROM embeddings)
            """
            )

            self.conn.commit()
            removed_count = len(invalid_ids) + self.c.rowcount
            logging.info(f"Cleaned database. Removed {removed_count} invalid entries.")
            return removed_count
        except Exception as e:
            logging.error(f"Error cleaning database: {e}")
            self.conn.rollback()
            return 0

    def reset_database(self):
        try:
            # Delete all data from the tables
            self.c.execute("DELETE FROM embeddings")
            self.c.execute("DELETE FROM persons")

            # Reset the auto-increment counters
            self.c.execute(
                "DELETE FROM sqlite_sequence WHERE name='embeddings' OR name='persons'"
            )

            # Commit the changes
            self.conn.commit()

            # Close the current connection
            self.close_connection()

            # Reopen the connection
            self.open_connection()

            # Vacuum the database to reclaim freed space and reset internal IDs
            self.conn.execute("VACUUM")

            logging.info(
                "Database reset successfully. All data has been erased and IDs reset."
            )
            return True
        except Exception as e:
            logging.error(f"Error resetting database: {e}")
            return False
