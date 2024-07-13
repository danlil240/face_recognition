import sqlite3
import numpy as np
import logging
import threading
from contextlib import contextmanager


class FaceDatabase:

    def __init__(self, db_name="face_recognition.db"):
        self.db_name = db_name
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables(conn)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_name, timeout=60)
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT);

            CREATE TABLE IF NOT EXISTS embeddings
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             person_id INTEGER,
             embedding BLOB,
             weight REAL,
             update_count INTEGER,
             FOREIGN KEY(person_id) REFERENCES persons(id));
        """
        )

    def migrate_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Check if the update_count column exists
                cursor.execute("PRAGMA table_info(embeddings)")
                columns = [column[1] for column in cursor.fetchall()]

                if "update_count" not in columns:
                    # Add the update_count column
                    cursor.execute(
                        "ALTER TABLE embeddings ADD COLUMN update_count INTEGER DEFAULT 1"
                    )
                    conn.commit()
                    logging.info("Added update_count column to embeddings table")
            except Exception as e:
                logging.error(f"Error migrating database: {e}")

    def merge_persons(self, person_id1, person_id2):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Ensure person_id1 is the lower ID
                if person_id1 > person_id2:
                    person_id1, person_id2 = person_id2, person_id1

                name1 = self.get_person_name(person_id1)
                name2 = self.get_person_name(person_id2)

                # Merge names if necessary
                if name1 == name2:
                    merged_name = name1
                elif name1 and name2:
                    merged_name = f"{name1}/{name2}"
                else:
                    merged_name = name1 or name2 or ""

                cursor.execute(
                    "UPDATE persons SET name = ? WHERE id = ?",
                    (merged_name, person_id1),
                )

                # Fetch embeddings for both persons
                cursor.execute(
                    "SELECT embedding, weight, update_count FROM embeddings WHERE person_id IN (?, ?)",
                    (person_id1, person_id2),
                )
                embeddings = cursor.fetchall()

                # Combine embeddings
                combined_embedding = np.zeros(512, dtype=np.float32)
                total_weight = 0
                total_count = 0

                for emb, weight, count in embeddings:
                    embedding_array = np.frombuffer(emb, dtype=np.float32)
                    combined_embedding += embedding_array * weight
                    total_weight += weight
                    total_count += count

                if total_weight > 0:
                    combined_embedding /= total_weight
                combined_embedding = combined_embedding / np.linalg.norm(
                    combined_embedding
                )
                # Update the merged embedding
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embeddings 
                    (person_id, embedding, weight, update_count) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        person_id1,
                        sqlite3.Binary(combined_embedding.tobytes()),
                        total_weight,
                        total_count,
                    ),
                )

                # Delete the old person and their embeddings
                cursor.execute(
                    "DELETE FROM embeddings WHERE person_id = ?", (person_id2,)
                )
                cursor.execute("DELETE FROM persons WHERE id = ?", (person_id2,))

                conn.commit()
                logging.info(
                    f"Persons {person_id1} and {person_id2} merged successfully."
                )
                return True
            except Exception as e:
                logging.error(f"Error merging persons: {e}")
                conn.rollback()
                return False

    def insert_person(self, embedding, name=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
                person_id = cursor.lastrowid
                self.update_embedding(cursor, person_id, embedding, 1.0, 1)
                conn.commit()
                return person_id
        except sqlite3.Error as e:
            logging.error(f"Error inserting person: {e}")
            return None

    def add_embedding(self, person_id, embedding):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                embedding_bytes = embedding.astype(np.float32).tobytes()
                cursor.execute(
                    "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                    (person_id, sqlite3.Binary(embedding_bytes)),
                )
                conn.commit()
            except Exception as e:
                logging.error(f"Error adding embedding: {e}")

    def update_embedding(self, cursor, person_id, new_embedding, weight, update_count):
        try:
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
            embedding_bytes = new_embedding.astype(np.float32).tobytes()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (person_id, embedding, weight, update_count) 
                VALUES (?, ?, ?, ?)
            """,
                (person_id, sqlite3.Binary(embedding_bytes), weight, update_count),
            )
            logging.info(f"Updated embedding for person ID: {person_id}")
        except sqlite3.Error as e:
            logging.error(f"Error updating embedding: {e}")

    def get_person_embeddings(self, person_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT embedding FROM embeddings WHERE person_id = ?", (person_id,)
                )
                embeddings = []
                for embedding in cursor.fetchall():
                    try:
                        e = np.frombuffer(embedding[0], dtype=np.float32)
                        if e.shape == (512,):
                            embeddings.append(e)
                        else:
                            logging.warning(
                                f"Invalid embedding shape {e.shape} for person {person_id}"
                            )
                    except Exception as e:
                        logging.error(
                            f"Error processing embedding for person {person_id}: {e}"
                        )
                return embeddings
            except Exception as e:
                logging.error(f"Error fetching person embeddings: {e}")
                return []

    def get_person_embedding_info(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT embedding, weight, update_count 
                    FROM embeddings 
                    WHERE person_id = ?
                """,
                    (person_id,),
                )
                result = cursor.fetchone()
                if result:
                    return {
                        "embedding": np.frombuffer(result[0], dtype=np.float32),
                        "weight": result[1],
                        "update_count": result[2],
                    }
                return None
        except sqlite3.Error as e:
            logging.error(f"Error fetching person embedding info: {e}")
            return None

    def get_all_persons(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM persons")
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error fetching all persons: {e}")
            return []

    def get_person_name(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            logging.error(f"Error fetching person name: {e}")
            return None

    def update_person_name(self, person_id, name):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE persons SET name = ? WHERE id = ?", (name, person_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error updating person name: {e}")

    def delete_person(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM embeddings WHERE person_id = ?", (person_id,)
                )
                cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error deleting person: {e}")

    def clean_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Remove embeddings that can't be converted to numpy arrays
                cursor.execute("SELECT id, embedding FROM embeddings")
                invalid_ids = []
                for row in cursor.fetchall():
                    embedding_id, embedding_bytes = row
                    try:
                        np.frombuffer(embedding_bytes, dtype=np.float32)
                    except:
                        invalid_ids.append(embedding_id)

                if invalid_ids:
                    placeholders = ",".join("?" for _ in invalid_ids)
                    cursor.execute(
                        f"DELETE FROM embeddings WHERE id IN ({placeholders})",
                        invalid_ids,
                    )

                # Remove persons with no associated embeddings
                cursor.execute(
                    """
                    DELETE FROM persons 
                    WHERE id NOT IN (SELECT DISTINCT person_id FROM embeddings)
                """
                )

                conn.commit()
                removed_count = len(invalid_ids) + cursor.rowcount
                logging.info(
                    f"Cleaned database. Removed {removed_count} invalid entries."
                )
                return removed_count
            except Exception as e:
                logging.error(f"Error cleaning database: {e}")
                conn.rollback()
                return 0

    def reset_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Delete all data from the tables
                cursor.execute("DELETE FROM embeddings")
                cursor.execute("DELETE FROM persons")

                # Reset the auto-increment counters
                cursor.execute(
                    "DELETE FROM sqlite_sequence WHERE name='embeddings' OR name='persons'"
                )

                # Commit the changes
                conn.commit()

                # Vacuum the database to reclaim freed space and reset internal IDs
                conn.execute("VACUUM")

                logging.info(
                    "Database reset successfully. All data has been erased and IDs reset."
                )
                return True
            except Exception as e:
                logging.error(f"Error resetting database: {e}")
                return False
