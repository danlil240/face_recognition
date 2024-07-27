import sqlite3
import numpy as np
import logging
import threading
from contextlib import contextmanager
import utils


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
            self._migrate_database(conn)

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
             name TEXT,
             embedding BLOB,
             count INTEGER,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
            """
        )

    def _migrate_database(self, conn):
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(persons)")
        except Exception as e:
            logging.error(f"Error migrating database: {e}")
            conn.rollback()

    def _get_lowest_available_id(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM persons ORDER BY id")
        used_ids = set(row[0] for row in cursor.fetchall())

        # Find the lowest available ID
        lowest_id = 1
        while lowest_id in used_ids:
            lowest_id += 1

        return lowest_id

    def insert_person(self, embedding, name=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                embedding_bytes = embedding.astype(np.float32).tobytes()

                # Get the lowest available ID
                lowest_id = self._get_lowest_available_id(conn)

                cursor.execute(
                    "INSERT INTO persons (id, name, embedding, count) VALUES (?, ?, ?, ?)",
                    (lowest_id, name, sqlite3.Binary(embedding_bytes), 1),
                )
                conn.commit()
                return lowest_id
        except sqlite3.Error as e:
            logging.error(f"Error inserting person: {e}")
            return None

    def update_embedding(self, person_id, new_embedding):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding, count FROM persons WHERE id = ?", (person_id,)
                )
                result = cursor.fetchone()
                old_embedding = result[0]
                count = result[1]
                count += 1
                if old_embedding is not None:
                    old_embedding = np.frombuffer(old_embedding, dtype=np.float32)
                new_embedding = 0.95 * old_embedding + 0.05 * new_embedding
                new_embedding = utils.normalize_embedding(new_embedding)
                embedding_bytes = new_embedding.astype(np.float32).tobytes()
                cursor.execute(
                    "UPDATE persons SET embedding = ?, count = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (sqlite3.Binary(embedding_bytes), count, person_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error updating embedding: {e}")

    def merge_persons(self, person1_id, person2_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if person1_id > person2_id:
                    temp1 = person1_id
                    person1_id = person2_id
                    person2_id = temp1
                # Fetch data for both persons
                cursor.execute(
                    "SELECT name, embedding, count FROM persons WHERE id IN (?, ?)",
                    (person1_id, person2_id),
                )
                results = cursor.fetchall()

                if len(results) != 2:
                    logging.error(
                        f"Could not find both persons with IDs {person1_id} and {person2_id}"
                    )
                    return False

                person1_data, person2_data = results
                person1_name, person1_embedding, person1_count = person1_data
                person2_name, person2_embedding, person2_count = person2_data

                # Decide which name to keep
                if person1_name and person2_name:
                    return

                merged_name = person1_name if person1_name else person2_name

                # Combine embeddings
                if person1_embedding and person2_embedding:
                    embedding1 = np.frombuffer(person1_embedding, dtype=np.float32)
                    embedding2 = np.frombuffer(person2_embedding, dtype=np.float32)
                    merged_embedding = (
                        embedding1 * person1_count + embedding2 * person2_count
                    )
                    merged_embedding = utils.normalize_embedding(merged_embedding)
                    merged_embedding = merged_embedding.astype(np.float32)
                else:
                    merged_embedding = person1_embedding or person2_embedding

                # Update person1 with merged data
                cursor.execute(
                    "UPDATE persons SET name = ?, embedding = ?, count = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (
                        merged_name,
                        sqlite3.Binary(merged_embedding.tobytes()),
                        person1_count + person2_count,
                        person1_id,
                    ),
                )

                # Delete person2
                cursor.execute("DELETE FROM persons WHERE id = ?", (person2_id,))

                # Commit the changes
                conn.commit()

                logging.info(
                    f"Successfully merged persons {person1_id} and {person2_id}"
                )
                return True

        except sqlite3.Error as e:
            logging.error(f"Error merging persons: {e}")
            return False

    def get_person_embedding(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM persons WHERE id = ?", (person_id,)
                )
                result = cursor.fetchone()
                if result and result[0]:
                    return np.frombuffer(result[0], dtype=np.float32)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error fetching person embedding: {e}")
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

    def get_person_count(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count FROM persons WHERE id = ?", (person_id,))
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
                    "UPDATE persons SET name = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (name, person_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error updating person name: {e}")

    def delete_person(self, person_id):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
                conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error deleting person: {e}")

    def clean_old_low_count_entries(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Delete entries older than 20 seconds with count less than 15
                cursor.execute(
                    """
                    DELETE FROM persons 
                    WHERE julianday('now') - julianday(timestamp) > 20.0 / 86400.0
                    AND count < 15
                """
                )

                deleted_count = cursor.rowcount
                conn.commit()
                if deleted_count > 0:
                    logging.info(
                        f"Cleaned {deleted_count} entries older than 20 seconds with count less than 15."
                    )
                return deleted_count

        except sqlite3.Error as e:
            logging.error(f"Error cleaning old entries with low count: {e}")
            return 0

    def clean_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT id, embedding FROM persons")
                invalid_ids = []
                for row in cursor.fetchall():
                    person_id, embedding_bytes = row
                    if embedding_bytes is None:
                        invalid_ids.append(person_id)
                    else:
                        try:
                            np.frombuffer(embedding_bytes, dtype=np.float32)
                        except:
                            invalid_ids.append(person_id)

                if invalid_ids:
                    placeholders = ",".join("?" for _ in invalid_ids)
                    cursor.execute(
                        f"DELETE FROM persons WHERE id IN ({placeholders})", invalid_ids
                    )

                conn.commit()
                removed_count = len(invalid_ids)
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
                cursor.execute("DROP TABLE IF EXISTS persons")
                conn.commit()
                self._create_tables(conn)
                conn.commit()
                logging.info("Database reset successfully.")
            except sqlite3.Error as e:
                logging.error(f"Error resetting database: {e}")
                conn.rollback()
