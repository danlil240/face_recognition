import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import logging
from functools import lru_cache
import utils


class FaceRecognizer:

    def __init__(self, database):
        self.database = database
        self.similarity_threshold = 0.6
        self.improvement_threshold = 0.8
        self.max_embeddings = 5
        self.merge_threshold = 0.7
        self.embedding_cache = {}
        self.max_update_count = 100

    @staticmethod
    def _hash_image(image):
        return hash(image.tobytes())

    @lru_cache(maxsize=100)
    def process_face(self, face_img_bytes, shape):
        face_img = np.frombuffer(face_img_bytes, dtype=np.uint8).reshape(shape)
        try:
            with utils.suppress_stdout():
                embedding_dict = DeepFace.represent(
                    face_img,
                    model_name="Facenet512",
                    enforce_detection=False,
                    detector_backend="mtcnn",
                    align=True,
                    normalization="base",
                )
            if not embedding_dict:
                logging.warning("DeepFace.represent returned an empty result")
                return None
            embedding = np.array(embedding_dict[0]["embedding"])
            return self.normalize_embedding(embedding)
        except IndexError:
            logging.error(
                "IndexError: embedding_dict is empty or doesn't contain expected data"
            )
            return None
        except Exception as e:
            logging.error(f"Error processing face: {e}")
            return None

    @staticmethod
    def normalize_embedding(embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize_person(self, processed_embedding):
        persons = self.database.get_all_persons()
        max_similarity = -1
        best_match_id = None

        for person_id, _ in persons:
            embedding_info = self.database.get_person_embedding_info(person_id)
            if embedding_info:
                similarity = self.cosine_similarity(
                    processed_embedding, embedding_info["embedding"]
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = person_id

        if max_similarity < self.similarity_threshold:
            new_id = self.database.insert_person(processed_embedding)
            return new_id, None

        if max_similarity > self.improvement_threshold:
            with self.database._get_connection() as conn:
                self.database.update_embedding(
                    conn,
                    best_match_id,
                    processed_embedding,
                    max_similarity,
                    self.max_update_count,
                )

        return best_match_id, self.database.get_person_name(best_match_id)

    def update_embedding(self, person_id, new_embedding, similarity, database):
        embedding_info = database.get_person_embedding_info(person_id)
        if embedding_info:
            stored_embedding = embedding_info["embedding"]
            stored_weight = embedding_info["weight"]
            update_count = min(
                embedding_info["update_count"] + 1, self.max_update_count
            )

            # Calculate new weight based on similarity and update count
            new_weight = similarity * (1 / update_count)

            # Update the weighted average of the embedding
            updated_embedding = (
                stored_embedding * stored_weight + new_embedding * new_weight
            ) / (stored_weight + new_weight)
            updated_weight = stored_weight + new_weight

            # Normalize the updated embedding
            normalized_embedding = updated_embedding / np.linalg.norm(updated_embedding)

            database.update_embedding(
                person_id, normalized_embedding, updated_weight, update_count
            )
            logging.info(f"Updated embedding for person ID: {person_id}")

    @lru_cache(maxsize=100)
    def get_person_embeddings(self, person_id):
        embeddings = self.database.get_person_embeddings(person_id)
        return embeddings

    def find_similar_persons(self):
        persons = self.database.get_all_persons()
        similar_pairs = []

        for i, (id1, _) in enumerate(persons):
            for id2, _ in persons[i + 1 :]:
                similarity = self.calculate_person_similarity(id1, id2)
                logging.info(f"Similarity between {id1} and {id2}: {similarity}")
                if similarity > self.merge_threshold:
                    similar_pairs.append((id1, id2, similarity))

        return similar_pairs

    def calculate_person_similarity(self, id1, id2):
        emb1 = self.database.get_person_embedding_info(id1)["embedding"]
        emb2 = self.database.get_person_embedding_info(id2)["embedding"]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity

    def auto_merge_similar_persons(self):
        similar_pairs = self.find_similar_persons()
        merged_count = 0

        if not similar_pairs:
            logging.info("No similar persons found for merging.")
            return merged_count

        for id1, id2, similarity in similar_pairs:
            logging.info(
                f"Attempting to merge persons {id1} and {id2} with similarity {similarity}"
            )

            # Get names before merge
            name1 = self.database.get_person_name(id1)
            name2 = self.database.get_person_name(id2)

            if self.database.merge_persons(id1, id2):
                merged_count += 1
                logging.info(
                    f"Successfully merged persons {id1} ({name1}) and {id2} ({name2})"
                )
            else:
                logging.warning(
                    f"Failed to merge persons {id1} ({name1}) and {id2} ({name2})"
                )

        logging.info(f"Total merges performed: {merged_count}")
        return merged_count

    def real_time_merge(self, recognized_person_id, embedding, database):
        persons = database.get_all_persons()
        for other_id, _ in persons:
            if other_id != recognized_person_id:
                other_embedding_info = database.get_person_embedding_info(other_id)
                if other_embedding_info:
                    similarity = self.cosine_similarity(
                        embedding, other_embedding_info["embedding"]
                    )
                    if similarity > self.merge_threshold:
                        database.merge_persons(recognized_person_id, other_id)
                        logging.info(
                            f"Merged person {other_id} into {recognized_person_id}"
                        )
                        return True
        return False

    def adjust_thresholds(self):
        person_count = len(self.database.get_all_persons())
        if person_count > 100:
            self.similarity_threshold = 0.65
            self.merge_threshold = 0.75
        elif person_count > 50:
            self.similarity_threshold = 0.62
            self.merge_threshold = 0.72
        else:
            self.similarity_threshold = 0.6
            self.merge_threshold = 0.7
        logging.info(
            f"Adjusted thresholds: similarity={self.similarity_threshold}, merge={self.merge_threshold}"
        )

    def update_person_name(self, person_id, new_name):
        # Update any internal caches or data structures
        if person_id in self.embedding_cache:
            del self.embedding_cache[person_id]
        # You might need to update other internal data structures here
        logging.info(
            f"Updated name for person {person_id} to {new_name} in FaceRecognizer"
        )
