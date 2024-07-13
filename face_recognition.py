import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import logging
from functools import lru_cache


class FaceRecognizer:
    def __init__(self, database):
        self.database = database
        self.similarity_threshold = 0.6
        self.improvement_threshold = 0.8
        self.max_embeddings = 5
        self.merge_threshold = 0.7
        self.embedding_cache = {}

    @staticmethod
    def _hash_image(image):
        return hash(image.tobytes())

    @lru_cache(maxsize=100)
    def process_face(self, face_img_bytes, shape):
        logging.info(f"Processing face image with shape: {shape}")
        face_img = np.frombuffer(face_img_bytes, dtype=np.uint8).reshape(shape)
        try:
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

    def recognize_person(self, processed_embedding):
        persons = self.database.get_all_persons()
        max_similarity = -1
        person_id, person_name = None, None

        for p_id, p_name in persons:
            stored_embeddings = self.get_person_embeddings(p_id)
            similarities = cosine_similarity(
                processed_embedding.reshape(1, -1), np.array(stored_embeddings)
            )
            max_person_similarity = np.max(similarities)
            if max_person_similarity > max_similarity:
                max_similarity = max_person_similarity
                person_id, person_name = p_id, p_name

        if max_similarity < self.similarity_threshold:
            new_id = self.database.insert_person(processed_embedding)
            return new_id, None

        if max_similarity > self.improvement_threshold:
            self.improve_embedding(person_id, processed_embedding)

        return person_id, person_name

    def improve_embedding(self, person_id, new_embedding):
        stored_embeddings = self.get_person_embeddings(person_id)
        if len(stored_embeddings) < self.max_embeddings:
            self.database.add_embedding(person_id, new_embedding)
        else:
            similarities = cosine_similarity(
                new_embedding.reshape(1, -1), np.array(stored_embeddings)
            )
            most_similar_index = np.argmin(similarities)
            self.database.update_embedding(
                person_id, int(most_similar_index), new_embedding
            )
        self.embedding_cache.pop(person_id, None)  # Invalidate cache

    @lru_cache(maxsize=100)
    def get_person_embeddings(self, person_id):
        embeddings = self.database.get_person_embeddings(person_id)
        valid_embeddings = [e for e in embeddings if e.shape == (512,)]
        if len(valid_embeddings) != len(embeddings):
            logging.warning(
                f"Filtered out {len(embeddings) - len(valid_embeddings)} invalid embeddings for person {person_id}"
            )
        return valid_embeddings

    def find_similar_persons(self):
        persons = self.database.get_all_persons()
        similar_pairs = []

        for i, (id1, _) in enumerate(persons):
            embeddings1 = self.get_person_embeddings(id1)
            if not embeddings1:
                continue
            embeddings1 = np.array([e for e in embeddings1 if e.shape == (512,)])
            if embeddings1.size == 0:
                logging.warning(f"No valid embeddings found for person {id1}")
                continue
            mean_embedding1 = np.mean(embeddings1, axis=0)

            for id2, _ in persons[i + 1 :]:
                embeddings2 = self.get_person_embeddings(id2)
                if not embeddings2:
                    continue
                embeddings2 = np.array([e for e in embeddings2 if e.shape == (512,)])
                if embeddings2.size == 0:
                    logging.warning(f"No valid embeddings found for person {id2}")
                    continue
                mean_embedding2 = np.mean(embeddings2, axis=0)

                similarity = cosine_similarity([mean_embedding1], [mean_embedding2])[
                    0, 0
                ]
                if similarity > self.merge_threshold:
                    similar_pairs.append((id1, id2, similarity))

        return similar_pairs

    def auto_merge_similar_persons(self):
        similar_pairs = self.find_similar_persons()
        merged_count = 0

        if not similar_pairs:
            logging.info("No similar persons found for merging.")
            return merged_count

        for id1, id2, _ in similar_pairs:
            self.database.merge_persons(id1, id2)
            merged_count += 1
            logging.info(f"Merged person {id2} into {id1}")
            self.embedding_cache.pop(id1, None)  # Invalidate cache
            self.embedding_cache.pop(id2, None)  # Invalidate cache

        return merged_count

    def real_time_merge(self, recognized_person_id, embedding):
        persons = self.database.get_all_persons()
        for other_id, _ in persons:
            if other_id != recognized_person_id:
                other_embeddings = self.get_person_embeddings(other_id)
                max_similarity = np.max(
                    cosine_similarity(
                        embedding.reshape(1, -1), np.array(other_embeddings)
                    )
                )
                if max_similarity > self.merge_threshold:
                    self.database.merge_persons(recognized_person_id, other_id)
                    logging.info(
                        f"Merged person {other_id} into {recognized_person_id}"
                    )
                    self.embedding_cache.pop(
                        recognized_person_id, None
                    )  # Invalidate cache
                    self.embedding_cache.pop(other_id, None)  # Invalidate cache
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
