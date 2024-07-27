from face_recognition import FaceRecognizer
from database import FaceDatabase
import queue
import threading
import concurrent.futures
import logging
import time
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class AdvancedVideoProcessor:
    def __init__(
        self,
        recognizer: FaceRecognizer,
        database: FaceDatabase,
        recognition_threads: int = 4,
    ):
        self.recognizer = recognizer
        self.database = database
        self.recognition_threads = recognition_threads

        self.frame_queue = queue.Queue(maxsize=recognition_threads * 2)
        self.recognition_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.running = threading.Event()
        self.frame_count = 0
        self.frame_count_lock = threading.Lock()

        self.recognition_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=recognition_threads, thread_name_prefix="Recognition"
        )

        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(threadName)s - %(message)s"
        )

    def start_processing(self):
        self.running.set()
        self.recognition_futures = []
        for _ in range(self.recognition_threads):
            future = self.recognition_executor.submit(self.recognition_worker)
            self.recognition_futures.append(future)

    def stop_processing(self):
        self.running.clear()
        for _ in range(self.recognition_threads):
            self.recognition_queue.put((None, None))
        concurrent.futures.wait(self.recognition_futures)
        self.recognition_executor.shutdown(wait=True)

    def recognition_worker(self):
        while self.running.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break

                start_time = time.time()

                processed_embeddings, faces_area = self.recognizer.process_face(frame)
                for embedding, face_area in zip(processed_embeddings, faces_area):
                    if embedding is not None and face_area is not None:
                        person_id, person_name = self.recognizer.recognize_person(
                            embedding
                        )
                        self.result_queue.put((face_area, person_id, person_name))

                end_time = time.time()
                duration = end_time - start_time
                # logging.debug(f"Recognition took {duration:.4f} seconds")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in recognition worker: {e}", exc_info=True)

    def add_frame(self, frame):
        with self.frame_count_lock:
            self.frame_count += 1
        try:
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            logger.warning("Frame queue is full, skipping frame")

    def get_results(self) -> List[Tuple[Any, Any, Any]]:
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
