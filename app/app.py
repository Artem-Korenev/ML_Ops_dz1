import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

sys.path.append(os.path.abspath("./src"))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/app/logs/service.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info("Initializing ProcessingService...")

        self.input_dir = os.path.abspath("input")
        self.output_dir = os.path.abspath("output")

        self.train, self.encoder, self.scaler = load_train_data()
        logger.info("Service initialized")

    def process_single_file(self, file_path):
        try:

            # Проверяем доступ
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Нет прав на чтение файла {file_path}")

            logger.info("Processing file: %s", file_path)
            input_df = pd.read_csv(file_path)

            logger.info("Starting preprocessing")
            processed_df = run_preproc(self.encoder, self.scaler, input_df)

            logger.info("Making prediction")
            submission = make_pred(processed_df, file_path)

            logger.info("Prepraring submission file")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(
                os.path.join(self.output_dir, output_filename), index=False
            )
            logger.info("Predictions saved to: %s", output_filename)

        except Exception as e:
            logger.error("Error processing file %s: %s", file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        print("Event detected:", event.src_path)  # Используется для диагностики
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.info("New file detected: %s", event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info("Starting ML scoring service...")
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info("File observer started")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        observer.stop()
    observer.join()
