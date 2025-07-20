import os
import urllib.request as request
from src.textSummarizer.logging import logger
from datasets import load_dataset

from src.textSummarizer.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_and_save_dataset(self):
        dataset = load_dataset(self.config.dataset_name)
        dataset.save_to_disk(self.config.save_path)
        logger.info(f"Dataset '{self.config.dataset_name}' saved to {self.config.save_path}")