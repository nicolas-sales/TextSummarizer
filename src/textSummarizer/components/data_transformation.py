import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.textSummarizer.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        dialogues = [str(d) for d in example_batch["dialogue"]]
        summaries = [str(s) for s in example_batch["summary"]]

        # Inputs
        input_encodings = self.tokenizer(
            dialogues,
            max_length=512,  # attention, 1024 peut faire crasher sur GPU faible
            padding="max_length",
            truncation=True
        )

        # Targets
        target_encodings = self.tokenizer(
            summaries,
            max_length=128,
            padding="max_length",
            truncation=True
        )

        # Replace [PAD] by -100 in the labels
        labels = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in target_encodings["input_ids"]
        ]

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))