import os
import json
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorForLanguageModeling

class CitationIntentDataLoader:
    def __init__(self, model_name, dataset_name, path, checkpoint_path):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.path = path
        self.checkpoint_path = checkpoint_path
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.num_labels = None

    def load_dataset(self, overwrite=False):
        if os.path.exists(self.checkpoint_path) and not overwrite:
            loaded_dataset = torch.load(self.checkpoint_path)
            print("Dataset loaded from checkpoint.")
            self._update_labels()
        else:
            loaded_dataset = self._load_and_process_dataset()
            torch.save(loaded_dataset, self.checkpoint_path)
            print("Dataset saved as checkpoint at:", self.checkpoint_path)

        # Transform to pytorch tensors and only output the required columns
        loaded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return loaded_dataset

    def get_data_collator(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        return data_collator

    def _load_and_process_dataset(self):
        data_files = {
            "train": self.path + "train.jsonl",
            "test": self.path + "test.jsonl",
            "dev": self.path + "dev.jsonl"
        }
        dataset = load_dataset("json", data_files=data_files)

        self._update_labels()

        print("Processing #{} encoded labels: {}".format(self.num_labels, self.label_encoder))

        dataset = dataset.map(self._encode_and_convert)

        return dataset

    def _extract_labels(self, file_path):
        labels = set()  
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                labels.add(data["label"])
        return labels

    def _encode_and_convert(self, batch):
        return self.tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

    def _convert_labels_to_integers(self, example):
        return {'label': self.label_encoder[example['label']]}

    def _update_labels(self):
        train_labels = self._extract_labels(f'data/{self.dataset_name}/train.jsonl')
        dev_labels = self._extract_labels(f'data/{self.dataset_name}/dev.jsonl')
        test_labels = self._extract_labels(f'data/{self.dataset_name}/test.jsonl')

        all_labels = train_labels.union(dev_labels).union(test_labels)

        self.label_encoder = {label: idx for idx, label in enumerate(all_labels)}
        self.num_labels = len(self.label_encoder)