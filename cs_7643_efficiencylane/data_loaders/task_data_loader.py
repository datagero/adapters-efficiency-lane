import os
import json
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorForLanguageModeling

class TaskDataLoader:
    """
    Note, currently only support roberta-base model and tokenizer.
    Has been tested for processing all task datasets in https://github.com/allenai/dont-stop-pretraining
        - For AG News, the test dataset has no id column. A manual fix is applied to add an id column, but this has to be uncommented.
    But it has only been used to run models for the below tasks:
        - citation_intent
        - sciie
        - chemprot
    """
    def __init__(self, model_name, dataset_name, path, checkpoint_path):

        if model_name not in ['roberta-base']:
            raise ValueError("Invalid model_name. Supported datasets: roberta-base")

        if dataset_name not in ['citation_intent', 'sciie', 'chemprot']:
            print("Trying to run with unsupported dataset. Supported datasets: citation_intent, sciie, chemprot")

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

        ## MANUAL FIX FOR ONE OF OUR DATASETS (AG test.jsonl has no id column) 
        ## The fix is not productionized
        # # Check that datafiles contain the same columns
        # with open(data_files["train"], 'r') as file:
        #     columns = set(json.loads(file.readline()).keys())
        
        # for file in data_files.values():
        #     with open(file, 'r') as file:
        #         if columns != set(json.loads(file.readline()).keys()):
        #             print("File:", file)
        #             print("Columns in data files:", columns)
        #             print("Columns in file:", set(json.loads(file.readline()).keys()))
        #             raise ValueError("Data files do not contain the same columns.")

        # # Identified problem with ag test dataset
        # # Add an id column to the dataset
        # # First, make backup of the original file
        # import os
        # backup_name = data_files['test'].replace('.jsonl', '_original.jsonl')
        # os.system(f"cp {data_files['test']} {backup_name}")

        # with open(backup_name, 'r') as file:
        #     data = [json.loads(line) for line in file]
        #     for idx, item in enumerate(data):
        #         item["id"] = idx
        #     with open(data_files['test'], 'w') as file:
        #         for item in data:
        #             json.dump(item, file)
        #             file.write("\n")

        dataset = load_dataset("json", data_files=data_files)
        self._update_labels()

        print("Processing #{} encoded labels: {}".format(self.num_labels, self.label_encoder))

        dataset = dataset.map(self._encode_batch)
        dataset = dataset.map(self._convert_labels_to_integers)
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
        
        return dataset

    def _extract_labels(self, file_path):
        labels = set()  
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                labels.add(data["label"])
        return labels

    def _encode_batch(self, batch):
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