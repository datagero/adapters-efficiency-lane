"""
pip install -qq adapters datasets
pip install accelerate
pip install transformers[torch]
"""
"""# Import libraries"""
from datasets import load_dataset
import json
from transformers import RobertaTokenizer, RobertaConfig
from transformers import RobertaForSequenceClassification, Trainer, EvalPrediction, TrainingArguments, EarlyStoppingCallback
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import matplotlib.pyplot as plt
from transformers import set_seed as transformers_set_seed


task_models = {
    "citation_intent": {
        "base": "roberta-base",
        "dapt": "allenai/cs_roberta_base",
        "tapt": "allenai/dsp_roberta_base_tapt_citation_intent_1688",
        "dapt_tapt": "allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688"
    },
    "sciie": {
        "base": "roberta-base",
        "dapt": "allenai/cs_roberta_base",
        "tapt": "allenai/dsp_roberta_base_tapt_sciie_3219",
        "dapt_tapt": "allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219"
    }
}

"""# Set up the dataset and the model"""
def setup_dataset(dataset_name, model_name='roberta-base', max_length=80):
    print("Loading dataset:" + dataset_name)
    path = f"data/{dataset_name}/"

    # Load all splits at once
    data_files = {
        "train": path + "train.jsonl",
        "test": path + "test.jsonl",
        "dev": path + "dev.jsonl"
    }
    print("Starting to load data...")
    dataset = load_dataset("json", data_files=data_files)
    print("Finished loading")
    def extract_labels(file_path):
        labels = set()  # Using a set to store unique labels
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                labels.add(data["label"])
        return labels

    train_labels = extract_labels(f'data/{dataset_name}/train.jsonl')
    dev_labels = extract_labels(f'data/{dataset_name}/dev.jsonl')
    test_labels = extract_labels(f'data/{dataset_name}/test.jsonl')

    all_labels = train_labels.union(dev_labels).union(test_labels)
    print("All unique labels in the dataset:", all_labels)
    label_encoder = {label: idx for idx, label in enumerate(all_labels)}
    print("Label Encoder:", label_encoder)
    num_labels = len(label_encoder)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    max_length=max_length

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(batch["text"], max_length=max_length, truncation=True, padding="max_length")

    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)

    #dataset['train'][0]

    def convert_labels_to_integers(example):
        return {'label': label_encoder[example['label']]}

    dataset = dataset.map(convert_labels_to_integers)

    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")

    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset, num_labels


def train_task(dataset_name, dataset, model_name, model_type, num_labels, ft_lr=2e-5, ad_lr=5e-4, seed=42, max_length=80):

    """# Define common Functions"""
    def compute_f1(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"macro_f1": f1_score(p.label_ids, preds, average='macro')}


    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": accuracy_score(p.label_ids, preds)}

    #We will first use fine-tuning to perform the given task using the pretrained model
    # Set up the model for fine-tuning
    fine_tuning_model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
        )

    # Training arguments
    fine_tuning_training_args = TrainingArguments(
        learning_rate=ft_lr,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=f"./{seed}_{dataset_name}_{model_type}_ft_logs",
        logging_steps=100,
        warmup_steps=500,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        output_dir=f"./{seed}_{dataset_name}_{model_type}_ft_output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        seed=seed
    )

    # Instantiate the base model using Trainer
    fine_tune_trainer = Trainer(
        model=fine_tuning_model,
        args=fine_tuning_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_f1
    )

    # Fine tune the base model
    fine_tune_trainer.train()

    """# Using Adapter instead of fine-tuning for the same task"""

    # Now we will create an adatper for the same task.
    # The adapter will freeze the parameters of the base model implicitly.
    # Only tune the adapter layers and the final classification layer.

    config = RobertaConfig.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    adapter_model = RobertaAdapterModel.from_pretrained(model_name,config=config)

    # Add a new adapter
    adapter_name = model_type+"_"+dataset_name
    adapter_model.delete_adapter(adapter_name)
    # Available config for adapters are listed on the right.
    # The old config are listed on the left. These are from adapter-transformers library
    # The new adapters library supports both, but prefers the configs on the right
    #houlsby -> double_seq_bn
    #pfeiffer -> seq_bn
    #parallel-> par_seq_bn
    #houlsby+inv -> double_seq_bn_inv
    #pfeiffer+inv-> seq_bn_inv
    adapter_model.add_adapter(adapter_name, config="seq_bn")

    # Add a matching classification head
    adapter_model.add_classification_head(
        adapter_name,
        num_labels=num_labels,
        overwrite_ok=True
    )

    # Activate the adapter
    adapter_model.train_adapter(adapter_name)

    #Set the training arguments for the adatper

    adapter_training_args = TrainingArguments(
        learning_rate=ad_lr,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir=f"./{seed}_{dataset_name}_{model_type}_adapter_logs",
        warmup_steps=500,
        logging_steps=10,
        output_dir=f"./{seed}_{dataset_name}_{model_type}_adapter_output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        seed=seed
    )

    # We are using the trainer from the adapters library. You can try changing it
    # to the transformers function Trainer
    adapter_trainer = AdapterTrainer(
        model=adapter_model,
        args=adapter_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_f1,
    )

    adapter_trainer.train()

    """# Compare the performance of the base fine-tuned model and the adapter."""

    # Evaluate the fine tuning model
    fine_tune_results = fine_tune_trainer.evaluate(dataset["test"])

    # Evaluate the adapter model
    adapter_results = adapter_trainer.evaluate(dataset["test"])

    print(fine_tune_results)
    print(adapter_results)

    print("Fine Tune Model Accuracy:", fine_tune_results['eval_macro_f1'])
    print("Adapter Model Accuracy:", adapter_results['eval_macro_f1'])
    return fine_tune_results['eval_macro_f1'], adapter_results['eval_macro_f1']


def set_seed(seed):
    np.random.seed(seed)
    transformers_set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_process(dataset_name, seed):
    print('Starting process')
    dataset, num_labels = setup_dataset(dataset_name)
    results = []


    # For each seed we will store their evaluation results for comparison
    set_seed(seed)
    model_type = 'base'
    model_name = task_models[dataset_name][model_type]
    ft_result = train_task(dataset_name=dataset_name, dataset=dataset, model_name=model_name, model_type=model_type,
                                      num_labels=num_labels, ft_lr=1.1e-5, ad_lr=7.2e-4, seed=seed)
    results.append([seed, dataset_name, model_type, ft_result, ad_result])
    model_type = 'dapt'
    model_name = task_models[dataset_name][model_type]
    ft_result, ad_result = train_task(dataset_name=dataset_name, dataset=dataset, model_name=model_name, model_type=model_type,
                                        num_labels=num_labels, ft_lr=1.2e-5, ad_lr=7e-4, seed=seed)
    results.append([seed, dataset_name, model_type, ft_result, ad_result])
    model_type = 'tapt'
    model_name = task_models[dataset_name][model_type]
    ft_result, ad_result = train_task(dataset_name=dataset_name, dataset=dataset,
                                        model_name=model_name,
                                        model_type=model_type, num_labels=num_labels, ft_lr=1.1e-5, ad_lr=5e-4,
                                        seed=seed)
    results.append([seed, dataset_name, model_type, ft_result, ad_result])
    model_type = 'dapt_tapt'
    model_name = task_models[dataset_name][model_type]
    ft_result, ad_result = train_task(dataset_name=dataset_name, dataset=dataset,
                                        model_name=model_name,
                                        model_type=model_type, num_labels=num_labels, ft_lr=1e-5, ad_lr=7.2e-4,
                                        seed=seed)
    results.append([seed, dataset_name, model_type, ft_result, ad_result])
    df = pd.DataFrame(results, columns=['Seed', 'Dataset', 'Model', 'Fine Tune Result', 'Adapter Result'])
    df['Seed'] = df['Seed'].astype(int)
    df['Fine Tune Result'] = df['Fine Tune Result'].astype(float)
    df['Adapter Result'] = df['Adapter Result'].astype(float)
    df.to_csv(f'{seed}_{dataset_name}_results.csv', index=False)


if __name__ == "__main__":
    print('Starting process')

