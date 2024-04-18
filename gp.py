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
from sklearn.metrics import accuracy_score, f1_score
from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import matplotlib.pyplot as plt

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


def train_task(dataset, model_name, model_type, num_labels, ft_lr=2e-5, ad_lr=5e-4, max_length=80):

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
        logging_dir=f"./{dataset_name}_{model_type}_ft_logs",
        logging_steps=100,
        warmup_steps=500,
        eval_steps=100,
        load_best_model_at_end=True,
        output_dir=f"./{dataset_name}_{model_type}_ft_output",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
    )

    # Instantiate the base model using Trainer
    fine_tune_trainer = Trainer(
        model=fine_tuning_model,
        args=fine_tuning_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_f1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
    adapter_name = model_name.replace("/", "_")+"_"+dataset_name
    adapter_model.delete_adapter(adapter_name)
    # Available config for adapters are listed on the right.
    # The old config are listed on the left. These are from adapter-transformers library
    # The new adapters library supports both, but prefers the configs on the right
    #houlsby -> double_seq_bn
    #pfeiffer -> seq_bn
    #parallel-> par_seq_bn
    #houlsby+inv -> double_seq_bn_inv
    #pfeiffer+inv-> seq_bn_inv
    adapter_model.add_adapter(adapter_name, config="double_seq_bn")

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
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir=f"./{dataset_name}_{model_type}_adapter_logs",
        logging_steps=10,
        output_dir=f"./{dataset_name}_{model_type}_adapter_output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    # We are using the trainer from the adapters library. You can try changing it
    # to the transformers function Trainer
    adapter_trainer = AdapterTrainer(
        model=adapter_model,
        args=adapter_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
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

    # Data to plot
    labels = ['Fine Tune Model', 'Adapter Model']
    accuracies = [fine_tune_results['eval_macro_f1'], adapter_results['eval_macro_f1']]

    # Creating the bar plot
    plt.bar(labels, accuracies, color=['blue', 'green'])
    plt.xlabel('Model Type')
    plt.ylabel('F1 Score')
    plt.title('Comparison of Model F1 scores')
    plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
    plt.savefig(f'{dataset_name}_{model_type}.png')
    plt.close()

if __name__ == "__main__":
    print('Starting process')
    dataset_name = 'citation_intent'
    dataset, num_labels = setup_dataset(dataset_name)
    train_task(dataset=dataset, model_name='roberta-base', model_type='base',num_labels=num_labels, ft_lr=1e-5, ad_lr=5.2e-4)
    train_task(dataset=dataset, model_name='allenai/cs_roberta_base',model_type='dapt',num_labels=num_labels, ft_lr=1.4e-5, ad_lr=6e-4)
    train_task(dataset=dataset, model_name='allenai/dsp_roberta_base_tapt_citation_intent_1688',model_type='tapt',num_labels=num_labels, ft_lr=1e-5, ad_lr=5.5e-4)
    train_task(dataset=dataset, model_name='allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688',model_type='dapt_tapt',num_labels=num_labels, ft_lr=1e-5, ad_lr=5e-4)