"""
This is adapted code from https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=huLjPAKHLA1g
to use the newer adapters library and to work with the Rotten Tomatoes dataset.
"""
import os
import json
import torch
from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaTokenizer, RobertaConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from transformers import TrainingArguments, EvalPrediction, TextClassificationPipeline
from adapters import AdapterTrainer
from data_loaders.citation_intent_data_loader import CitationIntentDataLoader
import mlflow
from sklearn.metrics import f1_score

# ======================================================
# Set-up and Load Data
# ======================================================
model_name = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = CitationIntentDataLoader(model_name="roberta-base",
                                  dataset_name="citation_intent",
                                  path=f"data/citation_intent/",
                                  checkpoint_path="data/citation_intent/processed_dataset.pt")

dataset = loader.load_dataset(overwrite=False)

# ======================================================
# Model Config & Training
# ======================================================

# Set up training for the Model and Adapter
config = RobertaConfig.from_pretrained(
    model_name,
    num_labels=loader.num_labels,
)

model = RobertaAdapterModel.from_pretrained(model_name, config=config)#.to(device)

# Add a new adapter and a matching classification head
adapter_name = model_name+"_"+loader.dataset_name
model.add_adapter(adapter_name, config="pfeiffer") #alternatively, config="lora")
model.add_classification_head(
    adapter_name,
    num_labels=loader.num_labels
  )

#  The train_adapter() method does two things:
#     It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
#     It activates the adapter and the prediction head such that both are used in every forward pass.
# Activate the adapter
model.train_adapter(adapter_name)


def objective(trial):
    # Suggest values for the learning rate and batch size
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    train_epochs = trial.suggest_int('train_epochs', 10, 100)

    # Get the trial number and define output dir
    trial_number = trial.number
    study_name = trial.study.study_name
    output_dir = f"./training_output/{study_name}/trial_{trial_number}/"

    # Note the differences in hyperparameters compared to full fine-tuning. Adapter training usually requires a few more training epochs than full fine-tuning.
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=200, #Log every 200 training steps (i.e., this is why we see decimal epochs)
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        metric_for_best_model="eval_loss"
        # evaluation_strategy="epoch",
    )

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)

        # label_ids = p.label_ids
        # unique_label_ids = np.unique(label_ids)
        # for label_id in unique_label_ids:
        #     total_occurrences = np.sum(label_ids == label_id)
        #     correct_predictions = np.sum((label_ids == label_id) & (preds == label_ids))
        #     accuracy = correct_predictions / total_occurrences
        #     print(f"Label_id {label_id}: Accuracy = {accuracy:.2f}, Total Occurance = {total_occurrences}")

        return {"acc": (preds == p.label_ids).mean()}

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"macro_f1": f1_score(p.label_ids, preds, average='macro')}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Start training and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Trial finished: Loss {eval_results['eval_loss']}, F1: {eval_results['eval_macro_f1']}")

    mlflow.log_metrics({"loss": eval_results['eval_loss'], "F1": eval_results['eval_macro_f1']})

    # Depending on what you want to optimize for:
    # return trainer.evaluate()['eval_loss'] # For loss minimization
    return -eval_results['eval_macro_f1']  # For accuracy maximization (note the negative sign)

import optuna


study_name = f"{adapter_name}_training-7"
storage = "sqlite:///db.sqlite3"

try:
    # Try loading the study
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Study '{study_name}' loaded from storage.")
except KeyError:
    # If the study does not exist, create a new one
    study = optuna.create_study(study_name=study_name, storage=storage, direction='minimize')
    print(f"Study '{study_name}' created.")

study.optimize(objective, n_trials=10)
print("Best trial:", study.best_trial.params)
#Best trial: {'learning_rate': 0.0002183987833471655, 'batch_size': 16}

# import mlflow

# mlflow.start_run()
# mlflow.log_params(study.best_trial.params)
# mlflow.end_run()

# import hydra
# from omegaconf import DictConfig
# @hydra.main(config_path='.', config_name='config')
# def train_model(cfg: DictConfig):
#     # Use cfg.learning_rate and cfg.batch_size in your TrainingArguments
#     pass


# Test the model
classifier = TextClassificationPipeline(model=model, tokenizer=loader.tokenizer, device=0)
classifier("We use the same set of binary features as in previous work on this dataset ( Pang et al. , 2002 ; Pang and Lee , 2004 ; Zaidan et al. , 2007 ) .")

model.save_adapter("./final_adapter", adapter_name)


# Check final size of the adapter
# !ls -lh final_adapter

# Share Adapter to adapter hub
# model.push_adapter_to_hub(
#     "my-awesome-adapter",
#     "rotten_tomatoes",
#     adapterhub_tag="sentiment/rotten_tomatoes",
#     datasets_tag="rotten_tomatoes"
# )


1==1

