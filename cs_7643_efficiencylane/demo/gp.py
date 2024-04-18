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

# Our built utilities
from data_loaders.citation_intent_data_loader import CitationIntentDataLoader
from utils.yaml_utils import load_training_args_from_yaml
from utils.compute_metrics import macro_f1

"""# Set up the dataset and the model"""
def setup_dataset(dataset_name, model_name='roberta-base', max_length=80):
    # Note, currently this only works with citation_intent, but more classes can be developed for different datasets
    if dataset_name == 'citation_intent':
        loader = CitationIntentDataLoader(model_name="roberta-base",
                                        dataset_name=dataset_name,
                                        path=f"data/{dataset_name}/",
                                        checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")
    else:
        raise ValueError("Only citation_intent dataset is supported for this demo")


    dataset = loader.load_dataset(overwrite=False)
    num_labels = loader.num_labels
    return dataset, num_labels


def train_task(dataset, model_name, model_type, num_labels, ft_lr=2e-5, ad_lr=5e-4, max_length=80):

    #We will first use fine-tuning to perform the given task using the pretrained model
    # Set up the model for fine-tuning
    fine_tuning_model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
        )

     # Training arguments
    training_kwargs = load_training_args_from_yaml('cs_7643_efficiencylane/training_configs/gp_configs/classifier_training.yaml')
    fine_tuning_training_args = TrainingArguments(**training_kwargs)

    fine_tuning_training_args.learning_rate = ft_lr
    fine_tuning_training_args.logging_dir = f"./training_output/gp/{dataset_name}_{model_type}_ft_logs"
    fine_tuning_training_args.output_dir = f"./training_output/gp/{dataset_name}_{model_type}_ft_output" # Save model to output_dir

    # Instantiate the base model using Trainer
    fine_tune_trainer = Trainer(
        model=fine_tuning_model,
        args=fine_tuning_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=macro_f1,
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
    adapter_training_kwargs = load_training_args_from_yaml('cs_7643_efficiencylane/training_configs/gp_configs/adapter_training.yaml')
    adapter_training_args = TrainingArguments(**adapter_training_kwargs)

    adapter_training_args.learning_rate = ft_lr
    adapter_training_args.logging_dir = f"./training_output/gp/{dataset_name}_{model_type}_adapter_logs"
    adapter_training_args.output_dir = f"./training_output/gp/{dataset_name}_{model_type}_adapter_output" # Save model to output_dir

    # We are using the trainer from the adapters library. You can try changing it
    # to the transformers function Trainer
    adapter_trainer = AdapterTrainer(
        model=adapter_model,
        args=adapter_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=macro_f1,
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