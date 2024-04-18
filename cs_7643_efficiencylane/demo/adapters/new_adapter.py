"""
This is adapted code from https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=huLjPAKHLA1g
"""
import os
import torch
from omegaconf import DictConfig 
from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaConfig
from transformers import TextClassificationPipeline
from adapters import AdapterTrainer
import hydra

# Our built utilities
from data_loaders.citation_intent_data_loader import CitationIntentDataLoader
from utils import mlops, compute_metrics

@hydra.main(config_path='../../training_configs', config_name='adapter_citation_intent')
def main(cfg: DictConfig):
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

    model = RobertaAdapterModel.from_pretrained(model_name, config=config)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
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

        training_args = mlops.build_training_arguments_for_trial(trial, cfg)

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics.macro_f1,
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Trial finished: Loss {eval_results['eval_loss']}, F1: {eval_results['eval_macro_f1']}")

        # Depending on what you want to optimize for:
        # return trainer.evaluate()['eval_loss'] # For loss minimization
        return -eval_results['eval_macro_f1']  # For accuracy maximization (note the negative sign)


    # Trials: During the optimization process, Optuna conducts multiple trials, 
    # each time evaluating the objective function with a different set of hyperparameters. 
    # These trials could be run sequentially or in parallel.
    # We could define an optimization algorithm for hyperparameter search. e.g., TPE, Random Search, Grid Search, etc.
    study_name = f"{adapter_name}_training-test-3"
    storage = "sqlite:///db.sqlite3"
    study = mlops.create_or_load_study(study_name, storage, direction='minimize')
    study.optimize(objective, n_trials=2)
    print("Best trial:", study.best_trial.params)

    # Test the model
    classifier = TextClassificationPipeline(model=model, tokenizer=loader.tokenizer, device=0)
    classifier("We use the same set of binary features as in previous work on this dataset ( Pang et al. , 2002 ; Pang and Lee , 2004 ; Zaidan et al. , 2007 ) .")

    out_fldr = "./adapters"
    if not os.path.exists(out_fldr):
        os.makedirs(out_fldr)

    model.save_adapter(f"{out_fldr}/adapter_citation_intent", adapter_name)

if __name__ == "__main__":
    main()


