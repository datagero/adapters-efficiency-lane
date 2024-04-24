from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import adapters
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

# Our built utilities
from utils import yaml_utils, compute_metrics #mlops,
from data_loaders.citation_intent_data_loader import CSTasksDataLoader

# ======================================================
# Set-up and Load Data
# ======================================================

class AdapterCreator:
    def __init__(
            self,
            model_type='base',
            model_name='roberta-base',
            dataset_name='citation_intent',
            seed=1,
            ad_lr=7.2e-4,   
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.ad_lr= ad_lr
        self.loader = CSTasksDataLoader(model_name=model_name,
                                  dataset_name=dataset_name,
                                  path=f"data/{dataset_name}/",
                                  checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")

        self.dataset = self.loader.load_dataset(overwrite=False)


    def create_and_save_adapter(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)


        config = RobertaConfig.from_pretrained(
            self.model_name,
            num_labels=self.loader.num_labels,
            problem_type="single_label_classification",
            hidden_dropout_prob=0.1,
        )

        adapter_name = 'adapter_'+self.model_type+"_"+self.dataset_name
        adapter_model = RobertaAdapterModel.from_pretrained(self.model_name,config=config)
        adapter_model.to(device)
        adapter_model.add_adapter(adapter_name, config="seq_bn")

        # Add a matching classification head
        adapter_model.add_classification_head(
            adapter_name,
            num_labels=self.loader.num_labels,
            overwrite_ok=True
        )
        adapter_model.train_adapter(adapter_name)

        adapter_training_args = TrainingArguments(
            learning_rate=self.ad_lr,
            num_train_epochs=10,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            logging_dir=f"./{self.seed}_{self.dataset_name}_{self.model_type}_adapter_logs",
            warmup_steps=500,
            logging_steps=10,
            output_dir=f"./{self.seed}_{self.dataset_name}_{self.model_type}_adapter_output",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            seed=self.seed
        )

        # We are using the trainer from the adapters library. You can try changing it
        # to the transformers function Trainer
        adapter_trainer = AdapterTrainer(
            model=adapter_model,
            args=adapter_training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["dev"],
            compute_metrics=compute_metrics.macro_f1,
        )

        adapter_trainer.train()
        evaluation_results = adapter_trainer.evaluate()
        test_results = adapter_trainer.evaluate(self.dataset["test"])
        print("Evaluation Results-------------")
        print(evaluation_results)
        print("Test Macro-F1 Score:", evaluation_results['eval_macro_f1'])
        print("Test Results-------------------")
        print(test_results)
        print("Test Macro-F1 Score:", test_results['eval_macro_f1'])

        # adapter_model.save_adapter(f'./adapter{seed}_{dataset_name}_{model_type}',adapter_name)
        adapter_model.save_adapter(f'./saved/{adapter_name}',adapter_name)
        return adapter_model
    

        # print('beginning reload')
        # adapter_name_test = self.model_type+"_"+self.dataset_name+'_test'
        # adapter_model_test = RobertaAdapterModel.from_pretrained(self.model_name,config=config)
        # adapter_model_test.to(device)
        # # adapters.init(adapter_model_test)
        # adapter_model_test.load_adapter(f'./saved/{adapter_name}', with_head=True)
        # adapter_model_test.set_active_adapters(adapter_name)

        # adapter_trainer_test = AdapterTrainer(
        #     model=adapter_model_test,
        #     args=adapter_training_args,
        #     train_dataset=self.dataset["train"],
        #     eval_dataset=self.dataset["dev"],
        #     compute_metrics=compute_metrics.macro_f1,
        # )
        # print("testing reload of adapter")
        # test_results = adapter_trainer_test.evaluate(self.dataset["test"])
        # print(test_results)
        # print("Test Macro-F1 Score:", test_results['eval_macro_f1'])