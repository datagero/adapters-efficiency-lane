from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import adapters.composition as ac
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

import csv

# Our built utilities
from utils import yaml_utils, compute_metrics 
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
            batch_size=32,
            epochs=10,
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.ad_lr= ad_lr
        self.batch_size = batch_size
        self.epochs = epochs
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
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir=f"./{self.seed}_{self.dataset_name}_{self.model_type}_adapter_logs",
            # warmup_steps=500,
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
        csv_data = [adapter_name, evaluation_results['eval_macro_f1'], test_results['eval_macro_f1']]
        with open('./saved/test_results.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(csv_data)
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


class FusionCreator:
    def __init__(
            self,
            model_type='base',
            model_name='roberta-base',
            dataset_name='citation_intent',
            adapter1_name = 'adapter_base_citation_intent',
            adapter2_name = 'adapter_base_sciie',
            seed=1,
            ad_lr=7.2e-4,
            warmup_steps=0,
            batch_size=32,
            epochs=10,
            trial=1,
                 ):
        self.model_type = model_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.adapter1_name = adapter1_name
        self.adapter2_name = adapter2_name
        self.seed = seed
        self.ad_lr= ad_lr
        self.warmup_steps=warmup_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.trial=trial
        self.loader = CSTasksDataLoader(model_name=model_name,
                                  dataset_name=dataset_name,
                                  path=f"data/{dataset_name}/",
                                  checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")

        self.dataset = self.loader.load_dataset(overwrite=False)
        

    def create_and_save_adapter(self):     
        dataset_name_shortener = {
            'citation_intent': 'ci',
            'sciie': 'sci'
        }

        adapter_name_shortener = {
            'adapter_base_citation_intent': 'ad_bs_ci',
            'adapter_base_sciie': 'ad_bs_sci',
            'adapter_cs_citation_intent': 'ad_cs_ci',
            'adapter_cs_sciie': 'ad_cs_sci',    
        }
        adapter_name_full_name = {
            'adapter_base_citation_intent': 'roberta-base_citation_intent_seq_bn_training_adapter_v01_best/trial_1/seed_128',
            'adapter_base_sciie': 'roberta-base_sciie_seq_bn_training_adapter_v01_best/trial_1/seed_42',
            'adapter_cs_citation_intent': 'cs_roberta_base_citation_intent_seq_bn_training_adapter_v01_best/trial_2/seed_128',
            'adapter_cs_sciie': 'cs_roberta_base_sciie_seq_bn_training_adapter_v01_best/trial_2/seed_42',
        }

        # Locked
        adapter1_name_short = adapter_name_shortener[self.adapter1_name]
        adapter2_name_short = adapter_name_shortener[self.adapter2_name]
        adapter1_name_full = adapter_name_full_name[self.adapter1_name]
        adapter2_name_full = adapter_name_full_name[self.adapter2_name]

        # Create Fusion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = CSTasksDataLoader(model_name=self.model_name,
                                    dataset_name=self.dataset_name,
                                    path=f"data/{self.dataset_name}/",
                                    checkpoint_path=f"data/{self.dataset_name}/processed_dataset.pt")
        dataset = loader.load_dataset(overwrite=False)

        config = RobertaConfig.from_pretrained(
            self.model_name,
            num_labels=loader.num_labels,
            problem_type="single_label_classification",
            hidden_dropout_prob=0.1,
        )

        fusion_name = self.model_type+"_"+dataset_name_shortener[self.dataset_name]
        fusion_model = RobertaAdapterModel.from_pretrained(self.model_name,config=config)
        fusion_model.to(device)

        fusion_model.load_adapter(f'./saved/{adapter1_name_full}', load_as='ad1', with_head=False)
        fusion_model.load_adapter(f'./saved/{adapter2_name_full}', load_as='ad2', with_head=False)
        adapter_setup = [
            [
                'ad1',
                'ad2',
            ]
        ]
        fusion_model.add_adapter_fusion(adapter_setup[0], "dynamic")
        fusion_model.train_adapter_fusion(adapter_setup)
        # fusion_model.add_adapter_fusion(ac.Fuse(citation_intent_adapter_name_short, sciie_adapter_name_short))
        # fusion_model.set_active_adapters(ac.Fuse(citation_intent_adapter_name_short, sciie_adapter_name_short))


        # Add a matching classification head
        fusion_model.add_classification_head(
            fusion_name,
            num_labels=loader.num_labels,
            overwrite_ok=True
        )

        fusion_model.train_adapter_fusion(ac.Fuse('ad1', 'ad2'))

        fusion_training_args = TrainingArguments(
            learning_rate=self.ad_lr,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_dir=f"./{self.seed}_{self.dataset_name}_{self.model_type}_fn_logs_{self.trial}",
            warmup_steps=self.warmup_steps,
            logging_steps=10,
            output_dir=f"./{self.seed}_{self.dataset_name}_{self.model_type}_fn_out_{self.trial}",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # lr_scheduler_type='constant',
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            seed=self.seed
        )

        # We are using the trainer from the adapters library. You can try changing it
        # to the transformers function Trainer
        fusion_trainer = AdapterTrainer(
            model=fusion_model,
            args=fusion_training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            compute_metrics=compute_metrics.macro_f1,
        )
        print("SKLHTEISOJFISEJFLISDFLKJSKLFJTHRLRRARRRAIONOINOINONOINOIN")
        fusion_trainer.train()
        evaluation_results = fusion_trainer.evaluate()
        test_results = fusion_trainer.evaluate(dataset["test"])
        print(evaluation_results)
        print("Test Macro-F1 Score:", evaluation_results['eval_macro_f1'])
        print(test_results)
        print("Test Macro-F1 Score:", test_results['eval_macro_f1'])
        csv_data = [f'{self.seed}_{fusion_name}_trial_{self.trial}_lr_{self.ad_lr}_bs_{self.batch_size}_ep_{self.epochs}_{adapter1_name_short}{adapter2_name_short}', evaluation_results['eval_macro_f1'], test_results['eval_macro_f1']]
        with open('./saved/test_results.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(csv_data)
