from adapters import AutoAdapterModel, RobertaAdapterModel, AdapterTrainer
import adapters.composition as ac
import adapters
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

# Our built utilities
from utils import yaml_utils, compute_metrics #mlops,
from data_loaders.citation_intent_data_loader import CSTasksDataLoader
from demo.adapter_fusion_utils import AdapterCreator

# ======================================================
# CREATE THE ADAPTERS
# ======================================================

# citation_adapter_creator = AdapterCreator(
#     model_type = 'base',
#     model_name = 'roberta-base',
#     dataset_name = 'citation_intent',
#     seed = 1,
#     ad_lr=7.2e-4
# )
# citation_adapter_model = citation_adapter_creator.create_and_save_adapter()

# sciie_adapter_creator = AdapterCreator(
#     model_type = 'base',
#     model_name = 'roberta-base',
#     dataset_name = 'sciie',
#     seed = 1,
#     ad_lr=7.2e-4
# )
# sciie_adapter_model = sciie_adapter_creator.create_and_save_adapter()
dataset_name_shortener = {
    'citation_intent': 'ci',
    'sciie': 'sci'
}


# Create Fusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# loader = CSTasksDataLoader(model_name="roberta-base",
#                                   dataset_name="citation_intent",
#                                   path=f"data/citation_intent/",
#                                   checkpoint_path="data/citation_intent/processed_dataset.pt")

# Locked
citation_intent_adapter_name = 'adapter_base_citation_intent'
citation_intent_adapter_name_short = 'ad_ci'
sciie_adapter_name = 'adapter_base_sciie'
sciie_adapter_name_short = 'ad_sci'

# Variables
model_type = 'base'
model_name = 'roberta-base'
dataset_name = 'sciie'
dataset_name_short = dataset_name_shortener[dataset_name]
seed = 1
ad_lr=5e-5

loader = CSTasksDataLoader(model_name=model_name,
                            dataset_name=dataset_name,
                            path=f"data/{dataset_name}/",
                            checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")
dataset = loader.load_dataset(overwrite=False)

config = RobertaConfig.from_pretrained(
    model_name,
    num_labels=loader.num_labels,
    problem_type="single_label_classification",
    hidden_dropout_prob=0.1,
)

fusion_name = model_type+"_"+dataset_name
fusion_model = RobertaAdapterModel.from_pretrained(model_name,config=config)
fusion_model.to(device)

fusion_model.load_adapter(f'./saved/{citation_intent_adapter_name}', load_as=citation_intent_adapter_name_short, with_head=False)
fusion_model.load_adapter(f'./saved/{sciie_adapter_name}', load_as=sciie_adapter_name_short, with_head=False)
adapter_setup = [
    [
        citation_intent_adapter_name_short,
        sciie_adapter_name_short,
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

fusion_model.train_adapter_fusion(ac.Fuse(citation_intent_adapter_name_short, sciie_adapter_name_short))

fusion_training_args = TrainingArguments(
    learning_rate=ad_lr,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir=f"./saved/{seed}_{dataset_name}_{model_type}_fn_logs_lrlow",
    warmup_steps=500,
    logging_steps=10,
    output_dir=f"./saved/{seed}_{dataset_name}_{model_type}_fn_out_lrlow",
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

# # adapter_model.save_adapter(f'./adapter{seed}_{dataset_name}_{model_type}',adapter_name)
# adapter_model.save_adapter(f'./saved/{adapter_name}',adapter_name)

# print('beginning reload')
# adapter_name_test = model_type+"_"+dataset_name+'_test'
# adapter_model_test = RobertaAdapterModel.from_pretrained(model_name,config=config)
# adapter_model_test.to(device)
# # adapters.init(adapter_model_test)
# adapter_model_test.load_adapter(f'./saved/{adapter_name}', with_head=True)
# adapter_model_test.set_active_adapters(adapter_name)

# adapter_trainer_test = AdapterTrainer(
#     model=adapter_model_test,
#     args=adapter_training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["dev"],
#     compute_metrics=compute_metrics.macro_f1,
# )
# print("testing reload of adapter")
# test_results = adapter_trainer_test.evaluate(dataset["test"])
# print(test_results)
# print("Test Macro-F1 Score:", test_results['eval_macro_f1'])