from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

# Our built utilities
from utils import yaml_utils, mlops, compute_metrics
from data_loaders.citation_intent_data_loader import CitationIntentDataLoader

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
    problem_type="single_label_classification",
    hidden_dropout_prob=0.1,
)

"""
Run different pre-trained models by changing the model_variant variable.
We will then build a classification head for the model.
e.g. 
    roberta-base is the base pre-trained model.
    ./pretrained_models/tapt_mlm_model is our pre-trained roberta-base with tapt data.
    ./pretrained_models/dapt_tapt_mlm_model is our pre-trained cs_roberta_base with tapt data.
    allenai/cs_roberta_base is the published (2020) pre-trained with dapt data.
    allenai/dsp_roberta_base_tapt_citation_intent_1688 is the published (2020) pre-trained with tapt data.
"""
# model_variant = model_name
# model_variant = './pretrained_models/tapt_mlm_model'
# model_variant = './pretrained_models/dapt_tapt_mlm_model'
# model_variant = 'allenai/cs_roberta_base'
model_variant = "allenai/dsp_roberta_base_tapt_citation_intent_1688" 

model = RobertaForSequenceClassification.from_pretrained(model_variant, config=config)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

training_kwargs = yaml_utils.load_training_args_from_yaml('cs_7643_efficiencylane/training_configs/classifier_head.yaml')
training_args = TrainingArguments(**training_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics.macro_f1
)

# Start training
trainer.train()
evaluation_results = trainer.evaluate()
print(evaluation_results)
print("Test Macro-F1 Score:", evaluation_results['eval_macro_f1'])
